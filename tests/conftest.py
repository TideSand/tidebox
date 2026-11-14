"""Shared test helpers: servers on background threads, a scripted fake provider, and a mini
harness that behaves like a coding agent's tool loop."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import urllib.request
from collections.abc import Callable
from typing import ClassVar

import pytest
from aiohttp import web

from tidebox.api.providers.base import sse_chunk


class Served:
    def __init__(self, app: web.Application) -> None:
        self.app = app
        self.loop = asyncio.new_event_loop()
        self.port: int | None = None
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self._ready.wait(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.runner = web.AppRunner(self.app, shutdown_timeout=2.0)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self._ready.set()
        self.loop.run_forever()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if not self.loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop)
        try:
            fut.result(15)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(10)


@pytest.fixture
def serve():
    served: list[Served] = []

    def _serve(app: web.Application) -> Served:
        s = Served(app)
        served.append(s)
        return s

    yield _serve
    for s in reversed(served):  # clients before the servers they hold connections to
        s.stop()


# -- a fake Anthropic provider ---------------------------------------------------------------


def anthropic_message(blocks: list[dict], stop_reason: str = "end_turn") -> dict:
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "fake", "content": blocks, "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 2}}


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, **input_) -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def to_sse(message: dict) -> list[str]:
    head = {k: v for k, v in message.items() if k not in ("content", "stop_reason", "stop_sequence")}
    head["content"] = []
    out = [sse_chunk({"type": "message_start", "message": head}, "message_start")]
    for i, b in enumerate(message["content"]):
        if b["type"] == "text":
            out.append(sse_chunk({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}, "content_block_start"))
            half = len(b["text"]) // 2
            for piece in (b["text"][:half], b["text"][half:]):
                out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": piece}}, "content_block_delta"))
        else:
            out.append(sse_chunk({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}}, "content_block_start"))
            js = json.dumps(b["input"])
            out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": js[:3]}}, "content_block_delta"))
            out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": js[3:]}}, "content_block_delta"))
        out.append(sse_chunk({"type": "content_block_stop", "index": i}, "content_block_stop"))
    out.append(sse_chunk({"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": 2}}, "message_delta"))
    out.append(sse_chunk({"type": "message_stop"}, "message_stop"))
    return out


def fake_anthropic(script: Callable[[dict], dict]) -> web.Application:
    """A provider whose answer to each request is `script(body)`: a complete message dict."""

    async def messages(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if request.headers.get("x-api-key") != "test-key":
            return web.json_response({"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}, status=401)
        message = script(body)
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "request-id": "req_fake"})
            await resp.prepare(request)
            for chunk in to_sse(message):
                await resp.write(chunk.encode())
                await asyncio.sleep(0.005)
            await resp.write_eof()
            return resp
        return web.json_response(message, headers={"request-id": "req_fake"})

    async def count(request: web.Request) -> web.Response:
        return web.json_response({"input_tokens": 42})

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/messages/count_tokens", count)
    return app


# -- a mini harness: the tool loop of a coding agent, driven by any Messages API server ---------


class MiniHarness:
    """Sends a conversation to a Messages API endpoint, runs Bash calls for real in `cwd`,
    and nests a conversation for Agent calls. Returns the final text of the main conversation."""

    TOOLS: ClassVar[list[dict]] = [{"name": "Bash", "input_schema": {"type": "object"}}, {"name": "Agent", "input_schema": {"type": "object"}}, {"name": "TodoWrite", "input_schema": {"type": "object"}}]

    def __init__(self, base_url: str, cwd: str, stream: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.cwd = cwd
        self.stream = stream
        self.calls: list[dict] = []
        self.results: list[dict] = []

    def call_model(self, messages: list[dict]) -> dict:
        body = {"model": "fake", "max_tokens": 100, "system": "you are a test agent", "messages": messages, "tools": self.TOOLS, "stream": self.stream}
        req = urllib.request.Request(self.base_url + "/v1/messages", data=json.dumps(body).encode(), headers={"content-type": "application/json", "x-api-key": "test-key", "anthropic-version": "2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        self.calls.append(body)
        if self.stream:
            from tidebox.api.providers.anthropic import assemble_stream
            from tidebox.api.providers.base import sse_events
            return assemble_stream(sse_events(raw))
        return json.loads(raw)

    def run_tool(self, block: dict) -> dict:
        if block["name"] == "Bash":
            p = subprocess.run(["/bin/bash", "-c", block["input"]["command"]], check=False, cwd=self.cwd, capture_output=True, text=True)
            out = p.stdout.rstrip("\n") + (("\n" + p.stderr.rstrip("\n")) if p.stderr else "")
            if p.returncode != 0:
                out = f"Exit code {p.returncode}\n{out}" if out else f"Exit code {p.returncode}"
            return {"type": "tool_result", "tool_use_id": block["id"], "content": out, "is_error": p.returncode != 0}
        if block["name"] == "Agent":
            final = self.run_conversation(block["input"]["prompt"])
            return {"type": "tool_result", "tool_use_id": block["id"], "content": final or ""}
        return {"type": "tool_result", "tool_use_id": block["id"], "content": "Todos have been modified successfully."}

    def run_conversation(self, prompt: str) -> str | None:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        final = None
        while True:
            message = self.call_model(messages)
            messages.append({"role": "assistant", "content": message["content"]})
            calls = [b for b in message["content"] if b["type"] == "tool_use"]
            texts = [b["text"] for b in message["content"] if b["type"] == "text"]
            if texts:
                final = "".join(texts)
            if not calls:
                return final
            results = [self.run_tool(b) for b in calls]
            self.results.extend(results)
            messages.append({"role": "user", "content": results})
