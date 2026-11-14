"""LLM replay (`tidebox replay --llm`): look like the provider to a real harness.

    POST [/r/<run>]/v1/messages              Anthropic
    POST [/r/<run>]/v1/chat/completions      OpenAI Chat Completions
    POST [/r/<run>]/v1/responses             OpenAI Responses

Matches the request with the matcher, serves the raw recorded response through the provider
adapter (streamed at the recorded pace when the request asks for a stream), learns identifiers
from the tool results the request carries, and hands each request/response pair to the replay
recorder when there is one. One controller serves one tape; several replays of it may run at
once, told apart by the `/r/<run>` prefix (each gets its own matcher, identifiers and replay
tape, written next to the main output as `<stem>.<run>.json`). Other paths (token counting,
model listing) get a minimal canned answer so the harness does not stall.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from aiohttp import web

from tidebox.api.providers import detect_provider, get_provider
from tidebox.api.providers.base import scrub_headers
from tidebox.api.recorder import Recorder
from tidebox.api.tape import Tape, TapeMeta, Timing
from tidebox.controller.identifiers import IdentifierMap
from tidebox.controller.matcher import Matcher
from tidebox.controller.pacing import Pacer

log = logging.getLogger("tidebox.controller")
RUN_PREFIX = re.compile(r"^/r/([^/]+)(/.*)$")


class Run:
    """One replay of the tape by one harness."""

    def __init__(self, name: str, tape: Tape, output: Path | None) -> None:
        self.name = name
        self.provider = get_provider(tape.meta.provider)
        self.matcher = Matcher(tape, self.provider)
        self.identifiers = IdentifierMap()
        self.served = 0
        self.unmatched = 0
        self.t0 = time.monotonic()
        self.recorder: Recorder | None = None
        if output is not None:
            meta = TapeMeta(
                tape_id=f"{tape.meta.tape_id}-replay" + ("" if name == "default" else f"-{name}"),
                harness=tape.meta.harness, provider=tape.meta.provider, model=tape.meta.model,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                harness_version=tape.meta.harness_version, task=dict(tape.meta.task), replay_of=tape.meta.tape_id,
                labels={**tape.meta.labels, "mode": "llm"},
            )
            path = output if name == "default" else output.with_name(f"{output.stem}.{name}{output.suffix}")
            # Spilled: a replay tape is as large as the recording, and several controllers run
            # at once, so completed steps go to disk as they finish instead of staying in memory.
            self.recorder = Recorder(path, meta, self.provider, spill=True)

    def now(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)


class Simulator:
    def __init__(self, tape: Tape, pacer: Pacer, output: Path | None = None) -> None:
        self.tape = tape
        self.pacer = pacer
        self.output = output
        self.runs: dict[str, Run] = {}
        self.in_flight = 0  # requests being served or recorded; the server waits for zero before finalizing

    def routes(self) -> list[web.RouteDef]:
        return [web.route("*", "/{path:.*}", self.handle)]

    def run(self, name: str) -> Run:
        if name not in self.runs:
            self.runs[name] = Run(name, self.tape, self.output)
        return self.runs[name]

    def state(self) -> dict:
        return {
            name: {"served": run.served, "unmatched": run.unmatched, "identifiers": len(run.identifiers.mapping)}
            for name, run in self.runs.items()
        }

    def finalize(self) -> None:
        for run in self.runs.values():
            if run.recorder is not None:
                run.recorder.finalize()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.in_flight += 1
        try:
            return await self._handle(request)
        finally:
            self.in_flight -= 1

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        path = "/" + request.match_info["path"]
        name = "default"
        m = RUN_PREFIX.match(path)
        if m:
            name, path = m.group(1), m.group(2)
        provider = detect_provider(path)
        if provider is None or request.method != "POST":
            return self._canned(path)
        run = self.run(name)
        if provider.name != run.provider.name:
            return web.json_response({"error": f"tape is {run.provider.name}, request is {provider.name}"}, status=400)
        try:
            raw = await request.read()
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return web.Response(body=provider.render_error(400, "request body is not JSON"), status=400, content_type="application/json")
        request_at = run.now()
        req = provider.parse_request(body, scrub_headers({k.lower(): v for k, v in request.headers.items()}), path)
        match = run.matcher.match(req)
        if match is None:
            run.unmatched += 1
            log.warning("no recorded step matches request %s (%s)", path, provider.conversation_key(req))
            if self.output is not None:
                # Kept for diagnosis: what the harness sent that the tape has no answer for.
                d = Path(str(self.output) + ".unmatched")
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{hashlib.sha256(raw).hexdigest()[:16]}.json").write_bytes(raw)
            return web.Response(body=provider.render_error(404, "tidebox: no recorded response matches this request"), status=404, content_type="application/json")
        thread_id, step, how = match
        thread = self.tape.thread(thread_id)
        if step.index > 0:
            recorded = {r.tool_call_id: r for r in thread.steps[step.index - 1].tool_results}
            for result in provider.tool_results(req):
                if result.tool_call_id in recorded:
                    run.identifiers.learn(recorded[result.tool_call_id], result)
        response = run.identifiers.apply(step.response)
        run.served += 1
        # The body's digest names this request in other components' logs (Tidelet's LLM proxy
        # logs the same digest with each delivery), so a step can be joined with them.
        log.info("run %s: %s step %d by %s (%d tool calls) body=%s", name, thread_id, step.index, how, len(step.tool_calls),
                 hashlib.sha256(raw).hexdigest())

        # A retry is recorded like a first serve when the first one never was: its stream broke
        # off before the end (the agent's sandbox was stopped, a rollback, while it arrived).
        record = run.recorder is not None and (how != "retry" or not run.recorder.has_response(thread_id, response))
        if record and thread_id not in {t.id for t in run.recorder.tape.threads}:
            run.recorder.new_thread(thread.parent, thread_id)

        stream = req.stream
        rendered = provider.render_response(response, stream)
        headers = {k: v for k, v in response.headers.items() if k.lower() in ("request-id", "anthropic-organization-id")}
        first_byte_at = last_byte_at = None
        if stream and isinstance(rendered, list):
            out = web.StreamResponse(status=response.status, headers={"Content-Type": provider.content_type(True), "Cache-Control": "no-cache", **headers})
            await out.prepare(request)
            for delay, data in self.pacer.chunk_delays(step, rendered):
                if delay > 0:
                    await asyncio.sleep(delay)
                if first_byte_at is None:
                    first_byte_at = run.now()
                await out.write(data.encode())
            await out.write_eof()
            last_byte_at = run.now()
        else:
            delay = self.pacer.response_delay(step)
            if delay > 0:
                await asyncio.sleep(delay)
            first_byte_at = last_byte_at = run.now()
            out = web.Response(body=rendered if isinstance(rendered, bytes) else "".join(d for _, d in rendered).encode(), status=response.status, content_type=provider.content_type(False), headers=headers)

        if record:
            run.recorder.record(req, response, Timing(request_at, first_byte_at, last_byte_at), thread_id=thread_id)
            if how == "retry":
                log.info("run %s: %s step %d recorded at its retry (its first serve broke off)", name, thread_id, step.index)
        elif run.recorder is not None and run.recorder.retime(thread_id, response, Timing(request_at, first_byte_at, last_byte_at)):
            log.info("run %s: %s step %d retimed to the retry", name, thread_id, step.index)
        return out

    def _canned(self, path: str) -> web.Response:
        if path.endswith("/count_tokens"):
            return web.json_response({"input_tokens": 0})
        if path.endswith("/models"):
            return web.json_response({"data": [{"id": self.tape.meta.model, "type": "model", "object": "model"}], "object": "list"})
        return web.json_response({"error": {"type": "not_found_error", "message": f"tidebox simulator does not serve {path}"}}, status=404)
