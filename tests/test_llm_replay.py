"""Record a scripted run through a minimal recording proxy, then replay the tape with the LLM controller
(`tidebox replay --llm`): a harness pointed at the controller gets the recorded responses and runs its tools again."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import aiohttp
import pytest
from aiohttp import web

from tests.conftest import MiniHarness, anthropic_message, fake_anthropic, text, tool_use
from tidebox.api.io import read_tape, validate
from tidebox.api.providers import detect_provider
from tidebox.api.providers.base import scrub_headers
from tidebox.api.recorder import Recorder
from tidebox.api.tape import StreamChunk, TapeMeta, Timing
from tidebox.controller.compare import compare, summarize
from tidebox.controller.server import ControllerServer


def script(body: dict) -> dict:
    """What the 'model' says, keyed by where the conversation is."""
    messages = body["messages"]
    first = messages[0]["content"]
    last = messages[-1]
    if first == "build it":  # main agent
        if len(messages) == 1:
            return anthropic_message([text("Starting."), tool_use("toolu_01", "Bash", command="echo hello > hello.txt"), tool_use("toolu_02", "TodoWrite", todos=[{"content": "x"}])], "tool_use")
        if len(messages) == 3:
            return anthropic_message([tool_use("toolu_03", "Agent", prompt="count the files", description="count")], "tool_use")
        if len(messages) == 5:
            counted = last["content"][0]["content"]
            return anthropic_message([tool_use("toolu_04", "Bash", command=f"echo '{counted}' > count.txt; cat count.txt; false")], "tool_use")
        return anthropic_message([text("All done.")])
    if first == "count the files":  # subagent
        if len(messages) == 1:
            return anthropic_message([tool_use("toolu_11", "Bash", command="ls | wc -l")], "tool_use")
        n = last["content"][0]["content"].strip()
        return anthropic_message([text(f"there are {n} files")])
    raise AssertionError(f"unexpected conversation: {first!r}")


def recording_proxy(upstream: str, recorder: Recorder) -> web.Application:
    """A pass-through proxy to `upstream` that records every model call into `recorder`, as recordings are made."""
    started = time.monotonic()
    now = lambda: int((time.monotonic() - started) * 1000)  # noqa: E731

    async def handle(request: web.Request) -> web.StreamResponse:
        path = "/" + request.match_info["path"]
        body = await request.read()
        request_at, chunks, first_byte_at = now(), [], None
        forward = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "x-api-key", "anthropic-version")}
        async with aiohttp.ClientSession() as session, session.post(upstream + path, data=body, headers=forward) as up:
            response = web.StreamResponse(status=up.status, headers={"content-type": up.headers.get("content-type", "application/json")})
            await response.prepare(request)
            async for piece in up.content.iter_any():
                first_byte_at = first_byte_at if first_byte_at is not None else now()
                chunks.append(StreamChunk(now(), piece.decode()))
                await response.write(piece)
            await response.write_eof()
            stream = up.headers.get("content-type", "").startswith("text/event-stream")
        provider = detect_provider(path, dict(request.headers))
        if provider is not None and path.endswith("/messages"):
            req = provider.parse_request(json.loads(body), scrub_headers({k.lower(): v for k, v in request.headers.items()}), path)
            resp = provider.parse_response(None if stream else json.loads("".join(c.data for c in chunks)), chunks if stream else None, up.status, {})
            recorder.record(req, resp, Timing(request_at, first_byte_at, now()))
        return response

    app = web.Application()
    app.router.add_post("/{path:.*}", handle)
    return app


@pytest.fixture
def recording(serve, tmp_path):
    provider = serve(fake_anthropic(script))
    tape_path = tmp_path / "run.json"
    recorder = Recorder(tape_path, TapeMeta("rec1", "claude-code", "anthropic", "fake", "2026-01-01T00:00:00Z"))
    proxy = serve(recording_proxy(provider.url, recorder))
    workdir = tmp_path / "record"
    workdir.mkdir()
    harness = MiniHarness(proxy.url, str(workdir))
    assert harness.run_conversation("build it") == "All done."
    proxy.stop()
    recorder.finalize()
    return tape_path, harness


def test_recording_is_a_valid_tape(recording):
    tape = read_tape(recording[0])
    assert validate(tape) == []
    assert [t.id for t in tape.threads] == ["t0", "t1"] and tape.threads[1].parent == ("t0", "toolu_03")
    assert [len(t.steps) for t in tape.threads] == [4, 2]


def test_llm_replay(recording, serve, tmp_path):
    tape_path, original = recording
    tape = read_tape(tape_path)
    replay_path = tmp_path / "replay.json"
    controller = serve(ControllerServer(tape, speed=0, output=replay_path).app)
    workdir = tmp_path / "replay"
    workdir.mkdir()
    harness = MiniHarness(controller.url, str(workdir))
    assert harness.run_conversation("build it") == "All done."
    assert (workdir / "hello.txt").read_text() == "hello\n"
    assert (workdir / "count.txt").read_text() == "there are 1 files\n"
    # the recorded responses were served byte for byte: same tool-use ids in the same order
    assert [r["tool_use_id"] for r in harness.results] == [r["tool_use_id"] for r in original.results]
    # non-streaming requests and a second concurrent run work too
    plain = MiniHarness(controller.url + "/r/second", str(workdir), stream=False)
    assert plain.run_conversation("build it") == "All done."
    state = json.loads(urllib.request.urlopen(controller.url + "/tidebox/state").read())
    assert state["simulator"]["default"]["served"] == 6 and state["simulator"]["second"]["unmatched"] == 0
    controller.stop()
    replay = read_tape(replay_path)
    assert replay.meta.replay_of == "rec1" and validate(replay) == []
    report = compare(tape, replay)
    assert report["complete"] and report["identical"], summarize(report)
    second = read_tape(tmp_path / "replay.second.json")
    assert second.meta.labels["mode"] == "llm" and compare(tape, second)["complete"]


def test_unknown_conversation_is_refused(recording, serve):
    controller = serve(ControllerServer(read_tape(recording[0]), speed=0).app)
    body = json.dumps({"model": "fake", "messages": [{"role": "user", "content": "something else"}], "tools": []}).encode()
    req = urllib.request.Request(controller.url + "/v1/messages", data=body, headers={"content-type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 404 and "no recorded response" in e.value.read().decode()
