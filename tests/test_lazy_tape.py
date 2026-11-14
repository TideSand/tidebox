"""LazyTape reads a tape step by step from its byte ranges and matches read_tape exactly."""

import json

from tidebox.api.io import read_tape, to_dict, validate
from tidebox.api.lazy import LazyTape
from tidebox.api.providers import get_provider
from tidebox.api.recorder import Recorder
from tidebox.api.tape import ContentBlock, Message, ModelRequest, ModelResponse, TapeMeta, Timing
from tidebox.controller.compare import compare
from tidebox.controller.matcher import Matcher


def _tape(path):
    rec = Recorder(path, TapeMeta(tape_id="lazy", harness="claude-code", provider="anthropic", model="m", started_at="2026-01-01T00:00:00Z"))
    prev = None
    for i in range(5):
        messages = [Message("user", [ContentBlock("text", text="task with ünïcödé and \"quotes\"")])]
        if prev:
            messages += [Message("assistant", [ContentBlock("tool_use", tool_call_id=prev, name="Bash", input={"command": "ls"})]),
                         Message("user", [ContentBlock("tool_result", tool_call_id=prev, text="out\n" * 50)])]
        req = ModelRequest(messages=messages, tools=[{"name": "Bash"}, {"name": "Read"}], raw={"m": "x" * 300}, path="/v1/messages", stream=True)
        resp = ModelResponse(content=[ContentBlock("tool_use", tool_call_id=f"c{i}", name="Bash", input={"command": f"echo {i}"})], stop_reason="tool_use",
                             raw={"id": f"msg{i}", "text": "é" * 20})
        rec.record(req, resp, Timing(request_at=i * 1000, first_byte_at=i * 1000 + 5, last_byte_at=i * 1000 + 50), thread_id="t0")
        prev = f"c{i}"
    # a second (sub-agent) thread
    rec.record(ModelRequest(messages=[Message("user", [ContentBlock("text", text="sub task")])], tools=[{"name": "Bash"}], path="/v1/messages"),
               ModelResponse(content=[ContentBlock("text", text="done")], stop_reason="end_turn"), Timing(request_at=7000, last_byte_at=7100), thread_id="t1")
    return rec.finalize()


def test_lazy_matches_read_tape(tmp_path):
    path = tmp_path / "run.json"
    _tape(path)
    full = read_tape(path)
    with LazyTape(path) as lazy:
        assert lazy.version == full.version and to_dict(lazy.meta) == to_dict(full.meta)
        assert [t.id for t in lazy.threads] == [t.id for t in full.threads]
        assert [len(t.steps) for t in lazy.threads] == [5, 1]
        for lt, ft in zip(lazy.threads, full.threads):
            assert lt.parent == ft.parent
            assert [to_dict(s) for s in lt.steps] == [to_dict(s) for s in ft.steps]
            assert to_dict(lt.steps[-1]) == to_dict(ft.steps[-1]) and to_dict(lt.steps[2 % len(lt.steps)]) == to_dict(ft.steps[2 % len(ft.steps)])
        assert lazy.main_thread().id == "t0" and lazy.thread("t1").steps[0].response.stop_reason == "end_turn"
        assert validate(lazy) == validate(full)
        # The matcher uses the prebuilt index and finds the same steps.
        provider = get_provider("anthropic")
        req = read_tape(path).threads[0].steps[3].request  # taken before any matcher empties requests
        m_lazy, m_full = Matcher(lazy, provider), Matcher(full, provider)
        a, b = m_lazy.match(req), m_full.match(req)
        assert a[0] == b[0] and a[1].index == b[1].index == 3 and a[2] == b[2]
        # and comparing a tape with itself through lazy readers is identical
        with LazyTape(path) as again:
            report = compare(lazy, again)
            # complete, and every call that has a recorded result is "same"; the last call of a
            # thread has no result in either tape, which compare reports as "no result"
            assert report["complete"] and report["recorded_steps"] == report["replayed_steps"] == 6
            outcomes = [c["outcome"] for t in report["threads"].values() for st in t["steps"] for c in st["calls"].values()]
            assert outcomes.count("same") == 4 and set(outcomes) <= {"same", "no result"}


def test_index_is_small(tmp_path):
    path = tmp_path / "run.json"
    _tape(path)
    with LazyTape(path) as lazy:
        assert len(lazy.refs) == 6
        assert len(json.dumps([r.__dict__ for r in lazy.refs])) < 6 * 400


def test_index_carries_step_timing(tmp_path):
    """Timelines are drawn from the index alone, so each ref carries its step's timing as the
    full tape records it, including the recorder's next_request_at."""
    path = tmp_path / "timing.json"
    _tape(path)
    full = read_tape(path)
    with LazyTape(path) as tape:
        for ref in tape.refs:
            step = full.thread(ref.thread).steps[ref.index]
            assert (ref.request_at, ref.last_byte_at, ref.next_request_at) == (step.timing.request_at, step.timing.last_byte_at, step.timing.next_request_at)
            assert ref.tool_calls == len(step.tool_calls)
        t0 = [r for r in tape.refs if r.thread == "t0"]
        assert [r.request_at for r in t0] == [0, 1000, 2000, 3000, 4000]
        assert all(r.next_request_at == r.request_at + 1000 for r in t0[:-1]) and t0[-1].next_request_at is None
