"""The delta tape format: requests stored as changes against the previous step's, read back
exactly, indexed and served without reconstruction, and converted from the legacy format."""

import json

import pytest

from tidebox.api.delta import TapeWriter, decode_request, decode_tape, encode_request, encode_tape, is_delta
from tidebox.api.io import read_tape, to_dict, validate, write_tape
from tidebox.api.lazy import DeferredRequest, LazyTape
from tidebox.api.providers import get_provider
from tidebox.api.recorder import Recorder
from tidebox.api.tape import (
    TAPE_VERSION,
    ContentBlock,
    ModelRequest,
    ModelResponse,
    Tape,
    TapeMeta,
    Timing,
)
from tidebox.controller.matcher import Matcher


def _meta(tape_id="delta"):
    return TapeMeta(tape_id=tape_id, harness="claude-code", provider="anthropic", model="m", started_at="2026-01-01T00:00:00Z")


def _body(i, prev_ids):
    """What Claude Code sends at step i: the conversation so far, the newest user message as a
    plain string the first time and as blocks once it is history (so the previous request's
    last message is never byte-equal), a tool result for every previous call."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "do the task"}]}]
    for j, cid in enumerate(prev_ids):
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": cid, "name": "Bash", "input": {"command": f"step {j}"}}]})
        result = {"type": "tool_result", "tool_use_id": cid, "content": [{"type": "text", "text": f"out{j}\n" * 40}]}
        messages.append({"role": "user", "content": [result] if j < len(prev_ids) - 1 else [result, {"type": "text", "text": "and continue"}]})
    if not prev_ids:
        messages[0]["content"] = "do the task"
    return {"model": "m", "max_tokens": 100, "stream": True, "system": [{"type": "text", "text": "sys"}], "tools": [{"name": "Bash"}, {"name": "Read"}],
            "metadata": {"user_id": "u"}, "messages": messages}


def _record(path, steps=5, spill=False, subagent=True):
    p = get_provider("anthropic")
    rec = Recorder(path, _meta(), p, spill=spill)
    ids = []
    for i in range(steps):
        req = p.parse_request(_body(i, ids), {"anthropic-beta": "x"}, "/v1/messages")
        resp = ModelResponse(content=[ContentBlock("text", text=f"thinking {i}"), ContentBlock("tool_use", tool_call_id=f"toolu_{i}", name="Bash", input={"command": f"step {i}"})],
                             stop_reason="tool_use", raw={"id": f"msg_{i}"})
        rec.record(req, resp, Timing(request_at=i * 1000, first_byte_at=i * 1000 + 5, last_byte_at=i * 1000 + 50))
        ids.append(f"toolu_{i}")
    if subagent:
        req = p.parse_request({"model": "m", "messages": [{"role": "user", "content": "sub task"}], "tools": [{"name": "Bash"}]}, {}, "/v1/messages")
        rec.record(req, ModelResponse(content=[ContentBlock("text", text="done")], stop_reason="end_turn"), Timing(request_at=7000, last_byte_at=7100), thread_id="sub")
    return rec.finalize()


# -- encoding ----------------------------------------------------------------------------------


def test_request_delta_roundtrip_and_shape():
    prev = {"system": "s", "messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "x"}], "tools": [{"name": "a"}], "params": {"model": "m", "max_tokens": 5},
            "stream": True, "raw": {"model": "m", "messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "x"}], "gone": 1}, "headers": {}, "path": "/v1/messages"}
    cur = {"system": "s", "messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "x"}]}, {"role": "assistant", "content": "ok"}],
           "tools": [{"name": "a"}], "params": {"model": "m", "max_tokens": 6}, "stream": False,
           "raw": {"model": "m", "messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "x"}]}, {"role": "assistant", "content": "ok"}], "extra": [1]},
           "headers": {}, "path": "/v1/messages"}
    stored = encode_request(prev, cur)
    assert is_delta(stored)
    d = stored["$delta"]
    assert d["system"] == {"$same": 1} and d["tools"] == {"$same": 1} and d["headers"] == {"$same": 1}
    assert d["messages"] == {"$shared": 1, "$new": cur["messages"][1:]}, "the modified last message and the new ones are the tail"
    assert d["params"] == {"$delta": {"model": {"$same": 1}, "max_tokens": 6}}
    assert d["stream"] is False
    assert d["raw"]["$delta"]["messages"]["$shared"] == 1 and "gone" not in d["raw"]["$delta"] and d["raw"]["$delta"]["extra"] == [1]
    back = decode_request(prev, stored)
    assert back == cur and list(back) == list(cur) and list(back["raw"]) == list(cur["raw"]), "same content, same key order"
    assert encode_request(None, cur) is cur and decode_request(None, cur) is cur
    with pytest.raises(ValueError):
        decode_request(None, stored)


def test_marker_lookalikes_are_refused():
    with pytest.raises(ValueError):
        encode_request(None, {"raw": {"$same": 1}})
    with pytest.raises(ValueError):
        encode_request({"a": 1}, {"a": 2, "b": {"$shared": 1, "$new": []}})
    assert encode_request({"a": 1}, {"a": {"$other": 1}}) == {"$delta": {"a": {"$other": 1}}}, "other dollar keys are plain values"


# -- tapes ----------------------------------------------------------------------------------------


def test_recorded_tape_is_version_2_and_reads_back(tmp_path):
    tape = _record(tmp_path / "t.json")
    assert tape.version == TAPE_VERSION == 2
    raw = json.loads((tmp_path / "t.json").read_text())
    assert raw["version"] == 2
    main = raw["threads"][0]["steps"]
    assert not is_delta(main[0]["request"]) and all(is_delta(s["request"]) for s in main[1:])
    assert all(not is_delta(t["steps"][0]["request"]) for t in raw["threads"]), "every thread's first step is full"
    assert main[3]["request"]["$delta"]["messages"]["$shared"] == len(main[2]["request"]["$delta"]["messages"]["$new"]) + main[2]["request"]["$delta"]["messages"]["$shared"] - 1
    again = read_tape(tmp_path / "t.json")
    assert to_dict(again) == to_dict(tape) and validate(again) == []
    assert to_dict(read_tape(tmp_path / "legacy.json")) == {**to_dict(tape), "version": 1} if write_tape(Tape(1, tape.meta, tape.threads), tmp_path / "legacy.json") is None else False
    assert decode_tape(encode_tape(tape)) == to_dict(tape)
    # A longer run shows the point: the legacy encoding grows with the square of the length.
    long = _record(tmp_path / "long.json", steps=30, subagent=False)
    write_tape(Tape(1, long.meta, long.threads), tmp_path / "long-legacy.json")
    assert (tmp_path / "long-legacy.json").stat().st_size > 4 * (tmp_path / "long.json").stat().st_size


def test_spill_writes_the_same_bytes(tmp_path):
    _record(tmp_path / "plain.json", spill=False)
    _record(tmp_path / "spilled.json", spill=True)
    a, b = json.loads((tmp_path / "plain.json").read_text()), json.loads((tmp_path / "spilled.json").read_text())
    a["meta"]["started_at"] = b["meta"]["started_at"] = ""
    assert a == b


def test_tape_writer_matches_write_tape(tmp_path):
    tape = _record(tmp_path / "t.json")
    with TapeWriter(tmp_path / "w.json", 2, to_dict(tape.meta)) as w:
        for thread in tape.threads:
            w.begin_thread(thread.id, thread.parent)
            for step in thread.steps:
                w.write_step(to_dict(step))
    assert (tmp_path / "w.json").read_text() == (tmp_path / "t.json").read_text()


def test_compressed_tapes_are_refused(tmp_path):
    tape = _record(tmp_path / "t.json")
    import lzma

    with lzma.open(tmp_path / "t.json.xz", "wb") as f:
        f.write((tmp_path / "t.json").read_bytes())
    with pytest.raises(ValueError):
        LazyTape(tmp_path / "t.json.xz")
    with pytest.raises(ValueError):
        read_tape(tmp_path / "t.json.xz")
    with pytest.raises(ValueError):
        write_tape(tape, tmp_path / "w.json.xz")


# -- on-demand reading -----------------------------------------------------------------------------


def test_lazy_indexes_version_2_without_reconstructing(tmp_path):
    tape = _record(tmp_path / "t.json")
    full = read_tape(tmp_path / "t.json")
    p = get_provider("anthropic")
    with LazyTape(tmp_path / "t.json") as lazy:
        assert lazy.version == 2
        # Keys, openings and tool counts come out as from the full requests.
        for thread in full.threads:
            for step in thread.steps:
                ref = lazy.thread(thread.id).steps.refs[step.index]
                assert ref.key == p.conversation_key(step.request) and ref.opening == (step.request.last_assistant() is None) and ref.tools == len(step.request.tools)
        assert lazy.reconstructions == 0, "indexing reads no request"
        # Serving touches responses and tool results only.
        step = lazy.thread("t0").steps[3]
        assert isinstance(step.request, DeferredRequest) and not step.request.loaded
        assert step.response.tool_calls()[0].id == "toolu_3" and step.tool_results[0].tool_call_id == "toolu_3"
        assert lazy.reconstructions == 0
        # Reading the request reconstructs it, equal to the full tape's.
        assert step.request == full.threads[0].steps[3].request and step.request.loaded
        assert isinstance(step.request, ModelRequest) and lazy.reconstructions == 4
        assert to_dict(step) == to_dict(full.threads[0].steps[3])
        # The next step in order costs one more delta; an earlier one restarts from the thread's first.
        lazy.thread("t0").steps[4].request.messages
        assert lazy.reconstructions == 5
        lazy.thread("t0").steps[1].request.messages
        assert lazy.reconstructions == 7
        # The whole tape, step by step, equals read_tape; in order, that is one delta per
        # step from the cached step 1 on (steps 2 to 4), the first steps being full.
        assert [to_dict(s) for t in lazy.threads for s in t.steps] == [to_dict(s) for t in full.threads for s in t.steps]
        assert lazy.reconstructions == 10
        # The matcher works from the index alone.
        m = Matcher(lazy, p)
        req = p.parse_request(_body(2, ["toolu_0", "toolu_1"]), {}, "/v1/messages")
        tid, matched, how = m.match(req)
        assert (tid, matched.index, how) == ("t0", 2, "key")
        assert lazy.reconstructions == 10


def test_lazy_reads_legacy_tapes_too(tmp_path):
    tape = _record(tmp_path / "t.json")
    write_tape(Tape(1, tape.meta, tape.threads), tmp_path / "legacy.json")
    with LazyTape(tmp_path / "legacy.json") as lazy:
        assert lazy.version == 1
        assert [r.key for r in lazy.refs] == [get_provider("anthropic").conversation_key(s.request) for t in tape.threads for s in t.steps]
        step = lazy.thread("t0").steps[2]
        assert not isinstance(step.request, DeferredRequest) and to_dict(step) == to_dict(tape.threads[0].steps[2])


# -- converting ------------------------------------------------------------------------------------


