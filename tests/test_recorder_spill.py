"""The spilling recorder writes the same tape as the in-memory one, holding one step per thread."""

import json

from tidebox.api.io import read_tape, to_dict
from tidebox.api.lazy import spilled_refs
from tidebox.api.recorder import Recorder
from tidebox.api.tape import ContentBlock, Message, ModelRequest, ModelResponse, TapeMeta, Timing


def _meta():
    return TapeMeta(tape_id="spill", harness="claude-code", provider="anthropic", model="m", started_at="2026-01-01T00:00:00Z")


def _call(i, with_prev_result=None):
    """A request that continues the conversation (carrying the previous tool result) and a
    response that makes one tool call."""
    messages = [Message("user", [ContentBlock("text", text="task")])]
    if with_prev_result is not None:
        messages.append(Message("assistant", [ContentBlock("tool_use", tool_call_id=with_prev_result, name="Bash", input={"command": "ls"})]))
        messages.append(Message("user", [ContentBlock("tool_result", tool_call_id=with_prev_result, text="file" * 1000)]))
    req = ModelRequest(messages=messages, tools=[{"name": "Bash"}], raw={"messages": "x" * 5000}, path="/v1/messages")
    resp = ModelResponse(content=[ContentBlock("tool_use", tool_call_id=f"call{i}", name="Bash", input={"command": "ls"}), ContentBlock("text", text="ok")],
                         stop_reason="tool_use", usage={"input_tokens": 1}, raw={"id": f"msg{i}"})
    return req, resp


def _record(rec):
    prev = None
    for i in range(4):
        req, resp = _call(i, prev)
        rec.record(req, resp, Timing(request_at=i * 1000, first_byte_at=i * 1000 + 10, last_byte_at=i * 1000 + 20), thread_id="t0")
        prev = f"call{i}"
    return rec.finalize()


def test_spill_matches_in_memory(tmp_path):
    plain = _record(Recorder(tmp_path / "plain.json", _meta()))
    spilled_rec = Recorder(tmp_path / "spilled.json", _meta(), spill=True)
    # While recording, completed steps are emptied in memory and only the last one is whole.
    for i in range(3):
        req, resp = _call(i, None if i == 0 else f"call{i-1}")
        spilled_rec.record(req, resp, Timing(request_at=i * 1000, first_byte_at=i * 1000 + 10, last_byte_at=i * 1000 + 20), thread_id="t0")
    steps = spilled_rec.tape.threads[0].steps
    assert steps[0].request.messages == [] and steps[1].response.content == [] and steps[2].response.content, "only the last step stays in memory"
    assert spilled_rec.spill_path.exists() and spilled_rec.spill_path.read_text().count("\n") == 2
    req, resp = _call(3, "call2")
    spilled_rec.record(req, resp, Timing(request_at=3000, first_byte_at=3010, last_byte_at=3020), thread_id="t0")
    spilled_rec.finalize()
    assert not spilled_rec.spill_path.exists(), "spill file is removed after assembling"

    a = json.loads((tmp_path / "plain.json").read_text())
    b = json.loads((tmp_path / "spilled.json").read_text())
    for d in (a, b):
        d["meta"]["started_at"] = ""  # the only field that legitimately differs
    assert a == b
    # And it reads back as a tape with results attached to the right steps.
    t = read_tape(tmp_path / "spilled.json")
    assert [len(s.tool_results) for s in t.threads[0].steps] == [1, 1, 1, 0]
    assert to_dict(t.threads[0].steps[1].tool_results[0])["tool_call_id"] == "call1"
    assert plain.threads[0].steps[2].timing.next_request_at == 3000 == t.threads[0].steps[2].timing.next_request_at


def test_spill_without_output_is_a_noop(tmp_path):
    rec = Recorder(None, _meta(), spill=True)
    assert not rec.spill and rec.write_every_step
    _record(rec)


def test_spilled_steps_are_readable_while_recording(tmp_path):
    rec = Recorder(tmp_path / "live.json", _meta(), spill=True)
    for i in range(3):
        req, resp = _call(i, None if i == 0 else f"call{i-1}")
        rec.record(req, resp, Timing(request_at=i * 1000, first_byte_at=i * 1000 + 10, last_byte_at=i * 1000 + 20), thread_id="t0")
    # The two completed steps are on disk with their timing; the last waits for its next request.
    refs = spilled_refs(rec.spill_path)
    assert [(r.thread, r.index, r.tool_calls) for r in refs] == [("t0", 0, 1), ("t0", 1, 1)]
    assert [(r.request_at, r.last_byte_at, r.next_request_at) for r in refs] == [(0, 20, 1000), (1000, 1020, 2000)]
    # A line still being written is skipped, not an error.
    with open(rec.spill_path, "a", encoding="utf-8") as f:
        f.write('t0\t{"index": 2, "timing": {"request_')
    assert len(spilled_refs(rec.spill_path)) == 2
