import json

from tidebox.api.io import from_dict, read_tape, to_dict, validate, write_tape
from tidebox.api.providers import detect_provider, get_provider
from tidebox.api.recorder import Recorder
from tidebox.api.tape import TAPE_VERSION, ContentBlock, ModelResponse, TapeMeta, Timing, ToolResult


def test_anthropic_request_roundtrip():
    p = get_provider("anthropic")
    body = {"model": "m", "max_tokens": 5, "system": [{"type": "text", "text": "sys"}], "stream": True, "tools": [{"name": "Bash"}],
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a\nb"}], "is_error": False}]}]}
    req = p.parse_request(body, {"x-api-key": "secret", "anthropic-beta": "x"}, "/v1/messages")
    assert req.system == "sys" and req.stream and "x-api-key" not in req.headers and req.params["model"] == "m"
    assert p.conversation_key(req) == "ids:toolu_1"
    assert p.tool_results(req)[0].text() == "a\nb"
    assert req.first_user_text() == "hi"


def test_anthropic_stream_roundtrip():
    p = get_provider("anthropic")
    r = ModelResponse(content=[ContentBlock("text", text="Hello "), ContentBlock("tool_use", tool_call_id="toolu_2", name="Bash", input={"command": "echo hi"})], stop_reason="tool_use", usage={"output_tokens": 3})
    chunks = p.render_response(r, stream=True)
    r2 = p.parse_response(None, chunks)
    assert r2.text() == "Hello " and r2.tool_calls()[0].input == {"command": "echo hi"} and r2.stop_reason == "tool_use"
    assert p.response_key(r2) == "ids:toolu_2"
    body = json.loads(p.render_response(r2, stream=False))
    assert body["content"][1]["input"] == {"command": "echo hi"}


def test_openai_chat_and_responses():
    o = get_provider("openai")
    req = o.parse_request({"model": "g", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
                                                      {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "{\"cmd\":[\"ls\"]}"}}]},
                                                      {"role": "tool", "tool_call_id": "call_1", "content": "out"}], "stream": True}, {}, "/v1/chat/completions")
    assert req.system == "s" and o.conversation_key(req) == "ids:call_1" and o.tool_results(req)[0].text() == "out"
    r = ModelResponse(content=[ContentBlock("tool_use", tool_call_id="call_2", name="shell", input={"cmd": ["pwd"]})])
    r2 = o.parse_response(None, o.render_response(r, True))
    assert r2.tool_calls()[0].input == {"cmd": ["pwd"]} and r2.stop_reason == "tool_calls"
    resp_body = {"id": "resp_1", "object": "response", "status": "completed", "output": [{"type": "function_call", "call_id": "call_9", "name": "shell", "arguments": "{\"command\":[\"ls\"]}"}], "usage": {"input_tokens": 1, "output_tokens": 2}}
    r = o.parse_response(resp_body, None)
    assert r.tool_calls()[0].id == "call_9"
    assert o.parse_response(None, o.render_response(r, True)).tool_calls()[0].name == "shell"
    req = o.parse_request({"model": "m", "instructions": "inst", "input": [{"role": "user", "content": [{"type": "input_text", "text": "task"}]}, {"type": "function_call", "call_id": "call_9", "name": "shell", "arguments": "{}"}, {"type": "function_call_output", "call_id": "call_9", "output": "done"}]}, {}, "/v1/responses")
    assert req.system == "inst" and o.conversation_key(req) == "ids:call_9" and o.tool_results(req)[0].text() == "done"


def test_detect_provider():
    assert detect_provider("/v1/messages").name == "anthropic"
    assert detect_provider("/v1/messages/count_tokens") is None
    assert detect_provider("/v1/responses").name == "openai"
    assert detect_provider("/openai/v1/chat/completions").name == "openai"


def _req(p, msgs):
    return p.parse_request({"messages": msgs}, {}, "/v1/messages")


def _resp(p, blocks):
    return p.parse_response({"content": blocks, "stop_reason": "tool_use", "usage": {"output_tokens": 1}}, None)


def test_recorder_threads_and_results(tmp_path):
    p = get_provider("anthropic")
    rec = Recorder(tmp_path / "t.json", TapeMeta("t", "claude-code", "anthropic", "m", "now"))
    rec.record(_req(p, [{"role": "user", "content": "do task"}]), _resp(p, [{"type": "tool_use", "id": "c1", "name": "Agent", "input": {"prompt": "explore repo"}}]), Timing(0, 1, 2))
    rec.record(_req(p, [{"role": "user", "content": "explore repo"}]), _resp(p, [{"type": "tool_use", "id": "c2", "name": "Bash", "input": {"command": "ls"}}]), Timing(3, 4, 5))
    rec.record(_req(p, [{"role": "user", "content": "explore repo"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "c2", "name": "Bash", "input": {"command": "ls"}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c2", "content": "files"}]}]), _resp(p, [{"type": "text", "text": "done exploring"}]), Timing(6, 7, 8))
    rec.record(_req(p, [{"role": "user", "content": "do task"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "Agent", "input": {"prompt": "explore repo"}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "done exploring"}]}]), _resp(p, [{"type": "text", "text": "finished"}]), Timing(9, 10, 11))
    t = rec.finalize()
    assert [th.id for th in t.threads] == ["t0", "t1"] and t.threads[1].parent == ("t0", "c1")
    assert t.threads[0].steps[0].tool_results[0].text() == "done exploring"
    assert t.threads[1].steps[0].timing.next_request_at == 6 and t.threads[0].steps[0].timing.next_request_at == 9
    assert validate(t) == []
    again = read_tape(tmp_path / "t.json")
    assert to_dict(again) == to_dict(t)
    assert json.loads((tmp_path / "t.json").read_text())["version"] == TAPE_VERSION


def test_tape_json_roundtrip_and_validation(tmp_path):
    p = get_provider("anthropic")
    rec = Recorder(None, TapeMeta("t", "claude-code", "anthropic", "m", "now"))
    rec.record(_req(p, [{"role": "user", "content": "x"}]), _resp(p, [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {}}]), Timing(0, 1, 2))
    t = rec.finalize()
    t.threads[0].steps[0].tool_results.append(ToolResult.of_text("nope", "x"))
    assert any("unknown call" in p for p in validate(t))
    write_tape(t, tmp_path / "a" / "b.json")
    assert from_dict(to_dict(read_tape(tmp_path / "a" / "b.json"))).threads[0].steps[0].tool_calls[0].name == "Bash"


def test_anthropic_tool_results_after_trailing_system_message():
    """Claude Code 2.1.299 appends a `system` message after the user turn that carries the
    tool results; the results must still be found."""
    p = get_provider("anthropic")
    req = _req(p, [
        {"role": "user", "content": "do task"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "files"}]},
        {"role": "system", "content": [{"type": "text", "text": "<total_tokens>1 tokens left</total_tokens>"}]},
    ])
    results = p.tool_results(req)
    assert [r.tool_call_id for r in results] == ["c1"] and results[0].text() == "files"


