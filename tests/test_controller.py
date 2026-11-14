import json

from tidebox.api.tape import ContentBlock, ModelResponse, StreamChunk, ToolCall, ToolResult
from tidebox.controller.identifiers import IdentifierMap


def test_identifier_map_learns_and_applies():
    m = IdentifierMap()
    m.learn(ToolResult.of_text("c", "Command running in background with ID: task_ab12cd. Output: /tmp/x/4f3a9c1b.output"),
            ToolResult.of_text("c", "Command running in background with ID: task_zz99yy. Output: /tmp/x/deadbeef.output"))
    assert m.mapping == {"task_ab12cd": "task_zz99yy", "4f3a9c1b": "deadbeef"}
    r = ModelResponse(content=[ContentBlock("tool_use", tool_call_id="x", name="BashOutput", input={"bash_id": "task_ab12cd"}, raw={"type": "tool_use", "id": "x", "name": "BashOutput", "input": {"bash_id": "task_ab12cd"}})],
                      chunks=[StreamChunk(0, 'data: {"bash_id": "task_ab12cd"}\n\n')])
    out = m.apply(r)
    assert out.content[0].input == {"bash_id": "task_zz99yy"} and "task_zz99yy" in out.chunks[0].data
    assert r.content[0].input == {"bash_id": "task_ab12cd"}  # the tape is untouched
    assert m.apply_call(ToolCall("x", "BashOutput", {"bash_id": "task_ab12cd"})).input == {"bash_id": "task_zz99yy"}
    untouched = ModelResponse(content=[ContentBlock("text", text="hello")])
    assert m.apply(untouched) is untouched


def test_identifier_map_ignores_mismatched_shapes():
    m = IdentifierMap()
    m.learn_text("ids 123456 and 789012", "only 111111")
    assert m.mapping == {}


def test_identifier_map_substitution_is_safe():
    """A learned token never corrupts JSON structure, signatures or longer tokens."""
    m = IdentifierMap()
    m.learn_text("saved to /tmp/out/a1b2c3d4.txt pid 12345", "saved to /tmp/out/ffffffff.txt pid 67890")
    assert m.mapping == {"a1b2c3d4": "ffffffff", "12345": "67890"}
    # token adjacent to JSON syntax, inside a key, inside a longer token, inside a signature
    chunk = ('event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta",'
             '"partial_json":"{\\"path\\": \\"/tmp/out/a1b2c3d4.txt\\", \\"a1b2c3d4\\": 12345}"}}\n\n'
             'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"xx12345yya1b2c3d4=="}}\n\n')
    r = ModelResponse(content=[ContentBlock("text", text="see a1b2c3d4 and 12345 and xa1b2c3d4y")], chunks=[StreamChunk(0, chunk)])
    out = m.apply(r)
    assert out.content[0].text == "see ffffffff and 67890 and xa1b2c3d4y"
    for block in out.chunks[0].data.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                json.loads(line[6:])  # every data line is still valid JSON
    assert '"partial_json":"{\\"path\\": \\"/tmp/out/ffffffff.txt\\", \\"ffffffff\\": 67890}"' in out.chunks[0].data
    assert '"signature":"xx12345yya1b2c3d4=="' in out.chunks[0].data, "signatures are left alone"
    # a data line cut by a chunk boundary is left untouched rather than broken
    half = StreamChunk(0, 'data: {"type":"text_delta","text":"a1b2c3d4 is')
    assert m.apply(ModelResponse(content=[], chunks=[half])).chunks[0].data == half.data


def test_identifier_map_ignores_plain_identifiers():
    m = IdentifierMap()
    m.learn_text("def tool_call(): pass\nclass chat_completion:", "def streaming_events(): pass\nclass maybe_filter:")
    assert m.mapping == {}, "ordinary snake_case names are not run-specific identifiers"
    m.learn_text("task id task_01ab started", "task id task_99zz started")
    assert m.mapping == {"task_01ab": "task_99zz"}
