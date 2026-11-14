"""OpenAI Chat Completions (`/v1/chat/completions`) and Responses (`/v1/responses`) APIs.

Chat Completions: `tool_calls` on assistant messages, `tool` role messages as results,
streamed deltas (`data: ...` chunks ending with `[DONE]`).
Responses: `input` items, `output` items (`message`, `function_call`, `reasoning`),
`function_call_output` items as results; the stream's `response.completed` event carries the
whole response object, so assembly is a lookup.
"""

from __future__ import annotations

import json
from typing import Any

from tidebox.api.providers.base import Provider, scrub_headers, spread, sse_chunk, sse_events
from tidebox.api.tape import ContentBlock, Message, ModelRequest, ModelResponse, StreamChunk, ToolResult

KEPT_HEADERS = ("openai-beta", "openai-organization", "user-agent", "content-type", "originator", "session_id")
NOT_PARAMS = ("messages", "input", "tools", "stream", "instructions")


def _parse_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args or "{}")
    except json.JSONDecodeError:
        return {"_raw_arguments": args}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"))
    return ""


# -- Chat Completions ---------------------------------------------------------------------------


def _chat_message(m: dict[str, Any]) -> Message:
    role = m.get("role", "user")
    blocks: list[ContentBlock] = []
    if role == "tool":
        return Message("user", [ContentBlock("tool_result", text=_text_of(m.get("content")), tool_call_id=m.get("tool_call_id"), is_error=False, raw=m)])
    text = _text_of(m.get("content"))
    if text:
        blocks.append(ContentBlock("text", text=text))
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        blocks.append(ContentBlock("tool_use", tool_call_id=call.get("id"), name=fn.get("name"), input=_parse_args(fn.get("arguments")), raw=call))
    return Message(role, blocks)


def assemble_chat_stream(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    message: dict[str, Any] | None = None
    text = ""
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    for ev in events:
        if "_raw" in ev or "choices" not in ev:
            if ev.get("object") == "chat.completion":
                return ev
            continue
        if message is None:
            message = {k: v for k, v in ev.items() if k != "choices"}
            message["object"] = "chat.completion"
        if ev.get("usage"):
            message["usage"] = ev["usage"]
        for choice in ev.get("choices") or []:
            delta = choice.get("delta") or {}
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if message is None:
        return None
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    message["choices"] = [{"index": 0, "message": msg, "finish_reason": finish}]
    return message


# -- Responses ----------------------------------------------------------------------------------


def _response_item(item: dict[str, Any]) -> ContentBlock | None:
    t = item.get("type")
    if t == "message":
        return ContentBlock("text", text=_text_of(item.get("content")), raw=item)
    if t == "function_call":
        return ContentBlock("tool_use", tool_call_id=item.get("call_id"), name=item.get("name"), input=_parse_args(item.get("arguments")), raw=item)
    if t == "function_call_output":
        return ContentBlock("tool_result", text=_text_of(item.get("output")), tool_call_id=item.get("call_id"), is_error=False, raw=item)
    if t == "reasoning":
        return ContentBlock("thinking", text="".join(s.get("text", "") for s in item.get("summary") or []), raw=item)
    return ContentBlock("other", raw=item)


def _responses_messages(inp: Any) -> list[Message]:
    if isinstance(inp, str):
        return [Message("user", [ContentBlock("text", text=inp)])]
    out: list[Message] = []
    for item in inp or []:
        if not isinstance(item, dict):
            continue
        if "role" in item and item.get("type") in (None, "message"):
            out.append(Message(item["role"], [ContentBlock("text", text=_text_of(item.get("content")))]))
            continue
        block = _response_item(item)
        if block is None:
            continue
        role = "assistant" if block.type in ("tool_use", "thinking") else "user"
        if block.type == "text" and item.get("role") == "assistant":
            role = "assistant"
        if out and out[-1].role == role:
            out[-1].content.append(block)
        else:
            out.append(Message(role, [block]))
    return out


def assemble_responses_stream(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    final = None
    for ev in events:
        if ev.get("type") in ("response.completed", "response.incomplete", "response.failed") and ev.get("response"):
            final = ev["response"]
        elif ev.get("object") == "response":
            final = ev
    return final


class OpenAIProvider(Provider):
    name = "openai"
    paths = ("/v1/chat/completions", "/chat/completions", "/v1/responses", "/responses")

    @staticmethod
    def _is_responses(path: str, body: dict | None = None) -> bool:
        if path:
            return path.rstrip("/").endswith("/responses")
        return bool(body and "input" in body and "messages" not in body)

    def parse_request(self, body: dict, headers: dict[str, str], path: str = "") -> ModelRequest:
        kept = {k: v for k, v in scrub_headers(headers).items() if k.lower() in KEPT_HEADERS}
        if self._is_responses(path, body):
            messages = _responses_messages(body.get("input"))
            system = body.get("instructions")
        else:
            raw_messages = body.get("messages") or []
            system = "\n".join(_text_of(m.get("content")) for m in raw_messages if m.get("role") in ("system", "developer")) or None
            messages = [_chat_message(m) for m in raw_messages if m.get("role") not in ("system", "developer")]
        return ModelRequest(
            system=system,
            messages=messages,
            tools=list(body.get("tools") or []),
            params={k: v for k, v in body.items() if k not in NOT_PARAMS},
            stream=bool(body.get("stream")),
            raw=body,
            headers=kept,
            path=path,
        )

    def parse_response(self, body=None, chunks=None, status=200, headers=None) -> ModelResponse:
        obj = body
        if obj is None and chunks:
            text = "".join(c.data for c in chunks)
            events = sse_events(text)
            obj = assemble_responses_stream(events) or assemble_chat_stream(events)
            if obj is None:
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    obj = {"error": {"message": text}}
        obj = obj or {}
        content: list[ContentBlock] = []
        stop = None
        if obj.get("object") == "response" or "output" in obj:
            for item in obj.get("output") or []:
                block = _response_item(item)
                if block is not None:
                    content.append(block)
            stop = obj.get("status")
        elif obj.get("choices"):
            choice = obj["choices"][0]
            content = _chat_message(choice.get("message") or {}).content
            stop = choice.get("finish_reason")
        usage = {k: v for k, v in (obj.get("usage") or {}).items() if isinstance(v, int)}
        return ModelResponse(content=content, stop_reason=stop, usage=usage, raw=body if body is not None else (obj if not chunks else None), chunks=chunks, status=status, headers=dict(headers or {}))

    def tool_results(self, request: ModelRequest) -> list[ToolResult]:
        results: list[ToolResult] = []
        for message in reversed(request.messages):
            if message.role == "assistant":
                break
            for b in message.content:
                if b.type == "tool_result":
                    results.append(ToolResult(b.tool_call_id or "", [ContentBlock("text", text=b.text)], bool(b.is_error)))
        results.reverse()
        return results

    # -- rendering -----------------------------------------------------------------------------

    def render_response(self, response: ModelResponse, stream: bool) -> bytes | list[StreamChunk]:
        responses_api = bool(response.raw and (response.raw.get("object") == "response" or "output" in response.raw))
        if stream:
            if response.chunks:
                return response.chunks
            return spread(self._synthesize_responses_events(response) if responses_api else self._synthesize_chat_events(response))
        body = response.raw
        if body is None and response.chunks:
            events = sse_events("".join(c.data for c in response.chunks))
            body = assemble_responses_stream(events) or assemble_chat_stream(events)
        if body is None:
            body = self._synthesize_chat_body(response)
        return json.dumps(body).encode()

    def render_error(self, status: int, message: str) -> bytes:
        return json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": None}}).encode()

    def _synthesize_chat_body(self, response: ModelResponse) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": response.text() or None}
        calls = [
            {"id": b.tool_call_id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input or {})}}
            for b in response.content
            if b.type == "tool_use"
        ]
        if calls:
            msg["tool_calls"] = calls
        return {
            "id": "chatcmpl-tidebox",
            "object": "chat.completion",
            "created": 0,
            "model": "tidebox",
            "choices": [{"index": 0, "message": msg, "finish_reason": response.stop_reason or ("tool_calls" if calls else "stop")}],
            "usage": response.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _synthesize_chat_events(self, response: ModelResponse) -> list[str]:
        body = self._synthesize_chat_body(response)
        msg = body["choices"][0]["message"]
        head = {"id": body["id"], "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
        out = []
        if msg.get("content"):
            out.append(sse_chunk({**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": msg["content"]}, "finish_reason": None}]}))
        for i, call in enumerate(msg.get("tool_calls") or []):
            out.append(sse_chunk({**head, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **call}]}, "finish_reason": None}]}))
        out.append(sse_chunk({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": body["choices"][0]["finish_reason"]}], "usage": body["usage"]}))
        out.append(sse_chunk("[DONE]"))
        return out

    def _synthesize_responses_events(self, response: ModelResponse) -> list[str]:
        body = response.raw or {}
        return [
            sse_chunk({"type": "response.created", "response": {**body, "output": [], "status": "in_progress"}}),
            sse_chunk({"type": "response.completed", "response": body}),
        ]
