"""Anthropic Messages API (`POST /v1/messages`), streamed or not.

Covers: system as string or blocks, tool_use/tool_result blocks, thinking blocks,
server-sent event streams (message_start ... message_stop), beta headers.
`count_tokens` requests are passed through but are not model calls.
"""

from __future__ import annotations

import json
from typing import Any

from tidebox.api.providers.base import Provider, scrub_headers, spread, sse_chunk, sse_events
from tidebox.api.tape import ContentBlock, Message, ModelRequest, ModelResponse, StreamChunk, ToolResult

KEPT_HEADERS = ("anthropic-beta", "anthropic-version", "user-agent", "x-app", "content-type")
NOT_PARAMS = ("system", "messages", "tools", "stream")


def _block(raw: dict[str, Any]) -> ContentBlock:
    t = raw.get("type")
    if t == "text":
        return ContentBlock("text", text=raw.get("text", ""), raw=raw)
    if t == "thinking":
        return ContentBlock("thinking", text=raw.get("thinking", ""), raw=raw)
    if t == "redacted_thinking":
        return ContentBlock("thinking", text=None, raw=raw)
    if t == "tool_use":
        return ContentBlock("tool_use", tool_call_id=raw.get("id"), name=raw.get("name"), input=raw.get("input") or {}, raw=raw)
    if t == "tool_result":
        content = raw.get("content")
        text = content if isinstance(content, str) else "".join(c.get("text", "") for c in content or [] if c.get("type") == "text")
        return ContentBlock("tool_result", text=text, tool_call_id=raw.get("tool_use_id"), is_error=bool(raw.get("is_error")), raw=raw)
    if t == "image":
        return ContentBlock("image", raw=raw)
    return ContentBlock("other", raw=raw)


def _blocks(content: Any) -> list[ContentBlock]:
    if isinstance(content, str):
        return [ContentBlock("text", text=content)]
    return [_block(c) for c in content or [] if isinstance(c, dict)]


def _system_text(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")


def assemble_stream(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rebuild the complete message object from a Messages API event stream."""
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    partial: dict[int, str] = {}
    for ev in events:
        t = ev.get("type")
        if t == "message_start":
            message = dict(ev.get("message") or {})
            message["content"] = []
        elif t == "content_block_start":
            blocks[ev["index"]] = dict(ev.get("content_block") or {})
            partial[ev["index"]] = ""
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            b = blocks.setdefault(ev["index"], {"type": "text", "text": ""})
            dt = d.get("type")
            if dt == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif dt == "input_json_delta":
                partial[ev["index"]] = partial.get(ev["index"], "") + d.get("partial_json", "")
            elif dt == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
            elif dt == "signature_delta":
                b["signature"] = d.get("signature", "")
            elif dt == "citations_delta":
                b.setdefault("citations", []).append(d.get("citation"))
        elif t == "content_block_stop":
            b = blocks.get(ev["index"])
            if b is not None and b.get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
                js = partial.get(ev["index"], "")
                if js:
                    try:
                        b["input"] = json.loads(js)
                    except json.JSONDecodeError:
                        b["input"] = {"_partial_json": js}
        elif t == "message_delta":
            if message is None:
                message = {"content": []}
            message.update(ev.get("delta") or {})
            usage = ev.get("usage")
            if usage:
                message.setdefault("usage", {}).update(usage)
        elif t == "error":
            if message is None:
                message = {"content": []}
            message["error"] = ev.get("error")
    if message is None:
        return None
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


def _usage(raw: dict[str, Any]) -> dict[str, int]:
    return {k: v for k, v in (raw.get("usage") or {}).items() if isinstance(v, int)}


class AnthropicProvider(Provider):
    name = "anthropic"
    paths = ("/v1/messages",)

    def parse_request(self, body: dict, headers: dict[str, str], path: str = "") -> ModelRequest:
        kept = {k: v for k, v in scrub_headers(headers).items() if k.lower() in KEPT_HEADERS}
        return ModelRequest(
            system=_system_text(body.get("system")),
            messages=[Message(m.get("role", "user"), _blocks(m.get("content"))) for m in body.get("messages") or []],
            tools=list(body.get("tools") or []),
            params={k: v for k, v in body.items() if k not in NOT_PARAMS},
            stream=bool(body.get("stream")),
            raw=body,
            headers=kept,
            path=path,
        )

    def parse_response(self, body=None, chunks=None, status=200, headers=None) -> ModelResponse:
        message = body
        if message is None and chunks:
            message = assemble_stream(sse_events("".join(c.data for c in chunks)))
            if message is None:  # not a stream after all (an error body sent without events)
                try:
                    message = json.loads("".join(c.data for c in chunks))
                except json.JSONDecodeError:
                    message = {"content": [], "error": {"message": "".join(c.data for c in chunks)}}
        message = message or {}
        return ModelResponse(
            content=_blocks(message.get("content")),
            stop_reason=message.get("stop_reason"),
            usage=_usage(message),
            raw=body if body is not None else (message if not chunks else None),
            chunks=chunks,
            status=status,
            headers=dict(headers or {}),
        )

    def tool_results(self, request: ModelRequest) -> list[ToolResult]:
        """The tool results in the messages after the last assistant turn. Usually that is one
        user message; Claude Code 2.1.299+ appends a `system` message after it (a token-budget
        note), so every trailing message is read, not just the last."""
        trailing: list = []
        for message in reversed(request.messages):
            if message.role == "assistant":
                break
            trailing.append(message)
        trailing.reverse()
        return [
            ToolResult(b.tool_call_id or "", [ContentBlock("text", text=b.text)], bool(b.is_error))
            for message in trailing
            for b in message.content
            if b.type == "tool_result"
        ]

    # -- rendering -----------------------------------------------------------------------------

    def render_response(self, response: ModelResponse, stream: bool) -> bytes | list[StreamChunk]:
        if stream:
            if response.chunks:
                return response.chunks
            return spread(self._synthesize_events(response))
        body = response.raw
        if body is None and response.chunks:
            body = assemble_stream(sse_events("".join(c.data for c in response.chunks)))
        if body is None:
            body = self._synthesize_body(response)
        return json.dumps(body).encode()

    def render_error(self, status: int, message: str) -> bytes:
        kind = "not_found_error" if status == 404 else "invalid_request_error" if status < 500 else "api_error"
        return json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()

    @staticmethod
    def _raw_block(b: ContentBlock) -> dict[str, Any]:
        if b.raw is not None:
            return b.raw
        if b.type == "text":
            return {"type": "text", "text": b.text or ""}
        if b.type == "thinking":
            return {"type": "thinking", "thinking": b.text or "", "signature": ""}
        if b.type == "tool_use":
            return {"type": "tool_use", "id": b.tool_call_id, "name": b.name, "input": b.input or {}}
        return {"type": "text", "text": b.text or ""}

    def _synthesize_body(self, response: ModelResponse) -> dict[str, Any]:
        return {
            "id": "msg_tidebox",
            "type": "message",
            "role": "assistant",
            "model": "tidebox",
            "content": [self._raw_block(b) for b in response.content],
            "stop_reason": response.stop_reason or ("tool_use" if response.tool_calls() else "end_turn"),
            "stop_sequence": None,
            "usage": response.usage or {"input_tokens": 0, "output_tokens": 0},
        }

    def _synthesize_events(self, response: ModelResponse) -> list[str]:
        body = self._synthesize_body(response)
        head = {k: v for k, v in body.items() if k not in ("content", "stop_reason", "stop_sequence")}
        head["content"] = []
        out = [sse_chunk({"type": "message_start", "message": head}, "message_start")]
        for i, block in enumerate(body["content"]):
            t = block.get("type")
            if t == "text":
                out.append(sse_chunk({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}, "content_block_start"))
                out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block.get("text", "")}}, "content_block_delta"))
            elif t == "thinking":
                out.append(sse_chunk({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}}, "content_block_start"))
                out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")}}, "content_block_delta"))
                out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": block.get("signature", "")}}, "content_block_delta"))
            elif t == "tool_use":
                start = {k: v for k, v in block.items() if k != "input"}
                start["input"] = {}
                out.append(sse_chunk({"type": "content_block_start", "index": i, "content_block": start}, "content_block_start"))
                out.append(sse_chunk({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}}, "content_block_delta"))
            else:
                out.append(sse_chunk({"type": "content_block_start", "index": i, "content_block": block}, "content_block_start"))
            out.append(sse_chunk({"type": "content_block_stop", "index": i}, "content_block_stop"))
        out.append(sse_chunk({"type": "message_delta", "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": body["usage"].get("output_tokens", 0)}}, "message_delta"))
        out.append(sse_chunk({"type": "message_stop"}, "message_stop"))
        return out
