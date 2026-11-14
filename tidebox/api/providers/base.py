"""What every provider adapter implements, plus the server-sent-events helpers they share."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from tidebox.api.tape import Message, ModelRequest, ModelResponse, StreamChunk, ToolCall, ToolResult

SCRUBBED_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie", "proxy-authorization"}


def scrub_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in SCRUBBED_HEADERS}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sse_events(text: str) -> list[dict[str, Any]]:
    """Parse a server-sent-events stream into its `data:` JSON payloads, in order.
    Non-JSON payloads (OpenAI's `[DONE]`) are returned as {"_raw": payload}."""
    events: list[dict[str, Any]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data_lines = [line[5:].lstrip() for line in block.split("\n") if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            events.append({"_raw": payload})
            continue
        events.append(obj if isinstance(obj, dict) else {"_raw": payload})
    return events


def sse_chunk(data: dict[str, Any] | str, event: str | None = None) -> str:
    body = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return (f"event: {event}\n" if event else "") + f"data: {body}\n\n"


def spread(chunks: list[str], every_ms: int = 0) -> list[StreamChunk]:
    return [StreamChunk(i * every_ms, c) for i, c in enumerate(chunks)]


class Provider(ABC):
    """A provider is an API shape (Anthropic Messages, OpenAI Chat Completions / Responses),
    not a harness."""

    name: str
    paths: tuple[str, ...] = ()  # endpoint suffixes that are model calls worth recording

    def is_model_call(self, path: str) -> bool:
        return any(path.rstrip("/").endswith(p) for p in self.paths)

    @abstractmethod
    def parse_request(self, body: dict, headers: dict[str, str], path: str = "") -> ModelRequest: ...

    @abstractmethod
    def parse_response(
        self, body: dict | None, chunks: list[StreamChunk] | None, status: int = 200, headers: dict[str, str] | None = None
    ) -> ModelResponse:
        """From a complete body, or from the stream's chunks; both when the caller has both."""
        ...

    def tool_calls(self, response: ModelResponse) -> list[ToolCall]:
        return response.tool_calls()

    @abstractmethod
    def tool_results(self, request: ModelRequest) -> list[ToolResult]:
        """The tool results carried by a request, i.e. the outcome of the previous step's calls."""
        ...

    @abstractmethod
    def render_response(self, response: ModelResponse, stream: bool) -> bytes | list[StreamChunk]:
        """The response as the provider would send it: raw when the tape has it, synthesized
        from the normalized content otherwise."""
        ...

    @abstractmethod
    def render_error(self, status: int, message: str) -> bytes: ...

    def content_type(self, stream: bool) -> str:
        return "text/event-stream" if stream else "application/json"

    def conversation_key(self, request: ModelRequest) -> str:
        """Identifies which recorded step a request continues: the tool-call ids of its last
        assistant message (independent of tool result content), the digest of that message
        when it has no tool calls, or the digest of the first user message for an opening
        request."""
        return self.key_for(request.last_assistant(), request.first_user_text())

    def key_for(self, last_assistant: Message | None, first_user_text: str | None) -> str:
        """`conversation_key` from its two ingredients, for readers that index a tape without
        materializing its requests (`tidebox.api.lazy`)."""
        if last_assistant is not None:
            ids = [b.tool_call_id for b in last_assistant.content if b.type == "tool_use" and b.tool_call_id]
            if ids:
                return "ids:" + ",".join(ids)
            return "text:" + digest([(b.type, b.text) for b in last_assistant.content])
        return "open:" + digest(first_user_text)

    def response_key(self, response: ModelResponse) -> str:
        """The key a request continuing this response will carry; see `conversation_key`."""
        ids = [b.tool_call_id for b in response.content if b.type == "tool_use" and b.tool_call_id]
        if ids:
            return "ids:" + ",".join(ids)
        return "text:" + digest([(b.type, b.text) for b in response.content if b.type in ("text", "tool_use")])
