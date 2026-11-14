"""The tape: one recorded agent run.

A tape is a set of threads. A thread is one conversation with the model: the main agent is one
thread, each subagent (Claude Code's Agent tool, Codex's delegation) is another, linked to the
tool call that spawned it. A thread is a list of steps; a step is one model call and what the
harness did with its response.

    Tape
      meta            harness, provider, model, task, when
      threads[]
        Thread
          id, parent (thread id + tool call id that spawned it)
          steps[]
            Step
              request     what the harness sent (normalized + raw)
              response    what the model answered (normalized + raw, with stream timing)
              tool_calls  parsed from the response
              tool_results  taken from the *next* request of the same thread
              timing      request sent, first byte, last byte, next request sent

Everything is stored twice: normalized (provider-independent, what the analysis code
reads) and raw (the provider's bytes, what the controller serves back to a real harness in
simulator mode). The normalized form is derived from the raw form by `providers/`, never the
other way round.

Times are milliseconds since `TapeMeta.started_at`.

On disk a tape is one plain JSON object (see `io.py`), shaped so that it can be one row of a
Hugging Face dataset without transformation. Since version 2 each step stores its request as
a delta against the previous step's of the same thread (`delta.py`): a harness repeats the
whole conversation in every call, and storing it once keeps a tape linear in the run's length
instead of quadratic. Version 1 tapes, with every request in full, are still read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TAPE_VERSION = 2  # what new tapes are written as
KNOWN_VERSIONS = (1, 2)  # 1: every request in full; 2: requests after a thread's first as deltas


@dataclass
class TapeMeta:
    tape_id: str
    harness: str  # "claude-code", "codex", ... 
    provider: str  # "anthropic", "openai" — selects the provider adapter
    model: str
    started_at: str  # ISO 8601
    harness_version: str | None = None
    task: dict[str, Any] = field(default_factory=dict)  # free-form: benchmark, task id, image
    replay_of: str | None = None  # tape_id of the recording this tape is a replay of, if any
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class ContentBlock:
    """One block of a message, provider-independent."""

    type: str  # text | thinking | tool_use | tool_result | image | other
    text: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    is_error: bool | None = None
    raw: Any = None

    @staticmethod
    def text_block(text: str) -> ContentBlock:
        return ContentBlock(type="text", text=text)


@dataclass
class Message:
    role: str  # user | assistant | system | tool
    content: list[ContentBlock] = field(default_factory=list)

    def text(self) -> str:
        return "".join(b.text or "" for b in self.content if b.type == "text")


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_call_id: str
    content: list[ContentBlock] = field(default_factory=list)
    is_error: bool = False
    duration_ms: int | None = None  # measured by whoever ran the tool, when it was measured

    def text(self) -> str:
        return "".join(b.text or "" for b in self.content if b.type == "text")

    @staticmethod
    def of_text(tool_call_id: str, text: str, is_error: bool = False, duration_ms: int | None = None) -> ToolResult:
        return ToolResult(tool_call_id, [ContentBlock.text_block(text)], is_error, duration_ms)


@dataclass
class ModelRequest:
    system: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # model, max_tokens, temperature, thinking...
    stream: bool = False
    raw: dict[str, Any] = field(default_factory=dict)  # the body as sent, minus credentials
    headers: dict[str, str] = field(default_factory=dict)  # the subset worth keeping
    path: str = ""  # the endpoint, e.g. /v1/messages

    def last_assistant(self) -> Message | None:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message
        return None

    def first_user_text(self) -> str | None:
        for message in self.messages:
            if message.role == "user":
                return message.text()
        return None


@dataclass
class StreamChunk:
    """One piece of a streamed response and when it arrived."""

    at_ms: int
    data: str


@dataclass
class ModelResponse:
    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] | None = None  # the complete (non-streamed) body
    chunks: list[StreamChunk] | None = None  # the stream, for byte-exact and time-exact replay
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        return "".join(b.text or "" for b in self.content if b.type == "text")

    def tool_calls(self) -> list[ToolCall]:
        return [ToolCall(b.tool_call_id or "", b.name or "", b.input or {}) for b in self.content if b.type == "tool_use"]


@dataclass
class Timing:
    request_at: int = 0
    first_byte_at: int | None = None
    last_byte_at: int | None = None
    next_request_at: int | None = None  # when the thread's next step was sent; tools ran in between


@dataclass
class Step:
    index: int
    request: ModelRequest
    response: ModelResponse
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    timing: Timing = field(default_factory=Timing)


@dataclass
class Thread:
    id: str
    parent: tuple[str, str] | None = None  # (thread id, tool call id) that spawned this thread
    steps: list[Step] = field(default_factory=list)


@dataclass
class Tape:
    version: int
    meta: TapeMeta
    threads: list[Thread] = field(default_factory=list)

    def main_thread(self) -> Thread:
        for thread in self.threads:
            if thread.parent is None:
                return thread
        raise ValueError("tape has no main thread")

    def thread(self, thread_id: str) -> Thread:
        for thread in self.threads:
            if thread.id == thread_id:
                return thread
        raise KeyError(thread_id)

    def children_of(self, thread_id: str, tool_call_id: str) -> list[Thread]:
        return [t for t in self.threads if t.parent == (thread_id, tool_call_id)]

    def steps(self) -> list[tuple[Thread, Step]]:
        """Every step of every thread, in order of request time."""
        return sorted(((t, s) for t in self.threads for s in t.steps), key=lambda ts: ts[1].timing.request_at)
