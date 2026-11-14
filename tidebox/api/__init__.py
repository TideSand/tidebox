"""The tape format and everything the components agree on. Standard library only.

    tape.py       in-memory model of a tape (dataclasses)
    schema.json   the same, as a versioned JSON Schema for the on-disk form
    io.py         read, write, validate
    recorder.py   assembles model calls into a tape and keeps the file on disk current
    providers/    wire-format adapters: raw provider request/response <-> normalized tape records

The controller uses io (read), providers (serve raw bytes back) and recorder (record the replay).
"""

from tidebox.api.tape import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Step,
    StreamChunk,
    Tape,
    TapeMeta,
    Thread,
    Timing,
    ToolCall,
    ToolResult,
)

__all__ = [
    "ContentBlock",
    "Message",
    "ModelRequest",
    "ModelResponse",
    "Step",
    "StreamChunk",
    "Tape",
    "TapeMeta",
    "Thread",
    "Timing",
    "ToolCall",
    "ToolResult",
]
