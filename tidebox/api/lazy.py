"""A tape read on demand: one streaming pass indexes where each step's bytes are, and a step is
parsed only when something asks for it.

`read_tape` parses the whole file into objects, which costs several times the tape's size in
memory. A replay controller needs one step at a time, so `LazyTape` keeps, per step, its byte
range in the file plus what the matcher needs to pick it, about 200 bytes, and loads the step
itself when it is served. Tapes are plain JSON, never compressed, so steps are read by seeking.

Both tape versions are read. In version 2 (`delta.py`) a step's request is a delta against
the previous step's, so the index is built from the deltas alone, tracking per thread only
what the conversation key needs (the first user message, the position and key of each
assistant message, the tool count), and a loaded step's request is a `DeferredRequest`:
reconstructed from the thread's deltas the first time one of its fields is read, and never
if nothing reads it, which is the case when serving a replay. Reconstruction walks the thread
from its first step, keeping the last reconstructed request of a couple of threads so that
reading steps in order costs one delta each.

    with LazyTape.open("run.json") as tape:
        tape.meta, tape.threads[0].parent, len(tape.threads[0].steps)
        step = tape.threads[0].steps[7]                 # parsed now, from its byte range
        step.request.messages                           # version 2: reconstructed now

`LazyTape` quacks like `Tape` for the controller, the comparison and validation: `meta`,
`threads` (each with `id`, `parent` and a `steps` sequence), `thread(id)`, `main_thread()`.
"""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tidebox.api.delta import DELTA, NEW, SHARED, decode_request, is_delta
from tidebox.api.io import from_dict_as
from tidebox.api.providers import get_provider
from tidebox.api.providers.base import Provider
from tidebox.api.tape import Message, ModelRequest, ModelResponse, Step, TapeMeta, Timing, ToolCall, ToolResult

_CHUNK = 1 << 20


def _open_plain(path: Path | str):
    path = Path(path)
    if path.suffix in (".xz", ".gz"):
        raise ValueError(f"{path}: compressed tapes are not supported; inflate it first (xz -d) and use the .json")
    return open(path, "rb")


def read_head(path: Path | str) -> tuple[TapeMeta, str, Step]:
    """The tape's meta, its first thread's id and that thread's first step, without indexing
    the rest: what a driver needs to launch a harness (model, labels, the instruction). A
    thread's first step stores its request in full in every version."""
    with _open_plain(path) as f:
        sc = _Scanner(f)
        sc.expect("{")
        sc.key("version")
        sc.value()
        sc.key("meta")
        meta, _, _ = sc.value()
        sc.key("threads")
        sc.expect("[")
        sc.expect("{")
        sc.key("id")
        tid, _, _ = sc.value()
        sc.key("parent")
        sc.value()
        sc.key("steps")
        sc.expect("[")
        step, _, _ = sc.value()
    return from_dict_as(TapeMeta, meta), tid, from_dict_as(Step, step)


def tape_version(path: Path | str) -> int:
    """The format version a tape file declares, read from its first bytes."""
    with _open_plain(path) as f:
        sc = _Scanner(f)
        sc.expect("{")
        sc.key("version")
        version, _, _ = sc.value()
    return int(version)


@dataclass
class StepRef:
    """Where a step lives and what the matcher needs without loading it."""
    thread: str
    index: int
    offset: int  # byte offset of the step object in the uncompressed file
    length: int
    key: str  # provider.conversation_key of its request
    opening: bool  # no assistant message yet: the first request of a thread
    tools: int  # tools offered in the request
    tool_calls: int
    # the step's timing, so timelines can be drawn from the index alone
    request_at: int = 0
    last_byte_at: int | None = None
    next_request_at: int | None = None


class _Scanner:
    """Incremental JSON reader over a binary file: decodes one value at a time from a text
    window, tracking byte offsets so values can be re-read later."""

    def __init__(self, f) -> None:
        self.f = f
        self.buf = ""  # decoded text not yet consumed
        self.buf_start = 0  # byte offset of buf[0] in the file
        self.eof = False
        self.decoder = json.JSONDecoder()
        import codecs

        self._utf8 = codecs.getincrementaldecoder("utf-8")()

    def _fill(self) -> bool:
        if self.eof:
            return False
        data = self.f.read(_CHUNK)
        if not data:
            self.eof = True
            self.buf += self._utf8.decode(b"", final=True)
            return False
        self.buf += self._utf8.decode(data)
        return True

    def _ensure(self, n: int) -> None:
        while len(self.buf) < n and self._fill():
            pass

    def peek(self) -> str:
        self.skip_ws()
        self._ensure(1)
        return self.buf[:1]

    def skip_ws(self) -> None:
        while True:
            self._ensure(1)
            if self.buf[:1] in (" ", "\n", "\r", "\t", ","):
                self._consume(1)
            else:
                return

    def expect(self, literal: str) -> None:
        self.skip_ws()
        self._ensure(len(literal))
        if not self.buf.startswith(literal):
            raise ValueError(f"expected {literal!r} at byte {self.buf_start}, found {self.buf[:40]!r}")
        self._consume(len(literal))

    def _consume(self, n: int) -> None:
        self.buf_start += len(self.buf[:n].encode("utf-8"))
        self.buf = self.buf[n:]

    def value(self) -> tuple[Any, int, int]:
        """The next JSON value, with its (byte offset, byte length) in the file."""
        self.skip_ws()
        while True:
            try:
                obj, end = self.decoder.raw_decode(self.buf)
                break
            except json.JSONDecodeError:
                if not self._fill():
                    raise
        offset = self.buf_start
        length = len(self.buf[:end].encode("utf-8"))
        self._consume(end)
        return obj, offset, length

    def key(self, name: str) -> None:
        self.expect(f'"{name}":')


class _KeyState:
    """What the conversation key of a thread's next request depends on, kept small and
    updated from each stored request, full or delta, without materializing it: the first user
    message's text, the position and key of every assistant message, and the tool count."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.n = 0  # messages so far
        self.first_user: tuple[int, str] | None = None  # (position, text)
        self.assistants: list[tuple[int, str]] = []  # (position, key as the last assistant message)
        self.tools = 0

    def update(self, stored: dict) -> None:
        if is_delta(stored):
            body = stored[DELTA]
            m = body.get("messages")
            if isinstance(m, list):
                self._reset_messages(m)
            elif isinstance(m, dict) and SHARED in m:
                self._truncate(m[SHARED])
                self._extend(m[NEW])
            t = body.get("tools")
            if isinstance(t, list):
                self.tools = len(t)
            elif isinstance(t, dict) and SHARED in t:
                self.tools = t[SHARED] + len(t[NEW])
        else:
            self._reset_messages(stored.get("messages") or [])
            self.tools = len(stored.get("tools") or [])

    def _reset_messages(self, messages: list[dict]) -> None:
        self.n, self.first_user, self.assistants = 0, None, []
        self._extend(messages)

    def _truncate(self, n: int) -> None:
        self.assistants = [a for a in self.assistants if a[0] < n]
        if self.first_user is not None and self.first_user[0] >= n:
            self.first_user = None
        self.n = min(self.n, n)

    def _extend(self, messages: list[dict]) -> None:
        for m in messages:
            msg = from_dict_as(Message, m)
            if msg.role == "assistant":
                self.assistants.append((self.n, self.provider.key_for(msg, None)))
            elif msg.role == "user" and self.first_user is None:
                self.first_user = (self.n, msg.text())
            self.n += 1

    def key(self) -> str:
        if self.assistants:
            return self.assistants[-1][1]
        return self.provider.key_for(None, self.first_user[1] if self.first_user else None)

    def opening(self) -> bool:
        return not self.assistants


_REQUEST_FIELDS = frozenset(f.name for f in dataclasses.fields(ModelRequest))


class DeferredRequest(ModelRequest):
    """A step's request reconstructed from its thread's deltas the first time a field is
    read. Equal to, and reported as, a `ModelRequest`, so code that handles requests sees no
    difference; a step whose request nobody reads costs nothing beyond its own bytes."""

    def __init__(self, loader: Callable[[], ModelRequest]) -> None:  # noqa: D107 - not a dataclass init
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_loaded", False)

    @property
    def __class__(self):  # noqa: D105 - lets the dataclass __eq__ compare across the two classes
        return ModelRequest

    @property
    def loaded(self) -> bool:
        return object.__getattribute__(self, "_loaded")

    def _materialize(self) -> None:
        if not object.__getattribute__(self, "_loaded"):
            object.__setattr__(self, "_loaded", True)
            request = object.__getattribute__(self, "_loader")()
            for name in _REQUEST_FIELDS:
                object.__setattr__(self, name, getattr(request, name))

    def __getattribute__(self, name: str):
        if name in _REQUEST_FIELDS:
            object.__getattribute__(self, "_materialize")()
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value) -> None:
        if name in _REQUEST_FIELDS:
            object.__getattribute__(self, "_materialize")()
        object.__setattr__(self, name, value)


class LazySteps:
    """The steps of one thread, parsed on access. Supports len(), iteration and indexing,
    including negative indexes, which is what the controller and the comparison use."""

    def __init__(self, tape: LazyTape, refs: list[StepRef]) -> None:
        self._tape = tape
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __iter__(self) -> Iterator[Step]:
        for ref in self.refs:
            yield self._tape.load(ref)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._tape.load(r) for r in self.refs[i]]
        return self._tape.load(self.refs[i])

    def __bool__(self) -> bool:
        return bool(self.refs)


@dataclass
class LazyThread:
    id: str
    parent: tuple[str, str] | None
    steps: LazySteps


class LazyTape:
    REQUEST_CACHE_THREADS = 2  # threads whose last reconstructed request is kept

    def __init__(self, path: Path | str, provider_name: str | None = None) -> None:
        self.path = Path(path)
        if self.path.suffix in (".xz", ".gz"):
            raise ValueError(f"{self.path}: compressed tapes are not supported; inflate it first (xz -d) and use the .json")
        self.version: int = 0
        self.meta: TapeMeta
        self.threads: list[LazyThread] = []
        self.refs: list[StepRef] = []
        self._refs_by_thread: dict[str, list[StepRef]] = {}
        self._request_cache: OrderedDict[str, tuple[int, dict]] = OrderedDict()  # thread id -> (step index, its full request dict)
        self.reconstructions = 0  # delta applications so far, for tests and diagnostics
        self._scan(provider_name)
        self._fh = open(self.path, "rb")  # noqa: SIM115 - closed in close()
        index: dict[str, list[tuple[str, int]]] = {}
        openings: list[tuple[str, int, int]] = []
        opening_key: dict[str, str] = {}
        for ref in self.refs:
            index.setdefault(ref.key, []).append((ref.thread, ref.index))
            if ref.index == 0 and ref.opening:
                openings.append((ref.thread, 0, ref.tools))
                opening_key[ref.thread] = ref.key
        self._matcher_cache = (index, openings, opening_key)  # what controller.matcher.Matcher builds from a full tape

    @classmethod
    def open(cls, path: Path | str) -> LazyTape:
        return cls(path)

    def __enter__(self) -> LazyTape:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None

    # -- Tape-like surface ---------------------------------------------------------------------

    def thread(self, thread_id: str) -> LazyThread:
        for t in self.threads:
            if t.id == thread_id:
                return t
        raise KeyError(thread_id)

    def main_thread(self) -> LazyThread:
        for t in self.threads:
            if t.parent is None:
                return t
        return self.threads[0]

    def load(self, ref: StepRef) -> Step:
        """The step, parsed from its bytes. A delta-encoded request becomes a `DeferredRequest`."""
        data = self._step_dict(ref)
        stored = data.get("request") or {}
        if not is_delta(stored):
            return from_dict_as(Step, data)
        return Step(
            int(data.get("index", ref.index)),
            DeferredRequest(lambda: from_dict_as(ModelRequest, self.request_dict(ref.thread, ref.index))),
            from_dict_as(ModelResponse, data.get("response") or {}),
            from_dict_as(list[ToolCall], data.get("tool_calls") or []),
            from_dict_as(list[ToolResult], data.get("tool_results") or []),
            from_dict_as(Timing, data.get("timing") or {}),
        )

    def request_dict(self, thread_id: str, index: int) -> dict:
        """The full request dict of a step, reconstructed from the thread's stored requests up
        to it. Continues from the thread's last reconstruction when that one is earlier."""
        refs = self._refs_by_thread[thread_id]
        cached = self._request_cache.get(thread_id)
        if cached is not None and cached[0] <= index:
            start, current = cached[0] + 1, cached[1]
        else:
            start, current = 0, None
        for i in range(start, index + 1):
            current = decode_request(current, self._step_dict(refs[i]).get("request") or {})
            self.reconstructions += 1
        self._request_cache[thread_id] = (index, current)
        self._request_cache.move_to_end(thread_id)
        while len(self._request_cache) > self.REQUEST_CACHE_THREADS:
            self._request_cache.popitem(last=False)
        return current

    def _step_dict(self, ref: StepRef) -> dict:
        self._fh.seek(ref.offset)
        return json.loads(self._fh.read(ref.length).decode("utf-8"))

    # -- building ------------------------------------------------------------------------------

    def _scan(self, provider_name: str | None) -> None:
        with open(self.path, "rb") as f:
            sc = _Scanner(f)
            sc.expect("{")
            sc.key("version")
            self.version, _, _ = sc.value()
            sc.key("meta")
            meta, _, _ = sc.value()
            self.meta = from_dict_as(TapeMeta, meta)
            provider = get_provider(provider_name or self.meta.provider)
            sc.key("threads")
            sc.expect("[")
            while sc.peek() != "]":
                sc.expect("{")
                sc.key("id")
                tid, _, _ = sc.value()
                sc.key("parent")
                parent, _, _ = sc.value()
                sc.key("steps")
                sc.expect("[")
                refs: list[StepRef] = []
                state = _KeyState(provider)
                while sc.peek() != "]":
                    step, offset, length = sc.value()
                    state.update(step.get("request") or {})
                    timing = step.get("timing") or {}
                    refs.append(StepRef(tid, int(step.get("index", len(refs))), offset, length, state.key(), state.opening(), state.tools,
                                        len(step.get("tool_calls") or []),
                                        int(timing.get("request_at") or 0), timing.get("last_byte_at"), timing.get("next_request_at")))
                sc.expect("]")
                sc.expect("}")
                self.threads.append(LazyThread(tid, tuple(parent) if parent else None, LazySteps(self, refs)))
                self._refs_by_thread[tid] = refs
                self.refs.extend(refs)
            sc.expect("]")


def spill_path(tape_path: Path | str) -> Path:
    """Where a spilling recorder (`api.recorder`) appends a tape's steps while it records:
    `replay.json` -> `replay.json.steps.jsonl`. The file is removed once the tape is assembled."""
    p = Path(tape_path)
    return p.with_name(p.name + ".steps.jsonl")


def spilled_refs(path: Path | str) -> list[StepRef]:
    """The steps a spilling recorder has written so far (`<tape>.steps.jsonl`, one
    `<thread id>\\t<step json>` line per step), as `StepRef`s: what a run that is still going
    has on disk, for timelines drawn while it runs. A step is spilled once complete, when its
    thread's next request has brought its tool results, so the last step of each thread is
    not there yet; nor is the tape's meta. Only `thread`, `index`, `tool_calls` and the timing
    are filled in (`offset` and `length` are the line's): the conversation key is left empty,
    as working it out needs the thread's requests from the start. Times are milliseconds since
    the tape's start, as in the tape. A line still being written is ignored."""
    refs: list[StepRef] = []
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            length = len(line)
            if line.endswith(b"\n"):
                tid, _, data = line.partition(b"\t")
                try:
                    step = json.loads(data)
                except ValueError:
                    step = None
                if step is not None:
                    timing = step.get("timing") or {}
                    refs.append(StepRef(tid.decode("utf-8"), int(step.get("index", 0)), offset, length, "", False, 0, len(step.get("tool_calls") or []),
                                        int(timing.get("request_at") or 0), timing.get("last_byte_at"), timing.get("next_request_at")))
            offset += length
    return refs
