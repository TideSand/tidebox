"""Turns model calls into a tape and keeps the output file current.

Used to record a real run through a proxy and by the controller (recording a replay), so a
replay produces a tape in exactly the format of the recording it was played from, and the two
can be compared step by step.

Per model call: extract tool calls from the response and tool results from the request, and
attach the call to a thread. A request continues the thread whose last response's key it
carries (the tool-call ids of its last assistant message); a request with no assistant message
opens a new thread. A new thread's parent is the (thread, tool call) whose call is still
unanswered and whose input mentions the new thread's first user message, when such a call
exists. Callers that already know the thread pass it.

After every recorded step the whole tape is written to the output path (`io.write_tape`),
so the file is valid at all times. `finalize()` writes once more and returns the tape. Tapes
are written in `TAPE_VERSION` (2: requests as deltas, see `delta.py`); in spill mode each step
is delta-encoded as it goes to disk, against the previous request of its thread, which the
recorder keeps for that purpose.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from tidebox.api.delta import TapeWriter, encode_request
from tidebox.api.io import to_dict, write_tape
from tidebox.api.lazy import spill_path
from tidebox.api.providers import get_provider
from tidebox.api.providers.base import Provider
from tidebox.api.tape import (
    TAPE_VERSION,
    ModelRequest,
    ModelResponse,
    Step,
    Tape,
    TapeMeta,
    Thread,
    Timing,
    ToolCall,
    ToolResult,
)


class Recorder:
    def __init__(self, output: Path | str | None, meta: TapeMeta, provider: Provider | None = None, write_every_step: bool = True,
                 spill: bool = False):
        """With `spill`, a step is appended to `<output>.steps.jsonl` as soon as it is complete
        (when the thread's next request has delivered its tool results) and dropped from memory,
        and `finalize()` assembles the tape from that file; until then `lazy.spilled_refs`
        reads the steps so far, for watching a run as it goes. Memory then holds one step per
        thread instead of the whole tape, which matters for a controller replaying a long tape
        beside many others; the file written is the same as without spilling."""
        self.output = Path(output) if output else None
        self.tape = Tape(TAPE_VERSION, meta, [])
        self.provider = provider or get_provider(meta.provider)
        self.spill = bool(spill and self.output is not None)
        self.write_every_step = write_every_step and not self.spill
        self._lock = threading.Lock()
        self._by_key: dict[str, list[str]] = {}  # response key -> thread ids whose last step has it
        self._pending: list[tuple[str, ToolCall]] = []  # unanswered tool calls, (thread id, call)
        self._spilled: set[tuple[str, int]] = set()  # (thread id, step index) already on disk
        self._prev_request: dict[str, dict] = {}  # thread id -> full request dict of its last spilled step (version 2 deltas)
        self._spill_file = None
        if self.spill:
            self.spill_path = spill_path(self.output)
            self.spill_path.parent.mkdir(parents=True, exist_ok=True)
            self._spill_file = open(self.spill_path, "w", encoding="utf-8")  # noqa: SIM115 - closed in finalize

    # -- recording -----------------------------------------------------------------------------

    def record(
        self,
        request: ModelRequest,
        response: ModelResponse,
        timing: Timing,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        thread_id: str | None = None,
    ) -> tuple[Thread, Step]:
        """Tool calls and results default to what the provider adapter finds in the response
        and the request; the thread defaults to what the conversation implies. `tool_results`
        are the results of the *previous* step of the thread, as a request carries them."""
        with self._lock:
            thread = self._thread_for(request, thread_id)
            results = tool_results if tool_results is not None else self.provider.tool_results(request)
            if thread.steps:
                previous = thread.steps[-1]
                self._attach_results(previous, results)
                if previous.timing.next_request_at is None:
                    previous.timing.next_request_at = timing.request_at
                if self.spill:
                    self._spill_step(thread.id, previous)  # complete now: results and timing are in
            calls = tool_calls if tool_calls is not None else self.provider.tool_calls(response)
            step = Step(len(thread.steps), request, response, calls, [], timing)
            thread.steps.append(step)
            self._by_key.setdefault(self.provider.response_key(response), []).append(thread.id)
            self._pending.extend((thread.id, c) for c in calls)
            if self.write_every_step:
                self._write()
            return thread, step

    def retime(self, thread_id: str, response: ModelResponse, timing: Timing) -> bool:
        """A response served again: the harness sent the same request once more, as an agent
        does when its sandbox was restored from a checkpoint taken before the response reached
        it. While that response is still the thread's latest step and nothing has followed it,
        the step takes the new serve's timing: what the agent does with the response starts
        from the serve it acted on. Returns whether the step was retimed."""
        with self._lock:
            try:
                thread = self.tape.thread(thread_id)
            except KeyError:
                return False
            if not thread.steps:
                return False
            last = thread.steps[-1]
            if last.timing.next_request_at is not None or (thread_id, last.index) in self._spilled:
                return False
            if self.provider.response_key(last.response) != self.provider.response_key(response):
                return False
            last.timing.request_at, last.timing.first_byte_at, last.timing.last_byte_at = timing.request_at, timing.first_byte_at, timing.last_byte_at
            if self.write_every_step:
                self._write()
            return True

    def has_response(self, thread_id: str, response: ModelResponse) -> bool:
        """Whether this response was recorded in the thread already."""
        with self._lock:
            return thread_id in self._by_key.get(self.provider.response_key(response), [])

    def attach_results(self, thread_id: str, results: list[ToolResult], finished_at: int | None = None) -> None:
        """Results that arrive without a following request (the last step of a
        thread, whose results no later request carries)."""
        with self._lock:
            thread = self.tape.thread(thread_id)
            if thread.steps:
                self._attach_results(thread.steps[-1], results)
                if finished_at is not None and thread.steps[-1].timing.next_request_at is None:
                    thread.steps[-1].timing.next_request_at = finished_at
            if self.write_every_step:
                self._write()

    def new_thread(self, parent: tuple[str, str] | None, thread_id: str | None = None) -> Thread:
        with self._lock:
            return self._new_thread(parent, thread_id)

    def finalize(self) -> Tape:
        with self._lock:
            if self.spill:
                self._assemble()
            else:
                self._write()
            return self.tape

    # -- internals -----------------------------------------------------------------------------

    def _write(self) -> None:
        if self.output is not None:
            write_tape(self.tape, self.output)

    def _spill_step(self, thread_id: str, step: Step) -> None:
        """Appends a complete step to the spill file as `<thread id>\\t<json>` and empties it in
        memory. The step object stays in the thread so indexes and `steps[-1]` keep working."""
        if (thread_id, step.index) in self._spilled or self._spill_file is None:
            return
        data = to_dict(step)
        if self.tape.version >= 2:
            full = data["request"]
            data["request"] = encode_request(self._prev_request.get(thread_id), full)
            self._prev_request[thread_id] = full
        self._spill_file.write(thread_id + "\t" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._spill_file.flush()
        self._spilled.add((thread_id, step.index))
        step.request, step.response, step.tool_results = ModelRequest(), ModelResponse(), []

    def _assemble(self) -> None:
        """Writes the tape file from the spilled steps plus those still in memory, streaming so
        the whole tape is never held at once. The result is byte-for-byte what `write_tape`
        would produce from the same steps."""
        if self._spill_file is None:
            return
        self._spill_file.close()
        self._spill_file = None
        offsets: dict[str, list[tuple[int, int]]] = {}
        with open(self.spill_path, "rb") as f:
            pos = 0
            for line in f:
                tid = line.split(b"\t", 1)[0].decode()
                offsets.setdefault(tid, []).append((pos, len(line)))
                pos += len(line)
        with open(self.spill_path, "rb") as spill, TapeWriter(self.output, self.tape.version, to_dict(self.tape.meta)) as out:
            for thread in self.tape.threads:
                out.begin_thread(thread.id, thread.parent)
                for off, length in offsets.get(thread.id, []):
                    spill.seek(off)
                    out.write_step_encoded(spill.read(length).split(b"\t", 1)[1].rstrip(b"\n").decode("utf-8"))
                out.set_previous(self._prev_request.get(thread.id))
                for step in thread.steps:
                    if (thread.id, step.index) not in self._spilled:
                        out.write_step(to_dict(step))
        self.spill_path.unlink(missing_ok=True)

    def _attach_results(self, step: Step, results: list[ToolResult]) -> None:
        if not results:
            return
        call_ids = {c.id for c in step.tool_calls}
        step.tool_results = [r for r in results if r.tool_call_id in call_ids]
        answered = {r.tool_call_id for r in step.tool_results}
        self._pending = [(t, c) for t, c in self._pending if c.id not in answered]

    def _new_thread(self, parent: tuple[str, str] | None, thread_id: str | None = None) -> Thread:
        tid = thread_id or f"t{len(self.tape.threads)}"
        thread = Thread(tid, parent, [])
        self.tape.threads.append(thread)
        return thread

    def _thread_for(self, request: ModelRequest, thread_id: str | None) -> Thread:
        if thread_id is not None:
            try:
                return self.tape.thread(thread_id)
            except KeyError:
                return self._new_thread(None if not self.tape.threads else self._guess_parent(request), thread_id)
        key = self.provider.conversation_key(request)
        if not key.startswith("open:"):
            candidates = self._by_key.get(key) or []
            if candidates:
                # A retry of the same request continues the same thread; the most recent
                # thread with that key is the best guess when several have it.
                return self.tape.thread(candidates[-1])
            # Continues a conversation the tape has not seen (a harness that rewrote its
            # history, or a run recorded from the middle): treat as a new thread.
        return self._new_thread(None if not self.tape.threads else self._guess_parent(request))

    def _guess_parent(self, request: ModelRequest) -> tuple[str, str] | None:
        """The pending tool call whose input mentions the opening message: a subagent's prompt
        appears verbatim in the call that spawned it."""
        prompt = (request.first_user_text() or "").strip()
        if not prompt:
            return None
        for thread_id, call in reversed(self._pending):
            for value in call.input.values():
                if isinstance(value, str) and value.strip() and (value.strip() in prompt or prompt in value.strip()):
                    return (thread_id, call.id)
        return None
