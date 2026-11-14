"""Where the replay is in the tape.

One cursor per replay, one position per thread. `next(thread)` returns the step to serve and
advances; `fork(parent, tool_call_id)` locates the child thread the tape recorded for that
call. Re-serving the current step when a client retries is `current()`.
"""

from __future__ import annotations

import time

from tidebox.api.tape import Step, Tape, Thread


class Cursor:
    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.position: dict[str, int] = {t.id: 0 for t in tape.threads}
        self.phase: dict[str, tuple[str, float]] = {}  # thread -> ("model" | "tools" | "done", since)
        self.forked: set[str] = set()

    def next(self, thread_id: str) -> Step | None:
        thread = self.tape.thread(thread_id)
        i = self.position[thread_id]
        if i >= len(thread.steps):
            self.phase[thread_id] = ("done", time.time())
            return None
        self.position[thread_id] = i + 1
        self.phase[thread_id] = ("tools", time.time())
        return thread.steps[i]

    def current(self, thread_id: str) -> Step | None:
        """The step most recently served on a thread, for retries."""
        i = self.position[thread_id]
        return self.tape.thread(thread_id).steps[i - 1] if i > 0 else None

    def waiting(self, thread_id: str) -> None:
        self.phase[thread_id] = ("model", time.time())

    def fork(self, parent_thread_id: str, tool_call_id: str) -> Thread | None:
        for child in self.tape.children_of(parent_thread_id, tool_call_id):
            if child.id not in self.forked:
                self.forked.add(child.id)
                return child
        return None

    def exhausted(self) -> bool:
        return all(self.position[t.id] >= len(t.steps) for t in self.tape.threads)

    def served(self) -> int:
        return sum(self.position.values())

    def state(self) -> dict:
        """Per thread: waiting on the model or running tools, since when, steps served."""
        return {
            t.id: {
                "phase": self.phase.get(t.id, ("idle", 0.0))[0],
                "since": self.phase.get(t.id, ("idle", 0.0))[1],
                "served": self.position[t.id],
                "recorded": len(t.steps),
            }
            for t in self.tape.threads
        }
