"""Which recorded step answers a request (LLM replay, --llm).

A continuing request carries, in its last assistant message, the tool-call ids of the recorded
response that produced it: look up the step that followed. That works even when tool results
differ from the recording. An opening request is matched by the text of its first user
message, then by the size of its tool list against threads not yet opened. A request sent
again (a harness retries after a timeout) gets the same step again.
"""

from __future__ import annotations

from tidebox.api.providers.base import Provider
from tidebox.api.tape import ModelRequest, Step, Tape

# How many tools an opening request may have more or fewer than the recorded one and still
# be taken for it. The tool list changes only with the harness build, by a few tools at most.
TOOL_SLACK = 2


class Matcher:
    def __init__(self, tape: Tape, provider: Provider) -> None:
        self.tape = tape
        self.provider = provider
        self.served: dict[str, int] = {}
        self.opened: set[str] = set()
        cache = getattr(tape, "_matcher_cache", None)
        if cache is None:
            cache = self._build(tape, provider)
            tape._matcher_cache = cache  # noqa: SLF001 - the tape is shared by every Run of one controller  # type: ignore[attr-defined]
        self.index, self.openings, self._opening_key = cache

    @staticmethod
    def _build(tape: Tape, provider: Provider) -> tuple[dict[str, list[tuple[str, int]]], list[tuple[str, int, int]], dict[str, str]]:
        """Indexes every step by its conversation key and remembers, for each thread's opening
        request, its tool count, then drops the recorded requests' bodies: every request carries
        the whole conversation so far, so they are most of a tape's size, and after indexing
        only the responses are needed to serve a replay."""
        index: dict[str, list[tuple[str, int]]] = {}
        openings: list[tuple[str, int, int]] = []  # (thread id, step index, tool count)
        opening_key: dict[str, str] = {}
        for thread in tape.threads:
            for step in thread.steps:
                key = provider.conversation_key(step.request)
                index.setdefault(key, []).append((thread.id, step.index))
                if step.index == 0 and step.request.last_assistant() is None:
                    openings.append((thread.id, 0, len(step.request.tools)))
                    opening_key[thread.id] = key
                req = step.request
                req.messages, req.raw, req.tools, req.system, req.headers = [], {}, [], None, {}
        return index, openings, opening_key

    def match(self, request: ModelRequest) -> tuple[str, Step, str] | None:
        """(thread id, step, how it was matched) or None when nothing in the tape fits."""
        key = self.provider.conversation_key(request)
        candidates = self.index.get(key) or []
        if candidates:
            n = self.served.get(key, 0)
            how = "key"
            if n >= len(candidates):
                n, how = len(candidates) - 1, "retry"
            else:
                self.served[key] = n + 1
            thread_id, index = candidates[n]
            self.opened.add(thread_id)
            return thread_id, self.tape.thread(thread_id).steps[index], how
        if key.startswith("open:"):
            # An opening message that differs from the recording (the harness describes the
            # sandbox in it: date, kernel, git status). Take the unopened thread whose opening
            # request has the same number of tools, else the nearest within TOOL_SLACK: a
            # harness build can gain or lose a feature-gated tool between recording and replay.
            openings = [o for o in self.openings if o[0] not in self.opened]
            if openings:
                thread_id, index, ntools = min(openings, key=lambda o: abs(o[2] - len(request.tools)))
                gap = abs(ntools - len(request.tools))
                if gap <= TOOL_SLACK:
                    self.opened.add(thread_id)
                    self.served[self._opening_key[thread_id]] = 1
                    return thread_id, self.tape.thread(thread_id).steps[index], "tools" if gap == 0 else "tools~"
        return None
