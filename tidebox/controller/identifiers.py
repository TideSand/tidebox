"""Identifiers that differ between the recording and the replay.

A harness names its session, background commands, saved long outputs and subagents with fresh
random identifiers and tells the model about them in tool results. A recorded response that
refers to one names the recorded identifier, which does not exist in the replay. By comparing
each tool result the replay reports with the recorded one, learn which new identifier stands
for which recorded one, and substitute them in the responses served from then on.

Two results are compared token by token: when they contain the same number of identifier-like
tokens in the same positions and some differ, each differing pair becomes a mapping.

Substitution touches only the string values inside a response (text, tool input values, the
streamed JSON's string fields), matches whole tokens, and never rewrites JSON structure or
`signature` fields, so a mapping can never produce a malformed response: an earlier version
replaced raw text inside serialized JSON and broke two replays when a learned token overlapped
JSON syntax.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from tidebox.api.tape import ModelResponse, StreamChunk, ToolCall, ToolResult

IDENTIFIER = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # uuid
    r"|(?<![0-9a-zA-Z])[0-9a-f]{6,}(?![0-9a-zA-Z])"  # hex run (task ids, hashes)
    r"|(?<![0-9a-zA-Z_])[a-z]+_[0-9a-z]*[0-9][0-9a-z]*(?![0-9a-zA-Z_])"  # prefixed ids with a digit (task_ab12cd); not plain identifiers like tool_call
    r"|(?<![0-9a-zA-Z])\d{5,}(?![0-9a-zA-Z])"  # pids, ports, timestamps
)
_SKIP_KEYS = {"signature"}  # opaque fields a token may occur in by chance


class IdentifierMap:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self._pattern: re.Pattern | None = None

    def learn(self, recorded: ToolResult, observed: ToolResult) -> None:
        self.learn_text(recorded.text(), observed.text())

    def learn_text(self, recorded: str, observed: str) -> None:
        if recorded == observed:
            return
        a = IDENTIFIER.findall(recorded)
        b = IDENTIFIER.findall(observed)
        if not a or len(a) != len(b):
            return
        for x, y in zip(a, b):
            if x != y and x not in self.mapping:
                self.mapping[x] = y
                self._pattern = None

    # -- substitution ------------------------------------------------------------------------

    def _compiled(self) -> re.Pattern:
        if self._pattern is None:
            alternatives = "|".join(re.escape(k) for k in sorted(self.mapping, key=len, reverse=True))
            self._pattern = re.compile(rf"(?<![0-9A-Za-z_])(?:{alternatives})(?![0-9A-Za-z_])")
        return self._pattern

    def substitute(self, text: str) -> str:
        """Whole-token replacement in plain text, each token replaced once (no chains)."""
        if not self.mapping or not text:
            return text
        return self._compiled().sub(lambda m: self.mapping[m.group(0)], text)

    def _mentions(self, text: str) -> bool:
        return bool(self.mapping) and bool(text) and self._compiled().search(text) is not None

    def _sub_json(self, value: Any) -> Any:
        """Substitutes in the string values of a parsed JSON structure; keys and `signature`
        fields are left alone."""
        if isinstance(value, str):
            return self.substitute(value)
        if isinstance(value, list):
            return [self._sub_json(v) for v in value]
        if isinstance(value, dict):
            return {k: (v if k in _SKIP_KEYS else self._sub_json(v)) for k, v in value.items()}
        return value

    def _sub_chunk(self, data: str) -> str:
        """Substitutes inside a streamed chunk: each complete `data:` line is parsed, rewritten
        in its string values and re-serialized; a line a chunk boundary cut in two, or anything
        that is not JSON, is left as it is (a token split across chunks cannot be matched)."""
        out = []
        for line in data.split("\n"):
            if line.startswith("data: ") and self._mentions(line):
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                out.append("data: " + json.dumps(self._sub_json(payload), ensure_ascii=False, separators=(",", ":")))
            else:
                out.append(line)
        return "\n".join(out)

    def apply_call(self, call: ToolCall) -> ToolCall:
        if not self.mapping or not self._mentions(json.dumps(call.input, ensure_ascii=False)):
            return call
        return ToolCall(call.id, call.name, self._sub_json(call.input))

    def apply(self, response: ModelResponse) -> ModelResponse:
        """A copy of the response with recorded identifiers replaced by the replay's. Responses
        that mention none are returned as they are."""
        if not self.mapping:
            return response
        serialized = json.dumps({"content": [b.raw if b.raw is not None else (b.input if b.input is not None else b.text) for b in response.content],
                                 "chunks": [c.data for c in response.chunks or []], "raw": response.raw}, ensure_ascii=False)
        if not self._mentions(serialized):
            return response
        out = copy.deepcopy(response)
        for block in out.content:
            if block.text:
                block.text = self.substitute(block.text)
            if block.input is not None:
                block.input = self._sub_json(block.input)
            if block.raw is not None:
                block.raw = self._sub_json(block.raw)
        if out.raw is not None:
            out.raw = self._sub_json(out.raw)
        if out.chunks:
            out.chunks = [StreamChunk(c.at_ms, self._sub_chunk(c.data)) for c in out.chunks]
        return out
