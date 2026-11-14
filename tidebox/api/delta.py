"""The delta tape format (version 2): each step stores what its request adds to the previous
request of the same thread, instead of the whole conversation again.

A harness sends the entire conversation with every model call, so in the plain format
(version 1) a tape of n steps holds about n²/2 messages: a two-hour run is gigabytes before
compression. In version 2 the first step of a thread stores its request in full and every
later step stores a *delta* against the request before it:

    {"$delta": {                                  the request, field by field
        "system":   {"$same": 1},                 unchanged: copy the previous step's value
        "messages": {"$shared": 57, "$new": [..]},  a list: the first 57 items are the previous
                                                  step's, then the new ones
        "tools":    {"$same": 1},
        "params":   {"$same": 1},
        "stream":   true,                         anything else is the value itself
        "raw":      {"$delta": {                  a dict: a nested delta, same rules
            "model": {"$same": 1}, "messages": {"$shared": 57, "$new": [..]}, ...}},
        "headers":  {"$same": 1},
        "path":     {"$same": 1}}}

A key absent from a delta is absent from the request. Keys keep the order they have in the
full request, so decoding gives back exactly the dict that was encoded. Responses, tool calls,
tool results and timing are per step already and are stored as before. Values that are
themselves dicts with a single `$same`, `$shared` or `$delta` key cannot be told from the
markers, so encoding refuses them; no provider's request body has such keys.

`encode_request`/`decode_request` work on one step's request dict given the previous step's;
`encode_tape`/`decode_tape` on a whole tape dict. `TapeWriter` streams a tape to disk one
step at a time, for writers that cannot hold a tape in memory (the recorder's spill mode, the
reader). Reading a version 2 tape step by step is `tidebox.api.lazy`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tidebox.api.io import open_tape_file, to_dict
from tidebox.api.tape import Tape

DELTA, SAME, SHARED, NEW = "$delta", "$same", "$shared", "$new"
_MARKERS = frozenset({DELTA, SAME, SHARED, NEW})
_SEP = (",", ":")


def is_delta(stored: Any) -> bool:
    """Whether a stored request is a delta (as opposed to a full request dict)."""
    return isinstance(stored, dict) and len(stored) == 1 and DELTA in stored


def encode_request(prev: dict | None, cur: dict) -> dict:
    """The form `cur` takes in the tape: itself when there is no previous request in the
    thread, else a delta against `prev`."""
    if prev is None:
        _check_plain(cur)
        return cur
    return {DELTA: _encode_dict(prev, cur)}


def decode_request(prev: dict | None, stored: dict) -> dict:
    """The full request dict, from what the tape stores and the previous request decoded."""
    if not is_delta(stored):
        return stored
    if prev is None:
        raise ValueError("a delta request with no previous request in its thread")
    return _decode_dict(prev, stored[DELTA])


def _encode_dict(prev: dict, cur: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in cur.items():
        if k in prev:
            p = prev[k]
            if p == v:
                out[k] = {SAME: 1}
                continue
            if isinstance(p, list) and isinstance(v, list):
                n = _common_prefix(p, v)
                if n:
                    out[k] = {SHARED: n, NEW: v[n:]}
                    continue
            if isinstance(p, dict) and isinstance(v, dict) and p and v:
                out[k] = {DELTA: _encode_dict(p, v)}
                continue
        _check_plain(v)
        out[k] = v
    return out


def _decode_dict(prev: dict, delta: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in delta.items():
        if isinstance(v, dict) and len(v) <= 2 and v:
            if SAME in v and len(v) == 1:
                out[k] = prev[k]
                continue
            if SHARED in v and NEW in v:
                out[k] = prev[k][: v[SHARED]] + v[NEW]
                continue
            if DELTA in v and len(v) == 1:
                out[k] = _decode_dict(prev[k], v[DELTA])
                continue
        out[k] = v
    return out


def _common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _check_plain(value: Any) -> None:
    """Refuses a value a decoder would take for a marker."""
    if isinstance(value, dict):
        if 0 < len(value) <= 2 and set(value) <= _MARKERS and (SAME in value or SHARED in value or DELTA in value):
            raise ValueError(f"cannot store a dict with only the keys {sorted(value)}: they are delta markers")
        for v in value.values():
            _check_plain(v)
    elif isinstance(value, list):
        for v in value:
            _check_plain(v)


# -- whole tapes -------------------------------------------------------------------------------


def encode_tape(tape: Tape) -> dict:
    """The version 2 dict of a tape: `to_dict` with every request after a thread's first stored
    as a delta."""
    data = to_dict(tape)
    data["version"] = 2
    for thread in data["threads"]:
        prev = None
        for step in thread["steps"]:
            full = step["request"]
            step["request"] = encode_request(prev, full)
            prev = full
    return data


def decode_tape(data: dict) -> dict:
    """A version 2 tape dict with every request in full (the shape of version 1, version kept)."""
    for thread in data.get("threads") or []:
        prev = None
        for step in thread.get("steps") or []:
            full = decode_request(prev, step.get("request") or {})
            step["request"] = full
            prev = full
    return data


class TapeWriter:
    """Streams a tape file: head, then threads and their steps one at a time, so the whole
    tape is never in memory. In version 2 each step's request is delta-encoded against the
    previous step's of the thread; the caller passes requests in full. The bytes written are
    what `write_tape` would produce for the same tape."""

    def __init__(self, path: Path | str, version: int, meta: dict) -> None:
        self.path = Path(path)
        if self.path.suffix in (".xz", ".gz"):
            raise ValueError(f"{self.path}: compressed tapes are not supported; write a plain .json")
        self.version = version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._f = open_tape_file(self._tmp, "w")
        self._f.write('{"version":' + json.dumps(version) + ',"meta":' + json.dumps(meta, ensure_ascii=False, separators=_SEP) + ',"threads":[')
        self._threads = 0
        self._steps = 0
        self._prev: dict | None = None
        self._in_thread = False

    def begin_thread(self, thread_id: str, parent: tuple[str, str] | list | None) -> None:
        if self._in_thread:
            self.end_thread()
        self._f.write(("," if self._threads else "") + '{"id":' + json.dumps(thread_id) + ',"parent":' + json.dumps(list(parent) if parent else None) + ',"steps":[')
        self._threads += 1
        self._steps = 0
        self._prev = None
        self._in_thread = True

    def write_step(self, step: dict) -> None:
        """`step` is a full step dict (`to_dict(Step)`); its request is stored as the format
        requires. The dict is not modified."""
        full = step["request"]
        stored = encode_request(self._prev, full) if self.version >= 2 else full
        out = {k: (stored if k == "request" else v) for k, v in step.items()}
        self._f.write(("," if self._steps else "") + json.dumps(out, ensure_ascii=False, separators=_SEP))
        self._steps += 1
        self._prev = full

    def write_step_encoded(self, js: str) -> None:
        """A step already serialized (and, in version 2, already delta-encoded) by this class
        elsewhere: appended as it is. The caller then tells the writer the step's full request
        with `set_previous`, when more steps follow in the thread."""
        self._f.write(("," if self._steps else "") + js)
        self._steps += 1

    def set_previous(self, full_request: dict | None) -> None:
        self._prev = full_request

    def end_thread(self) -> None:
        if self._in_thread:
            self._f.write("]}")
            self._in_thread = False

    def close(self) -> None:
        if self._f is None:
            return
        self.end_thread()
        self._f.write("]}")
        self._f.close()
        self._f = None
        os.replace(self._tmp, self.path)

    def __enter__(self) -> TapeWriter:
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
        elif self._f is not None:
            self._f.close()
            self._f = None
            self._tmp.unlink(missing_ok=True)
