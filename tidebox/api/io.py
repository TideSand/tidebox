"""Reading, writing and validating tapes.

On disk a tape is one plain JSON file matching `schema.json`: a single object, so a tape can
be one row of a Hugging Face dataset as-is. A recording rewrites the file (atomically, via a
temporary file and rename) after every step it records, so a tape of a run that is still going
or that was cut short is always readable.

A tape is always a plain `.json` file: nothing in TideBox compresses or decompresses one,
so every tool can seek into a tape as it is.

Two versions exist. Version 1 stores every request in full, so a tape grows with the square of
its length (every request repeats the conversation). Version 2 stores a step's request as a
delta against the previous step's (`delta.py`). `read_tape` reads both and returns the same
`Tape`; `write_tape` writes the version the `Tape` says it is (new tapes: `TAPE_VERSION`).
Reading a large tape step by step without loading it is `lazy.py`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import types
import typing
from pathlib import Path
from typing import Any

from tidebox.api.tape import KNOWN_VERSIONS, Tape


def open_tape_file(path: Path | str, mode: str):
    """Opens a tape file for text I/O. Compressed files are not accepted: a `.xz` or `.gz`
    tape from before the plain-only rule must be inflated (`xz -d`) first."""
    path = Path(path)
    if path.suffix in (".xz", ".gz"):
        raise ValueError(f"{path}: compressed tapes are not supported; inflate it first (xz -d) and use the .json")
    return open(path, mode, encoding="utf-8")


def read_tape(path: Path | str) -> Tape:
    with open_tape_file(path, "r") as f:
        data = json.load(f)
    if (data.get("version") or 1) >= 2:
        from tidebox.api.delta import decode_tape  # delta.py imports this module

        data = decode_tape(data)
    return from_dict(data)


def write_tape(tape: Tape, path: Path | str) -> None:
    """Writes to `<path>.tmp` then renames, so readers never see a half-written file."""
    path = Path(path)
    if path.suffix in (".xz", ".gz"):
        raise ValueError(f"{path}: compressed tapes are not supported; write a plain .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tape.version >= 2:
        from tidebox.api.delta import encode_tape

        payload = encode_tape(tape)
    else:
        payload = to_dict(tape)
    with open_tape_file(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def validate(tape: Tape) -> list[str]:
    """Problems found, empty when the tape is well-formed. Checks the cross-references: tool
    results refer to tool calls of their step, threads refer to existing parents, timings are
    monotone within a thread."""
    problems: list[str] = []
    if tape.version not in KNOWN_VERSIONS:
        problems.append(f"version {tape.version}, known versions {', '.join(map(str, KNOWN_VERSIONS))}")
    ids = {t.id for t in tape.threads}
    if len(ids) != len(tape.threads):
        problems.append("duplicate thread ids")
    if not any(t.parent is None for t in tape.threads) and tape.threads:
        problems.append("no main thread (every thread has a parent)")
    for thread in tape.threads:
        if thread.parent is not None:
            parent_id, call_id = thread.parent
            if parent_id not in ids:
                problems.append(f"thread {thread.id}: parent thread {parent_id} does not exist")
            else:
                calls = {c.id for s in tape.thread(parent_id).steps for c in s.tool_calls}
                if call_id not in calls:
                    problems.append(f"thread {thread.id}: parent tool call {call_id} does not exist")
        last = -1
        for i, step in enumerate(thread.steps):
            if step.index != i:
                problems.append(f"thread {thread.id}: step {i} has index {step.index}")
            if step.timing.request_at < last:
                problems.append(f"thread {thread.id}: step {i} goes back in time")
            last = step.timing.request_at
            call_ids = {c.id for c in step.tool_calls}
            for result in step.tool_results:
                if result.tool_call_id not in call_ids:
                    problems.append(f"thread {thread.id}: step {i} has a result for unknown call {result.tool_call_id}")
    return problems


# ---------------------------------------------------------------------------------------------
# dataclass <-> plain dict, driven by the type hints in tape.py


def to_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_dict(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [to_dict(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    return value


def from_dict(data: dict) -> Tape:
    return _from(Tape, data)


def from_dict_as(hint: Any, data: Any) -> Any:
    """Builds one tape dataclass (a Step, a ModelRequest, a TapeMeta...) from its dict form,
    the way `from_dict` builds a whole Tape."""
    return _from(hint, data)


def _from(hint: Any, data: Any) -> Any:
    if data is None:
        return None
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        for arg in args:
            try:
                return _from(arg, data)
            except (TypeError, ValueError, KeyError):
                continue
        raise TypeError(f"{data!r} fits none of {args}")
    if origin is list:
        (item,) = typing.get_args(hint)
        return [_from(item, v) for v in data]
    if origin is tuple:
        return tuple(_from(a, v) for a, v in zip(typing.get_args(hint), data))
    if origin is dict or hint is dict:
        return dict(data)
    if origin is typing.Literal:
        return data
    if hint is Any:
        return data
    if dataclasses.is_dataclass(hint):
        if not isinstance(data, dict):
            raise TypeError(f"expected object for {hint.__name__}, got {type(data).__name__}")
        hints = typing.get_type_hints(hint)
        kwargs = {}
        for f in dataclasses.fields(hint):
            if f.name in data:
                kwargs[f.name] = _from(hints[f.name], data[f.name])
        return hint(**kwargs)
    if hint in (str, int, float, bool):
        if hint is int and isinstance(data, bool):
            raise TypeError("bool is not int")
        if not isinstance(data, hint) and not (hint is float and isinstance(data, int)):
            raise TypeError(f"expected {hint.__name__}, got {type(data).__name__}")
        return data
    return data
