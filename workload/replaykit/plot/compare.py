#!/usr/bin/env python3
"""Policies against the strict reference over wall-clock time: how replay runs under overcommit
policies progressed compared with the reference they replay.

    R=output/replay/algotune/run4-max-50
    plot.py compare --reference $R/strict-ref/strict-speed0 --reference-stat max --floor-ms 1000 --min-strict-ms 0 \\
        --strict-label strict --run $R/overcommit-1.0 --run $R/tidesand --family overcommit --no-recording-timeline

(scripts/analyze.sh runs the paper's compares.)

`--run` repeats, one line per run; `--labels` names them (default: the directory names). The
reference (`--reference`, `--recording` for short) is the recordings' directory, or a strict replay
of them on the same host, so that the reference shares the host and only the policy differs;
`--strict-label` then names it. A reference replay with several copies of a task (`--repeat`, or
the older `rep<k>/` layout) stands for the task by the median of each step's time over its
copies. The reference may still be running: a step is then compared once the reference has
completed it, so the comparison covers the tasks it has started and grows as it goes.
`--no-recording-timeline` (or `--hide-recording`) leaves the reference out of the figures; it
stays the on-time reference. `--at-throughput-of PREFIX=LABEL` (repeatable) puts the runs whose label starts with PREFIX at the
throughput of the run labelled LABEL in throughput-on-time (repetitions of that run at another
speed, whose own clock is not comparable): only their on-time share is theirs. `--family PREFIX` (repeatable) draws the runs whose label starts
with PREFIX as one family: one colour, with distinct line styles, in the figures over time, and
in throughput-on-time one marker, one legend entry and a line through their points, each point
labelled with the rest of the run's name. Figures go to `--out`, by default
`output/plots/<bench>/<run>/compare/<--name>` (name: `policies`), placed by the first run.

Six figures, x = hours since each run started:
    tasks-completed      tasks finished so far, strict and each overcommit run
    model-calls          model calls served so far (one per recorded step), strict and each run
    on-time-tool-calls   tool calls on time so far in each overcommit run; the strict run's
                         total is the reference
    on-time-rate         share of each overcommit run's tool calls so far that were on time
    resources            five panels over the run's clock: tasks completed; active sandboxes
                         (admitted, so started, and not yet finished); and the pool's CPU
                         (cores), memory (GB) and swap (GB): each task's telemetry averaged over
                         every `--resource-bin-min` (1) minute (CPU from the growth of its
                         cumulative counter, so a task present for part of a minute counts what
                         it used there), summed over the tasks
    throughput-on-time   throughput (tool calls finished per hour, every one; or with --throughput peak the most
                         model calls served in any minute) against the on-time share of tool
                         calls, one point per run (the whole run); `--points N` cuts each run into
                         N equal windows instead, one point per window, joined from the lowest
                         throughput to the highest; the strict recording sits at 100 %; the
                         plotted points also go to throughput-on-time.csv

A unit replayed by a later driver pass (its sandbox could not be created in the main pass, say,
and the run script was started again, which replays only the units without a complete replay) ran
alone after the run, so it is left out, and so that every run counts the same tasks, the union of
such units over all the runs compared is left out of each of them; a note names them. The
reference keeps all its tasks: it is matched per call, not counted as a run.

A tool call is *on time* when its execution time in the overcommit run is within
`--tolerance` (default 110 %) of the same call's execution time in the strict run, that time
floored at `--floor-ms` (default 10 s); calls whose strict execution took `--min-strict-ms`
(default 10 s) or less are left out of the on-time figures and counts altogether. The paper's
fidelity: the reference is the slowest of five strict speed-0 replays of each call
(`--reference <run>/strict-ref/strict-speed0 --reference-stat max`), 10 % tolerance, a 1 s floor,
every call counted (`--floor-ms 1000 --min-strict-ms 0`). Execution
time is the gap between the model's response and the harness's next request for that step,
which is what the tapes record; the calls of one step share that gap. When the run went
through Tidelet's LLM proxy, the proxy may have held the response back before the agent saw
it; that hold is not execution, so it is subtracted. The proxy logs each delivery with the
request body's digest and the hold (`tideletd.log` in the replay directory), and the
controller logs each served step with the same digest (`controller.log` in the task's
directory), which joins the two. Runs without those logs are measured hold-inclusive, and
`summary.csv` says so (`holds_known`). Per-call data goes to
`on-time-<label>.csv` next to the figures, and `summary.csv` has one row per run.

Both tape sets are read through the lazy index (`tidebox.api.lazy`), which carries each
step's timing, so no tape is loaded whole.

A task without its `meta.json` is unfinished: still running, or cut off when its run was stopped
(the controller then writes its tape, but the driver no `meta.json`). Its steps come from the tape,
or while it runs from the controller's spill file (`tape.json.steps.jsonl`); its telemetry from the
CSV written as it goes; its start from `status.json` or the driver's log. Unfinished tasks add their
tool calls, model calls and resource use so far, but no completion.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pickle
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from replaykit import layout
from replaykit.plot import style

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

STRICT, OVER = style.ORANGE, style.BLUE


def _iso(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _unit_start(meta: dict, unit: Path) -> float | None:
    """When a replayed unit started: its admission's running_at, its driver's started_at, or, for a
    unit whose driver wrote neither (an older layout, a cut-off task), the tape's own started_at."""
    s = ((meta.get("admission") or {}).get("running_at")) or (meta.get("replay") or {}).get("started_at")
    if not s:
        try:
            with open(unit / "tape.json", "rb") as f:
                m = re.search(rb'"started_at"\s*:\s*"([^"]+)"', f.read(8192))
            s = m.group(1).decode() if m else None
        except OSError:
            s = None
    return _iso(s) if s else None


def index_tape(path: str) -> list[dict]:
    """Per-step timing of a tape, from the lazy index: (thread, index, request_at, last_byte_at,
    next_request_at, tool_calls). A replay still running has no tape yet: its completed steps
    are read from the controller's spill file instead."""
    from tidebox.api.lazy import LazyTape, spill_path, spilled_refs

    if Path(path).exists():
        with LazyTape(path) as tape:
            refs = tape.refs
    else:
        refs = spilled_refs(spill_path(path)) if spill_path(path).exists() else []
    return [{"thread": r.thread, "index": r.index, "request_at": r.request_at, "last_byte_at": r.last_byte_at,
             "next_request_at": r.next_request_at, "tool_calls": r.tool_calls} for r in refs]


def proxy_holds(replay: Path) -> dict[str, int]:
    """Request-body digest -> milliseconds the proxy held that response, from the daemon's
    delivery log. The last delivery of a digest wins (a retried request is served again)."""
    holds: dict[str, int] = {}
    log = replay / "tideletd.log"
    if not log.exists():
        return holds
    pat = re.compile(r'msg="model response delivered".* body=([0-9a-f]{64}) .* heldMs=(\d+)')
    for line in open(log, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            holds[m.group(1)] = int(m.group(2))
    return holds


def proxy_rollbacks(replay: Path) -> tuple[dict[str, int], int]:
    """Request-body digest -> milliseconds the redo of that response's tool calls took, for the
    calls a policy rolled back (the daemon's `rollback ... killed=N` with N > 0, then `redo done
    ... tookMs=`), joined to the digest through the proxy's delivery line for the call; and the
    number of rollbacks that undid a call. Such a call's agent-visible execution ends at the kill
    (the agent sees an early failure, and the redo runs while its next response is held), so the
    call is judged by its redo, the run that did the work."""
    log = replay / "tideletd.log"
    if not log.exists():
        return {}, 0
    digest_of: dict[str, str] = {}
    undone: set[str] = set()
    redo_ms: dict[str, int] = {}
    deliv = re.compile(r'msg="model response delivered".* call=(\S+) body=([0-9a-f]{64})')
    roll = re.compile(r'msg=rollback .*call=(\S+) killed=(\d+)')
    redo = re.compile(r'msg="redo done" .*call=(\S+) .*tookMs=(\d+)')
    for line in open(log, encoding="utf-8", errors="replace"):
        if (m := deliv.search(line)):
            digest_of.setdefault(m.group(1), m.group(2))
        elif (m := roll.search(line)):
            if int(m.group(2)) > 0:
                undone.add(m.group(1))
        elif (m := redo.search(line)):
            redo_ms[m.group(1)] = int(m.group(2))
    return {digest_of[c]: redo_ms[c] for c in undone if c in redo_ms and c in digest_of}, len(undone)


def step_digests(task_dir: Path) -> dict[tuple[str, int], str]:
    """(thread, step) -> the digest of the request the controller served it for."""
    out: dict[tuple[str, int], str] = {}
    log = task_dir / "controller.log"
    if not log.exists():
        return out
    pat = re.compile(r"run \S+: (\S+) step (\d+) by \S+ \(\d+ tool calls\) body=([0-9a-f]{64})")
    for line in open(log, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            out[(m.group(1), int(m.group(2)))] = m.group(3)
    return out


def nested_holds(steps: list[dict], digests: dict[tuple[str, int], str], holds: dict[str, int]) -> dict[tuple[str, int], int]:
    """(thread, step) -> milliseconds the proxy held the responses of the threads nested in that
    step's tool window: a tool that calls the model itself (WebFetch's summariser, a Task subagent
    the agent waits for) runs its own thread between the step's response and the next request, and
    a hold on one of its responses is time the step's agent spent waiting on the proxy, not on the
    tool. A thread whose requests all fall inside one window is credited to the innermost such
    window; a background subagent spans several and is credited to none."""
    spans: dict[str, list] = {}
    for s in steps:
        if s["request_at"] is None:
            continue
        end = s["next_request_at"] if s["next_request_at"] is not None else (s["last_byte_at"] or s["request_at"])
        d = digests.get((s["thread"], s["index"]))
        sp = spans.setdefault(s["thread"], [s["request_at"], end, 0])
        sp[0] = min(sp[0], s["request_at"]); sp[1] = max(sp[1], end); sp[2] += holds.get(d, 0) if d else 0
    out: dict[tuple[str, int], int] = {}
    for thread, (first, last, held) in spans.items():
        if not held:
            continue
        inner = None
        for s in steps:
            if s["thread"] == thread or s["next_request_at"] is None or s["last_byte_at"] is None:
                continue
            if s["last_byte_at"] <= first and last <= s["next_request_at"]:
                width = s["next_request_at"] - s["last_byte_at"]
                if inner is None or width < inner[0]:
                    inner = (width, s["thread"], s["index"])
        if inner is not None:
            out[inner[1:]] = out.get(inner[1:], 0) + held
    return out


MAX_CORES = 32  # a task's counter growing faster than this many cores (its limit is 8) is a misreading, not usage


def cpu_growth(ts: np.ndarray, cpu_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The CPU a task's cumulative `cpu_ms` counter grew by between consecutive good samples:
    (interval starts, interval ends, core-seconds, index of each interval's end sample). A sample
    whose counter is below the last good one, or grew from it faster than MAX_CORES, is a misreading
    (a zero while the sandbox is rolled back, a doubled reading) and is skipped: the next good
    sample's growth is measured from the last good one, spanning it. Four such samples in a row are
    a counter that started again, and the fourth is the new baseline."""
    starts, ends, work, idx = [], [], [], []
    last, skipped = 0, 0
    for i in range(1, len(ts)):
        if ts[i] <= ts[last]:
            continue
        growth = cpu_ms[i] - cpu_ms[last]
        if not 0 <= growth <= MAX_CORES * 1000 * (ts[i] - ts[last]):
            skipped += 1
            if skipped >= 4:
                last, skipped = i, 0
            continue
        starts.append(ts[last])
        ends.append(ts[i])
        work.append(growth / 1000)
        idx.append(i)
        last, skipped = i, 0
    return np.array(starts), np.array(ends), np.array(work), np.array(idx, dtype=int)


def spread(starts: np.ndarray, ends: np.ndarray, amounts: np.ndarray, t0: float, bin_s: float, n: int) -> np.ndarray:
    """Sums amounts into n bins of bin_s from t0, each interval's amount split between the bins
    it overlaps by the share of its length in each."""
    out = np.zeros(n)
    if not len(starts):
        return out
    lo = np.floor((starts - t0) / bin_s).astype(int)
    hi = np.floor((ends - t0) / bin_s).astype(int)
    one = (lo == hi) & (lo >= 0) & (lo < n)
    out += np.bincount(lo[one], amounts[one], minlength=n)[:n]
    for s, e, a, l, h in zip(starts[~one], ends[~one], amounts[~one], lo[~one], hi[~one]):
        for k in range(max(l, 0), min(h, n - 1) + 1):
            overlap = min(e, t0 + (k + 1) * bin_s) - max(s, t0 + k * bin_s)
            if overlap > 0:
                out[k] += a * overlap / (e - s)
    return out


def telemetry_bins(args: tuple[str, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """`_telemetry_bins`, cached on disk like `analyse` (by the file's path, size and modification
    time, t0 and the bin width): parsing every task's telemetry is most of the resources figure."""
    path, t0, bin_s = args
    if os.environ.get("TIDEBOX_PLOT_CACHE") == "off":
        return _telemetry_bins(args)
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = hashlib.sha256(pickle.dumps(("telemetry", 1, os.path.realpath(path), st.st_size, st.st_mtime_ns, t0, bin_s))).hexdigest()
    cache = ANALYSE_CACHE / "telemetry" / f"{key}.pickle"
    try:
        with open(cache, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        pass
    r = _telemetry_bins(args)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    except OSError:
        pass
    return r


def _telemetry_bins(args: tuple[str, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One task's telemetry per bin on the run's clock: (bin index, cores, memory GB, swap GB),
    each a time average over the whole bin: the CPU is the growth of the task's cumulative
    counter in the bin over the bin's length (cpu_growth), memory and swap are each sample's
    value held until the next one. The CSV's absolute `time` column is used; args = (path, t0,
    bin_s)."""
    path, t0, bin_s = args
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return None
    with f:
        reader = csv.DictReader(f)
        ts, cpu_ms, mem, swp = [], [], [], []
        for row in reader:
            t = row.get("time")
            if not t:
                continue
            try:
                when = dt.datetime.fromisoformat(t[:26].rstrip("Z") + "+00:00").timestamp() if t.endswith("Z") else dt.datetime.fromisoformat(t).timestamp()
            except ValueError:
                continue
            ts.append(when)
            cpu_ms.append(float(row.get("cpu_ms") or 0))
            mem.append(float(row.get("memory_bytes") or 0) / 2**30)
            swp.append(float(row.get("swap_bytes") or 0) / 2**30)
    if len(ts) < 2:
        return None
    ts, cpu_ms, mem, swp = (np.array(x) for x in (ts, cpu_ms, mem, swp))
    starts, ends, work, idx = cpu_growth(ts, cpu_ms)
    keep = ends > t0  # intervals before the run started are impossible by construction
    starts, ends, work, idx = starts[keep], ends[keep], work[keep], idx[keep]
    if not len(starts):
        return None
    n = int((ends.max() - t0) // bin_s) + 1
    held = ends - starts
    return (np.arange(n), spread(starts, ends, work, t0, bin_s, n) / bin_s,
            spread(starts, ends, mem[idx] * held, t0, bin_s, n) / bin_s, spread(starts, ends, swp[idx] * held, t0, bin_s, n) / bin_s)


def harness_starts(replay: Path) -> dict[str, float]:
    """When Claude Code started in each task's sandbox, from the driver logs (the last start wins:
    a task redone later replaces its earlier attempt)."""
    starts: dict[str, float] = {}
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \S+ INFO (\S+): Claude Code in ")
    for log in sorted([*replay.glob("stdout*.log"), *replay.glob("replay*.log")]):
        for line in open(log, encoding="utf-8", errors="replace"):
            m = pat.match(line)
            if m:
                starts[m.group(2)] = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.UTC).timestamp()
    return starts


RUN_COLORS = [style.BLUE, style.GREEN, style.VERMILLION, style.GREY, style.BLACK, style.PURPLE, style.SKY]
# The throughput figure's own style per run: colour, marker, line.
RUN_STYLES = [("#C8703A", "^", "-"), ("#5B9BD5", "s", "--"), ("#4E9A5A", "D", "-."), ("#8E6BB5", "o", ":"), ("#B5A642", "v", "-"),
              (style.PURPLE, "P", "--"), (style.SKY, "X", "-.")]
# A policy keeps its colour across figures (and matches progress.py's): the first run of each
# family takes its entry; every other run takes the remaining entries in order.
FAMILY_COLORS = {"strict": style.GREY, "overcommit": style.VERMILLION, "tidesand": style.GREEN}
FAMILY_STYLES = {"overcommit": RUN_STYLES[0], "tidesand": RUN_STYLES[2]}


def by_family(labels: list[str], palette: list, families: dict, group_of: list | None = None) -> list:
    """Each run's palette entry, its policy family's where it has one (see FAMILY_COLORS); a current
    run picks before a "-legacy" one, so the family colour stays with the current build. The runs
    of one `--family` group (the same `group_of` entry) all take the entry of the group's first run."""
    group_of = group_of or [None] * len(labels)
    out, taken, shared = [None] * len(labels), set(), {}
    for i in sorted(range(len(labels)), key=lambda i: labels[i].endswith("-legacy")):
        if group_of[i] in shared:
            out[i] = shared[group_of[i]]
            continue
        family = next((f for f in families if labels[i].startswith(f) and f not in taken), None)
        if family:
            taken.add(family)
            out[i] = families[family]
            if group_of[i] is not None:
                shared[group_of[i]] = out[i]
    rest = iter(p for p in palette if p not in out)
    for i, o in enumerate(out):
        if o is None:
            out[i] = shared.get(group_of[i]) or next(rest)
            if group_of[i] is not None:
                shared.setdefault(group_of[i], out[i])
    return out


def line_styles(group_of: list, cycle: tuple = ("-", "--", "-.", ":")) -> list:
    """A line style per run for the figures over time: a run on its own is solid; the runs of a
    `--family` group, which share a colour, take the cycle's styles in turn."""
    nth, out = {}, []
    for g in group_of:
        if g is None:
            out.append(cycle[0])
            continue
        nth[g] = nth.get(g, -1) + 1
        out.append(cycle[nth[g] % len(cycle)])
    return out


def _run_time(meta: dict, key: str) -> str | None:
    """A reference task's `started_at` or `finished_at`: the recording's, or a strict replay's own,
    which a replay's meta.json keeps under `replay`; None for one it has not got yet (the
    `finished_at` of a strict replay still running)."""
    return meta.get(key) or (meta.get("replay") or {}).get(key)


def replay_record(replay_dir: Path) -> tuple[dict, bool]:
    """A replay directory's record and whether the replay has ended: its meta.json once it has,
    before that its status.json (the driver's record so far), else nothing."""
    for name, ended in (("meta.json", True), ("status.json", False)):
        try:
            return json.loads((replay_dir / name).read_text()), ended
        except (OSError, ValueError):
            continue
    return {}, False


def recorded_task(replay_dir: Path) -> str:
    """The recording a replay directory plays back (see `layout.unit_task`)."""
    return layout.unit_task(replay_dir)


def replay_units(replay: Path) -> tuple[list[str], list[str]]:
    """The replay's units that have something to show, as paths relative to the run (both
    layouts, see `layout.units`): those that ended with a tape (meta.json and tape.json), and the
    unfinished ones, with a tape or spilled steps but no meta.json (running, or cut off by a
    stopped run)."""
    from tidebox.api.lazy import spill_path

    ended, unfinished = [], []
    for u in layout.units(replay):
        has_steps = (u.dir / "tape.json").exists() or spill_path(u.dir / "tape.json").exists()
        if (u.dir / "meta.json").exists():
            if (u.dir / "tape.json").exists():
                ended.append(u.rel)
        elif has_steps:
            unfinished.append(u.rel)
    return sorted(ended), sorted(unfinished)


_DRIVER_START = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \S+ INFO .*replaying \d+ sandboxes")


def driver_starts(replay: Path) -> list[float]:
    """When each driver pass of the run started, epoch seconds, from the run's replay.log: the
    driver logs "replaying N sandboxes" once per start, so a run resumed after a failed unit has two
    (the log's clock is the host's, UTC on the pool)."""
    out = []
    try:
        with open(replay / "replay.log", errors="replace") as fh:
            for line in fh:
                m = _DRIVER_START.match(line)
                if m:
                    out.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.UTC).timestamp())
    except OSError:
        pass
    return out


def rerun_units(replay: Path) -> set[str]:
    """The run's units replayed by a later driver pass: started at or after the second driver start
    in replay.log. Such a unit failed in the main pass and ran alone when the run script was
    started again, on an empty pool, so its calls say nothing about the policy."""
    starts = driver_starts(replay)
    if len(starts) < 2:
        return set()
    ended, running = replay_units(replay)
    out = set()
    for t in ended + running:
        meta = replay_record(replay / t)[0]
        when = _unit_start(meta, replay / t)
        if when is not None and when >= starts[1] - 1:
            out.add(t)
    return out


def reruns(replays: list[Path]) -> set[str]:
    """The units to leave out of every run compared: the union of each run's rerun units, so that
    all the runs count the same set of tasks; a note on stderr names them."""
    out: set[str] = set()
    for r in replays:
        units = rerun_units(r)
        if units:
            print(f"note: {r.name}: {len(units)} unit(s) replayed by a later driver pass, left out of every run: {', '.join(sorted(units))}", file=sys.stderr)
        out |= units
    return out


def counted(min_strict_ms: float) -> str:
    """Which tool calls count, in words for the figures: "tool calls over 30 s", or all of them."""
    return f"tool calls over {min_strict_ms / 1000:g} s" if min_strict_ms > 0 else "tool calls"


def slower(tolerance: float) -> str:
    """The tolerance in words for the figures: 1.05 -> "5% slower"."""
    return f"{round((tolerance - 1) * 100, 1):g}% slower"


def _merge_reference(copies: list[list[dict]], stat: str = "median") -> list[dict]:
    """The reference's steps for one task from the steps of each of its copies: the first copy's
    steps, each with `gap` set to the median (`stat` "median") or the slowest (`stat` "max") over
    the copies of that step's tool-turn gap (next request minus the response's last byte), so a
    repeated strict replay stands for the task by its typical time, or by the slowest time it
    was ever seen to take unloaded (the paper's fidelity: the five speed-0 strict runs, the
    slowest copy of each call)."""
    first = copies[0]
    gaps: dict[tuple[str, int], list[int]] = {}
    for steps in copies:
        for s in steps:
            if s["next_request_at"] is not None and s["last_byte_at"] is not None:
                gaps.setdefault((s["thread"], s["index"]), []).append(s["next_request_at"] - s["last_byte_at"])
    agg = max if stat == "max" else statistics.median
    out = []
    for s in first:
        g = gaps.get((s["thread"], s["index"]))
        out.append({**s, "gap": agg(g) if g else None})
    return out


ANALYSE_CACHE = Path(os.environ.get("TIDEBOX_PLOT_CACHE", str(Path.home() / ".cache" / "tidebox" / "analyse")))
_CACHE_FILES = ("meta.json", "status.json", "tape.json", "controller.log", "compare.json", "tape.telemetry.csv")


def _fingerprint(root: Path) -> list:
    """What `analyse` reads under a run (two levels: units, and the copies of a nested reference),
    as (path, size, mtime) for each file it opens: an unchanged fingerprint means an unchanged result."""
    out = []
    for d in sorted([root, *root.glob("*/"), *root.glob("*/*/")]):
        for name in _CACHE_FILES + ("run.json", "replay.log", "stdout.log", "tideletd.log", "proxy.log"):
            try:
                st = (d / name).stat()
            except OSError:
                continue
            out.append((str(d.relative_to(root)), name, st.st_size, st.st_mtime_ns))
    return out


def analyse(recording: Path, replay: Path, tolerance: float, floor_ms: float, min_strict_ms: float, jobs: int,
            exclude: set[str] | None = None, stat: str = "median") -> dict:
    """`_analyse`, cached on disk (TIDEBOX_PLOT_CACHE, default ~/.cache/tidebox/analyse) by its
    arguments and the size and modification time of every file under the reference and the run,
    so a figure redrawn over finished runs reads each run once; TIDEBOX_PLOT_CACHE=off disables it.
    `warm_cache` fills it for several runs at once."""
    if os.environ.get("TIDEBOX_PLOT_CACHE") == "off":
        return _analyse(recording, replay, tolerance, floor_ms, min_strict_ms, jobs, exclude, stat)
    path = _cache_path(recording, replay, tolerance, floor_ms, min_strict_ms, exclude, stat)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        pass
    a = _analyse(recording, replay, tolerance, floor_ms, min_strict_ms, jobs, exclude, stat)
    try:
        ANALYSE_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except OSError:
        pass
    return a


def warm_cache(recording: Path, replays: list[Path], tolerance: float, floor_ms: float, min_strict_ms: float, jobs: int,
               exclude: set[str] | None = None, parallel: int = 0, stat: str = "median") -> None:
    """Analyse the runs not yet in the cache concurrently, one worker thread per run (at most
    `parallel`, default the number of runs), each writing its own cache file; the later
    `analyse` calls then read the cache. The `jobs` tape-indexing workers are shared out among
    the concurrent runs. Nothing happens with the cache off."""
    if os.environ.get("TIDEBOX_PLOT_CACHE") == "off":
        return
    todo = [r for r in replays if not _cached(recording, r, tolerance, floor_ms, min_strict_ms, exclude, stat)]
    if len(todo) < 2:
        return
    n = min(len(todo), parallel or len(todo))
    per = max(1, jobs // n)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda r: analyse(recording, r, tolerance, floor_ms, min_strict_ms, per, exclude, stat), todo))


def _cache_path(recording: Path, replay: Path, tolerance: float, floor_ms: float, min_strict_ms: float, exclude: set[str] | None,
                stat: str = "median") -> Path:
    key = hashlib.sha256(pickle.dumps((ANALYSE_VERSION, str(recording.resolve()), str(replay.resolve()), tolerance, floor_ms,
                                       min_strict_ms, sorted(exclude or ()), stat, _fingerprint(recording.resolve()),
                                       _fingerprint(replay.resolve())))).hexdigest()
    return ANALYSE_CACHE / f"{key}.pickle"


def _cached(recording: Path, replay: Path, tolerance: float, floor_ms: float, min_strict_ms: float, exclude: set[str] | None,
            stat: str = "median") -> bool:
    return _cache_path(recording, replay, tolerance, floor_ms, min_strict_ms, exclude, stat).exists()


ANALYSE_VERSION = 3  # bump when `_analyse` changes what it returns, so older cache entries are not read


def _analyse(recording: Path, replay: Path, tolerance: float, floor_ms: float, min_strict_ms: float, jobs: int,
             exclude: set[str] | None = None, stat: str = "median") -> dict:
    """One replay run against the reference: completion, model-call and on-time event times.
    Tasks still running count with what they have recorded so far (see the module docstring).
    `exclude` names units left out (the reruns of every run compared, see `reruns`)."""
    from tidebox.api.lazy import spill_path

    ended, running = replay_units(replay)
    if exclude:
        ended = [t for t in ended if t not in exclude]
        running = [t for t in running if t not in exclude]
    starts = harness_starts(replay)
    rep_meta = {t: replay_record(replay / t)[0] for t in ended + running}
    # A running task is placed on the run's clock by its start: the driver's status.json, or,
    # from a driver that predates it, when Claude Code started (the driver's log).
    running = [t for t in running if starts.get(t) or (rep_meta[t].get("replay") or {}).get("started_at")]
    rec = {t: recorded_task(replay / t) for t in ended + running}  # replay unit -> the recording's task
    # The reference may itself be a strict replay still running, with one or several copies of
    # each task: a task counts once a copy has steps (its tape, or its controller's spill file)
    # and a start (meta.json, or status.json while it runs), and each of its steps is compared
    # once the reference has completed it.
    refs: dict[str, list[Path]] = {}
    for t in ended + running:
        dirs = [d for d in layout.reference_units(recording, rec[t])
                if ((d / "tape.json").exists() or spill_path(d / "tape.json").exists()) and _run_time(replay_record(d)[0], "started_at")]
        if dirs:
            refs[t] = dirs
    tasks = sorted(refs)
    # A task whose driver wrote no end (removed under a cutoff, or a driver that failed after the tape was
    # written) is unfinished: what it recorded counts, placed by its start.
    finished = [t for t in tasks if t in ended and (rep_meta[t].get("replay") or {}).get("finished_at")]
    rec_meta = {t: replay_record(refs[t][0])[0] for t in tasks}
    rec_start = {t: _iso(((rec_meta[t].get("harbor_result") or {}).get("agent_execution") or {}).get("started_at") or _run_time(rec_meta[t], "started_at")) for t in tasks}
    rec_end = {}  # only for the reference tasks that have ended
    for t in tasks:
        end = ((rec_meta[t].get("harbor_result") or {}).get("agent_execution") or {}).get("finished_at") or _run_time(rec_meta[t], "finished_at")
        if end:
            rec_end[t] = _iso(end)
    rep_start = {t: starts.get(t) or _unit_start(rep_meta[t], replay / t) for t in tasks}
    for t in [t for t in tasks if rep_start[t] is None]:
        print(f"warning: {t}: no start time in {replay / t}; left out", file=sys.stderr)
        tasks.remove(t); refs.pop(t, None); rep_start.pop(t)
    finished = [t for t in finished if t in rep_start]
    rep_end = {t: _iso(rep_meta[t]["replay"]["finished_at"]) for t in finished}
    t0_rec = min(_iso(_run_time(rec_meta[t], "started_at")) for t in tasks) if tasks else 0.0
    t0_rep = min(_iso((rep_meta[t].get("replay") or {}).get("started_at")) if (rep_meta[t].get("replay") or {}).get("started_at") else rep_start[t] for t in tasks) if tasks else 0.0

    ref_jobs = [(t, str(d / "tape.json")) for t in tasks for d in refs[t]]
    jobs_ = [p for _, p in ref_jobs] + [str(replay / t / "tape.json") for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        steps = list(pool.map(index_tape, jobs_))
    copies: dict[str, list[list[dict]]] = {}
    for (t, _), s in zip(ref_jobs, steps[:len(ref_jobs)]):
        copies.setdefault(t, []).append(s)
    rec_steps = {t: _merge_reference(copies[t], stat) for t in tasks}
    rep_steps = dict(zip(tasks, steps[len(ref_jobs):]))

    holds = proxy_holds(replay)
    redone, rollbacks = proxy_rollbacks(replay)
    rows = []  # per replayed tool call
    steps_seen = steps_joined = 0
    for t in tasks:
        rec_by = {(s["thread"], s["index"]): s for s in rec_steps[t]}
        digests = step_digests(replay / t) if holds else {}
        nested = nested_holds(rep_steps[t], digests, holds)
        for s in rep_steps[t]:
            r = rec_by.get((s["thread"], s["index"]))
            if r is None or s["next_request_at"] is None or s["last_byte_at"] is None or r["gap"] is None:
                continue
            steps_seen += 1
            digest = digests.get((s["thread"], s["index"]))
            held = (holds.get(digest, 0) if digest else 0) + nested.get((s["thread"], s["index"]), 0)
            if digest in holds:
                steps_joined += 1
            rep_gap = s["next_request_at"] - s["last_byte_at"]  # model answered -> next request, hold included
            rep_exec = rep_gap - held  # the agent's own time with the response: the tool calls
            rolled = digest in redone
            if rolled:
                rep_exec = redone[digest]  # rolled back: the redo is the run that did the work
            rec_gap = r["gap"]
            if rec_gap <= min_strict_ms:
                continue
            when = (rep_start[t] + s["next_request_at"] / 1000 - t0_rep) / 3600
            for _ in range(max(1, s["tool_calls"])):
                rows.append((when, rep_exec <= tolerance * max(rec_gap, floor_ms), t, s["thread"], s["index"], rec_gap, rep_exec, held, rep_gap, rolled))
    rows.sort()
    return {
        "tasks": tasks,
        "rec_tasks": [rec[t] for t in tasks],
        "rec_dirs": [refs[t][0] for t in tasks],
        "t0_rec": t0_rec, "t0_rep": t0_rep,
        "rec_calls": sorted((rec_start[t] + s["request_at"] / 1000 - t0_rec) / 3600 for t in tasks for s in rec_steps[t]),
        "rep_calls": sorted((rep_start[t] + s["request_at"] / 1000 - t0_rep) / 3600 for t in tasks for s in rep_steps[t]),
        "rec_done": sorted((rec_end[t] - t0_rec) / 3600 for t in rec_end),
        "rep_done": sorted((rep_end[t] - t0_rep) / 3600 for t in finished),
        # When each task's sandbox started (it was admitted), every task, running ones included:
        # with rep_done, how many sandboxes were active (started, not finished) at any time.
        "rec_started": sorted((rec_start[t] - t0_rec) / 3600 for t in tasks),
        "rep_started": sorted((rep_start[t] - t0_rep) / 3600 for t in tasks),
        # When each tool call finished (its step's next model request), every one, whatever
        # min_strict_ms counts for the on-time figure: the throughput figure's unit.
        "rep_tool_done": sorted((rep_start[t] + s["next_request_at"] / 1000 - t0_rep) / 3600 for t in tasks for s in rep_steps[t]
                                if s["tool_calls"] > 0 and s["next_request_at"] is not None for _ in range(s["tool_calls"])),
        "rec_tool_done": sorted((rec_start[t] + s["next_request_at"] / 1000 - t0_rec) / 3600 for t in tasks for s in rec_steps[t]
                                if s["tool_calls"] > 0 and s["next_request_at"] is not None for _ in range(s["tool_calls"])),
        "rec_tool_calls": sum(max(1, s["tool_calls"]) for t in tasks for s in rec_steps[t] if s["gap"] is not None and s["gap"] > min_strict_ms),
        # Each task's residency (sandbox start, end, CPU limit) and the host's cores, for the
        # occupied-host-hours throughput: a running task is resident until its last model call.
        "rep_residency": [((rep_start[t] - t0_rep) / 3600,
                           (rep_end[t] - t0_rep) / 3600 if t in rep_end else max([(rep_start[t] + s["request_at"] / 1000 - t0_rep) / 3600 for s in rep_steps[t]] + [(rep_start[t] - t0_rep) / 3600]),
                           _cpus(rep_meta[t])) for t in tasks],
        "rec_residency": [((rec_start[t] - t0_rec) / 3600,
                           (rec_end[t] - t0_rec) / 3600 if t in rec_end else max([(rec_start[t] + s["request_at"] / 1000 - t0_rec) / 3600 for s in rec_steps[t]] + [(rec_start[t] - t0_rec) / 3600]),
                           _cpus(rec_meta[t])) for t in tasks],
        "rep_cores": max([_cores(rep_meta[t]) for t in tasks] + [0.0]),
        "rec_cores": max([_cores(rec_meta[t]) for t in tasks] + [0.0]),
        "rows": rows,
        "holds_known": steps_joined / steps_seen if steps_seen else 0.0,
        "rollbacks": rollbacks,
        "complete": sum(1 for t in finished if (rep_meta[t].get("compare") or {}).get("complete")),
        "timed_out": sum(1 for t in finished if (rep_meta[t].get("harness") or {}).get("timed_out")),
        "running": [t for t in tasks if t not in finished],
    }


def _peak(hours: list[float], window_min: float, cutoff_min: float = 0.0) -> float:
    """The most events among `hours` (sorted, in hours) in any `window_min` minutes (within the
    first `cutoff_min` minutes when given), per minute: the peak rate of model calls served."""
    ts = [h * 60 for h in hours if not cutoff_min or h * 60 < cutoff_min]
    best = j = 0
    for i, t in enumerate(ts):
        while ts[j] <= t - window_min:
            j += 1
        best = max(best, i - j + 1)
    return round(best / window_min, 2)


def _cpus(meta: dict) -> float:
    """A unit's CPU limit (cores) from its meta.json; 1 when unknown."""
    try:
        return float(((meta.get("container") or {}).get("limits") or {}).get("cpus") or 1.0)
    except (TypeError, ValueError):
        return 1.0


def _cores(meta: dict) -> float:
    """The host's allocatable cores as the admission record saw them; 0 when unknown."""
    try:
        return float(((meta.get("admission") or {}).get("node") or {}).get("allocatable", {}).get("cpu_millicores") or 0) / 1000
    except (TypeError, ValueError):
        return 0.0


def _occupied(calls: list[float], residency: list[tuple[float, float, float]], cores: float, lo: float = 0.0, hi: float = float("inf"),
              floor: float = 0.0) -> float:
    """Model calls per occupied host-hour: the host is charged, at each moment of [lo, hi), the
    fraction min(1, sum of the resident sandboxes' CPU limits / cores) of an hour (at least
    `floor`), so a thinly loaded tail costs a fraction of a host. 0 without calls or cores."""
    n = sum(1 for h in calls if lo <= h < hi)
    if not n or cores <= 0 or not residency:
        return 0.0
    events = sorted([(max(a, lo), c) for a, b, c in residency if b > lo and a < hi] + [(min(b, hi), -c) for a, b, c in residency if b > lo and a < hi])
    hours = 0.0; load = 0.0; prev = events[0][0]
    for t, d in events:
        if t > prev:
            hours += (t - prev) * max(floor, min(1.0, load / cores))
            prev = t
        load += d
    return round(n / hours, 1) if hours > 0 else 0.0


def _bulk(calls: list[float], started: list[float], fraction: float, origin: float | None = None) -> float:
    """Model calls per hour over the time from the first sandbox start (`origin` when given and
    earlier) to the call that completes `fraction` of `calls` (sorted, in hours): the stragglers
    past that call set no span. 0 without calls."""
    import math
    if not calls:
        return 0.0
    n = max(1, math.ceil(fraction * len(calls)))
    start = min([h for h in [origin, started[0] if started else None, calls[0]] if h is not None])
    span = calls[n - 1] - start
    return round(n / span, 1) if span > 0 else 0.0


def _busy(calls: list[float], started: list[float], done: list[float], fraction: float) -> float:
    """Model calls per hour over the busy period: from the first to the last moment the active
    sandboxes (started, not yet done) are at least `fraction` of their peak. 0 without calls."""
    events = sorted([(h, 1) for h in started] + [(h, -1) for h in done])
    active = peak = 0
    levels = []  # (time, active after the event)
    for h, d in events:
        active += d; peak = max(peak, active); levels.append((h, active))
    if not calls or peak == 0:
        return 0.0
    busy = [h for h, a in levels if a >= fraction * peak]
    lo, hi = busy[0], busy[-1]
    # the last busy moment lasts until the next event drops the level: take that event's time
    after = [h for h, _ in levels if h > hi]
    hi = after[0] if after else (calls[-1] if calls[-1] > hi else hi)
    n = sum(1 for h in calls if lo <= h < hi)
    return round(n / (hi - lo), 1) if hi > lo else 0.0


def add_common_arguments(ap: argparse.ArgumentParser, floor_default: float = 10000) -> None:
    """The reference, the runs and the on-time criteria, shared by compare, progress and scorecard."""
    ap.add_argument("--reference", "--recording", dest="recording", type=Path, required=True,
                    help="the reference: the recordings' directory, or a strict replay of them on the same host (copies are taken by their median)")
    ap.add_argument("--run", "--replay", dest="replay", type=Path, action="append", required=True, help="a replay run's directory (repeatable, one line each)")
    ap.add_argument("--labels", default="", help="comma-separated legend labels, one per --run (default: the directory names)")
    ap.add_argument("--out", type=Path, default=None, help="figure directory (default output/plots/<bench>/<run>/compare/<--name>, placed by the first run)")
    ap.add_argument("--name", default="policies", help="the comparison's name under compare/ when --out is not given (default policies)")
    ap.add_argument("--tolerance", type=float, default=1.10, help="on time when replay execution <= tolerance x max(strict, floor) (default 1.10)")
    ap.add_argument("--floor-ms", type=float, default=floor_default, help=f"the strict time is floored at this many ms before the tolerance applies (default {floor_default:g})")
    ap.add_argument("--min-strict-ms", type=float, default=10000, help="tool calls whose strict execution took this many ms or less are not counted (default 10000: calls over 10 s)")
    ap.add_argument("--reference-stat", choices=["median", "max"], default="median",
                    help="how the copies of a task in the reference stand for it: median (default) or max, the slowest copy of each call "
                         "(the paper's fidelity: --reference <run>/strict-ref/strict-speed0 --reference-stat max --floor-ms 1000)")
    ap.add_argument("--formats", default="png", help="comma-separated output formats, e.g. png,pdf (default png)")
    ap.add_argument("--jobs", type=int, default=8, help="tape-indexing worker processes (default 8), shared out among the runs analysed concurrently")
    ap.add_argument("--parallel", type=int, default=0, metavar="N", help="analyse up to N runs concurrently, each writing its own cache file (default: all of them)")


def resolve_out(args, fallback: str = "compare") -> Path:
    out = args.out or layout.compare_plots_for(args.replay[0], args.name, fallback)
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plot.py compare", description=__doc__.split("\n\n", 1)[0])
    add_common_arguments(ap)
    ap.add_argument("--strict-label", default="strict")
    ap.add_argument("--no-recording-timeline", "--hide-recording", action="store_true",
                    help="keep the reference as the per-call reference but leave it off the clock: no line in tasks-completed, model-calls and "
                         "resources and no point in throughput-on-time (for recordings collected in batches, whose wall clock means nothing)")
    ap.add_argument("--resource-bin-min", type=float, default=1.0, help="bin width in minutes for the resources figure (default 1)")
    ap.add_argument("--points", type=int, default=1, help="windows per run, of equal length, for the throughput against on-time figure (default 1: one point per run, the whole run)")
    ap.add_argument("--cutoff", type=float, default=0.0, metavar="MIN",
                    help="minutes: the throughput against on-time figure judges every run, the strict recording included, on its first MIN minutes "
                         "(tool calls finished per hour over that window, the on-time share of the calls counted in it), so a sample cut at MIN "
                         "minutes sits beside complete runs on the same footing (default 0: the whole run)")
    ap.add_argument("--window-min-calls", type=int, default=1, help="windows with fewer tool calls are left out of that figure (default 1)")
    ap.add_argument("--throughput", choices=["rate", "peak", "bulk", "busy", "occupied"], default="rate",
                    help="the throughput figure's x axis: rate = tool calls finished per hour over the window (default); "
                         "peak = the most model calls served in any --peak-window minutes of the window (calls per minute); "
                         "bulk = model calls per hour over the time from the first sandbox start to the call that completes "
                         "--bulk-fraction of the window's model calls (the stragglers past it do not count); "
                         "busy = model calls per hour over the busy period, from the first to the last moment the active sandboxes "
                         "(started, not finished) are at least --busy-fraction of their peak); "
                         "occupied = model calls per occupied host-hour: the host is charged at each moment the fraction "
                         "min(1, resident sandboxes' CPU limits / cores) of an hour (a thinly loaded tail costs a fraction of a host)")
    ap.add_argument("--occupied-floor", type=float, default=0.0, help="--throughput occupied: the least occupancy charged while any sandbox is resident (default 0)")
    ap.add_argument("--peak-window", type=float, default=1.0, metavar="MIN", help="minutes: the sliding window of --throughput peak (default 1)")
    ap.add_argument("--bulk-fraction", type=float, default=0.9, help="--throughput bulk: the share of the model calls whose service time counts (default 0.9)")
    ap.add_argument("--busy-fraction", type=float, default=0.5, help="--throughput busy: active sandboxes at or above this share of their peak are busy (default 0.5)")
    ap.add_argument("--family", action="append", default=[], metavar="PREFIX",
                    help="the runs whose label starts with PREFIX are one family: one colour with distinct line styles in the figures over time; "
                         "in throughput-on-time one marker, one legend entry (PREFIX), each point labelled with the rest of the run's name, and a "
                         "line through the family's points (repeatable)")
    ap.add_argument("--at-throughput-of", action="append", default=[], metavar="PREFIX=LABEL",
                    help="in throughput-on-time, the runs whose label starts with PREFIX sit at the throughput of the run labelled LABEL "
                         "(repetitions of that run at another speed: only their on-time share is theirs); repeatable")
    args = ap.parse_args(argv)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] or [r.name for r in args.replay]
    if len(labels) != len(args.replay):
        ap.error("--labels must name every --run")
    # A --family group's runs share one colour: count each group once.
    colour_slots = len({next((f for f in args.family if label.startswith(f)), f"run {i}") for i, label in enumerate(labels)})
    if colour_slots > len(RUN_COLORS):
        ap.error(f"at most {len(RUN_COLORS)} colours (a --family group counts once): there are only that many distinct colours")
    out = resolve_out(args)

    runs = []
    exclude = reruns(args.replay)
    warm_cache(args.recording, args.replay, args.tolerance, args.floor_ms, args.min_strict_ms, args.jobs, exclude, args.parallel, args.reference_stat)
    for replay, label in zip(args.replay, labels):
        a = analyse(args.recording, replay, args.tolerance, args.floor_ms, args.min_strict_ms, args.jobs, exclude, args.reference_stat)
        rows = a["rows"]
        with open(out / f"on-time-{re.sub(r'[^A-Za-z0-9_.-]+', '-', label).strip('-')}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hours", "on_time", "task", "thread", "step", "strict_ms", "overcommit_exec_ms", "held_ms", "overcommit_latency_ms", "rolled_back"])
            w.writerows(rows)
        on_time = int(sum(1 for r in rows if r[1]))
        last = f"{a['rep_done'][-1]:.2f} h to the last task" if a["rep_done"] else "none ended yet"
        print(f"{label}: {len(a['tasks'])} tasks ({len(a['running'])} unfinished), {a['complete']} complete compares, {a['timed_out']} timed out, "
              f"{len(a['rep_calls'])} model calls, {len(rows)} tool calls compared, {on_time / len(rows):.1%} on time, "
              + (f"{a['rollbacks']} rollbacks ({sum(1 for r in rows if r[9])} compared calls judged by their redo), " if a["rollbacks"] else "") +
              f"{last}" if rows else f"{label}: no tool calls compared")
        if a["holds_known"] < 1:
            print(f"  note: proxy hold known for {a['holds_known']:.0%} of the steps; the rest are measured hold-inclusive")
        runs.append((label, a))
    group_of = [next((f for f in args.family if label.startswith(f)), None) for label, _ in runs]
    colors = by_family([label for label, _ in runs], RUN_COLORS, FAMILY_COLORS, group_of)
    lstyles = line_styles(group_of)
    # The recording's own timeline is the same whichever run indexed it, as long as the runs replay the same
    # tasks; with `--repeat` runs of different sizes, the largest run's copies of the recordings stand for it.
    strict = max((a for _, a in runs), key=lambda a: len(a["tasks"]))
    clock = not args.no_recording_timeline

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "tasks", "complete_compares", "timed_out", "model_calls", "tool_calls_compared", "on_time", "on_time_rate", "holds_known", "mean_held_s", "hours_to_last_task", "still_running", "peak_model_calls_per_min", "bulk_calls_per_h", "busy_calls_per_h", "occupied_calls_per_h"])
        w.writerow([args.strict_label, len(strict["tasks"]), len(strict["tasks"]), 0, len(strict["rec_calls"]), strict["rec_tool_calls"], strict["rec_tool_calls"], 1.0, 1.0, 0.0, round(strict["rec_done"][-1], 3) if strict["rec_done"] else "", 0, _peak(strict["rec_calls"], args.peak_window, args.cutoff), _bulk(strict["rec_calls"], strict["rec_started"], args.bulk_fraction), _busy(strict["rec_calls"], strict["rec_started"], strict["rec_done"], args.busy_fraction), _occupied(strict["rec_calls"], strict["rec_residency"], strict["rec_cores"], floor=args.occupied_floor)])
        for label, a in runs:
            n = len(a["rows"]); ok = sum(1 for r in a["rows"] if r[1])
            held = np.mean([r[7] for r in a["rows"]]) / 1000 if n else 0.0
            w.writerow([label, len(a["tasks"]), a["complete"], a["timed_out"], len(a["rep_calls"]), n, ok, round(ok / n, 4) if n else "", round(a["holds_known"], 3), round(held, 2),
                        round(a["rep_done"][-1], 3) if a["rep_done"] else "", len(a["running"]), _peak(a["rep_calls"], args.peak_window, args.cutoff), _bulk(a["rep_calls"], a["rep_started"], args.bulk_fraction), _busy(a["rep_calls"], a["rep_started"], a["rep_done"], args.busy_fraction), _occupied(a["rep_calls"], a["rep_residency"], a["rep_cores"], floor=args.occupied_floor)])

    formats = style.parse_formats(args.formats)
    ncol = min(len(runs) + 1, 3)

    def figure():
        fig, ax = plt.subplots(figsize=(style.DOUBLE_COLUMN, 2.6))
        ax.set_xlabel("hours since the run started")
        return fig, ax

    def steps_line(ax, xs, color, label, ys=None, **kw):
        xs = np.asarray(xs)
        ys = np.arange(1, len(xs) + 1) if ys is None else np.asarray(ys)
        ax.step(np.concatenate([[0], xs]), np.concatenate([[0], ys]), where="post", color=color, label=label, **kw)

    fig, ax = figure()
    if clock:
        steps_line(ax, strict["rec_done"], STRICT, args.strict_label)
    for (label, a), color, ls in zip(runs, colors, lstyles):
        steps_line(ax, a["rep_done"], color, label, linestyle=ls)
    ax.set_ylabel("tasks completed")
    style.legend_above(ax, ncol)
    style.save(fig, out / "tasks-completed", formats)

    fig, ax = figure()
    if clock:
        steps_line(ax, strict["rec_calls"], STRICT, args.strict_label)
    for (label, a), color, ls in zip(runs, colors, lstyles):
        steps_line(ax, a["rep_calls"], color, label, linestyle=ls)
    ax.set_ylabel("model calls served")
    style.legend_above(ax, ncol)
    style.save(fig, out / "model-calls", formats)

    fig, ax = figure()
    if len({a["rec_tool_calls"] for _, a in runs}) == 1:
        ax.axhline(strict["rec_tool_calls"], color=STRICT, linestyle="--", linewidth=0.8, label=f"{args.strict_label}: all {counted(args.min_strict_ms)}")
    else:  # runs of different sizes: each run's own recorded total, dashed in its colour
        for (label, a), color, ls in zip(runs, colors, lstyles):
            ax.axhline(a["rec_tool_calls"], color=color, linestyle="--", linewidth=0.8)
        ax.plot([], [], color="0.45", linestyle="--", linewidth=0.8, label=f"dashed: each run's {counted(args.min_strict_ms)} in the {args.strict_label}")
    for (label, a), color, ls in zip(runs, colors, lstyles):
        if a["rows"]:
            hours = [r[0] for r in a["rows"]]
            steps_line(ax, hours, color, label, ys=np.cumsum([1 if r[1] else 0 for r in a["rows"]]), linestyle=ls)
    floor = f", floor {args.floor_ms / 1000:g} s" if args.floor_ms > args.min_strict_ms else ""
    ax.set_ylabel(f"{counted(args.min_strict_ms)} on time\n(at most {slower(args.tolerance)} than {args.strict_label}{floor})")
    style.legend_above(ax, ncol)
    style.save(fig, out / "on-time-tool-calls", formats)

    fig, ax = figure()
    ax.axhline(100, color=STRICT, linestyle="--", linewidth=0.8, label=args.strict_label)
    for (label, a), color, ls in zip(runs, colors, lstyles):
        if a["rows"]:
            hours = np.array([r[0] for r in a["rows"]])
            ok = np.cumsum([1 if r[1] else 0 for r in a["rows"]])
            ax.step(hours, 100 * ok / np.arange(1, len(hours) + 1), where="post", color=color, linestyle=ls, label=label)
    ax.set_ylabel(f"on-time rate, % of {counted(args.min_strict_ms)} so far\n(on time: at most {slower(args.tolerance)} than {args.strict_label})")
    ax.set_ylim(0, 105)
    style.legend_above(ax, ncol)
    style.save(fig, out / "on-time-rate", formats)
    # -- resources over time: tasks completed, cores, memory, swap, summed over running sandboxes --------
    bin_s = args.resource_bin_min * 60

    def aggregate(dirs, t0):
        jobs_ = [(str(d / "tape.telemetry.csv"), t0, bin_s) for d in dirs]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            parts = [r for r in pool.map(telemetry_bins, jobs_) if r is not None]
        if not parts:
            return None
        n = max(len(p[0]) for p in parts)
        cores, mem, swp = np.zeros(n), np.zeros(n), np.zeros(n)
        for idx, c, m, w in parts:
            cores[idx] += c
            mem[idx] += m
            swp[idx] += w
        hours = (np.arange(n) + 0.5) * bin_s / 3600
        return hours, cores, mem, swp

    def active(started, done):
        """The sandboxes active over time: started (admitted) and not yet finished, as the hours of
        each change and the count from then on; a task still running stays active to the end."""
        events = sorted([(h, 1) for h in started] + [(h, -1) for h in done], key=lambda e: (e[0], -e[1]))
        hours = np.concatenate([[0.0], [h for h, _ in events]])
        count = np.concatenate([[0], np.cumsum([d for _, d in events])])
        return hours, count

    series = [(args.strict_label, STRICT, "-", strict["rec_started"], strict["rec_done"], aggregate(strict["rec_dirs"], strict["t0_rec"]))] if clock else []
    for (label, a), color, ls in zip(runs, colors, lstyles):
        replay = args.replay[labels.index(label)]
        series.append((label, color, ls, a["rep_started"], a["rep_done"], aggregate([replay / t for t in a["tasks"]], a["t0_rep"])))
    fig, axes = plt.subplots(5, 1, figsize=(style.DOUBLE_COLUMN, 8.0), sharex=True)
    for label, color, ls, started, done, agg in series:
        xs = np.asarray(done)
        axes[0].step(np.concatenate([[0], xs]), np.concatenate([[0], np.arange(1, len(xs) + 1)]), where="post", color=color, linestyle=ls, label=label)
        hours, count = active(started, done)
        axes[1].step(hours, count, where="post", color=color, linestyle=ls, linewidth=0.8)
        if agg is None:
            continue
        hours, cores, mem, swp = agg
        axes[2].plot(hours, cores, color=color, linestyle=ls, linewidth=0.8)
        axes[3].plot(hours, mem, color=color, linestyle=ls, linewidth=0.8)
        axes[4].plot(hours, swp, color=color, linestyle=ls, linewidth=0.8)
    axes[0].set_ylabel("tasks completed")
    axes[1].set_ylabel("active sandboxes\n(admitted, not finished)")
    axes[2].set_ylabel("CPU (cores)")
    axes[3].set_ylabel("memory (GB)")
    axes[4].set_ylabel("swap (GB)")
    axes[4].set_xlabel("hours since the run started")
    for ax in axes:
        ax.set_ylim(bottom=0)
    style.legend_above(axes[0], min(len(series), 3))
    style.save(fig, out / "resources", formats)

    # -- throughput against on-time rate, per window ----------------------------------------------------
    def peak_per_min(hours, lo, hi):
        """The most events among `hours` (sorted, in hours) in any --peak-window minutes of [lo, hi), per minute."""
        return _peak([h for h in hours if lo <= h < hi], args.peak_window)

    def throughput_of(ser, lo, hi, w):
        """The window [lo, hi)'s throughput of `ser` (its calls, started and done series, in hours) under --throughput."""
        calls = [h for h in ser["calls"] if lo <= h < hi]
        if args.throughput == "peak":
            return _peak(calls, args.peak_window)
        if args.throughput == "bulk":
            return _bulk(calls, [h for h in ser["started"] if lo <= h < hi], args.bulk_fraction, lo)
        if args.throughput == "busy":
            return _busy(calls, [h for h in ser["started"] if lo <= h < hi], [h for h in ser["done"] if lo <= h < hi], args.busy_fraction)
        if args.throughput == "occupied":
            return _occupied(ser["calls"], ser["residency"], ser["cores"], lo, hi, args.occupied_floor)
        return sum(1 for h in ser["tool_done"] if lo <= h < hi) / w

    def windows(ser, call_rows, label):
        """(throughput, on-time share, calls) for each of the run's equal windows, sorted by
        throughput (see --throughput; `ser` holds the run's series: tool_done, calls, started,
        done, in hours); call_rows are the counted calls' (hours, on_time) pairs."""
        done_hours = ser["tool_done"] if args.throughput == "rate" else ser["calls"]
        if args.cutoff:
            cut = args.cutoff / 60
            done_hours = [h for h in done_hours if h < cut]
            call_rows = [r for r in call_rows if r[0] < cut]
        if not done_hours and not call_rows:
            return []
        end = args.cutoff / 60 if args.cutoff else max([h for h in done_hours] + [r[0] for r in call_rows])
        w = end / args.points
        out_ = []
        for i in range(args.points):
            lo, hi = i * w, (i + 1) * w if i < args.points - 1 else end + 1e-9
            calls = [r[1] for r in call_rows if lo <= r[0] < hi]
            if len(calls) < args.window_min_calls:
                continue
            out_.append((throughput_of(ser, lo, hi, w), 100 * sum(1 for c in calls if c) / len(calls), len(calls)))
        return sorted(out_)

    # Drawn in the slide style: bold sans-serif labels, a framed axis, one marker and line style
    # per run, a smooth curve through each run's points, the strict recording as a single grey dot.
    styles = by_family([label for label, _ in runs], RUN_STYLES, FAMILY_STYLES, group_of)
    served = lambda a: {"tool_done": a["rep_tool_done"], "calls": a["rep_calls"], "started": a["rep_started"], "done": a["rep_done"],  # noqa: E731
                        "residency": a["rep_residency"], "cores": a["rep_cores"]}
    strict_ser = {"tool_done": strict["rec_tool_done"], "calls": strict["rec_calls"], "started": strict["rec_started"], "done": strict["rec_done"],
                  "residency": strict["rec_residency"], "cores": strict["rec_cores"]}
    pins = {}  # --at-throughput-of: a label prefix -> the throughput (one window: the whole run) of the run it names
    for spec in args.at_throughput_of:
        prefix, _, target = spec.partition("=")
        a = next((a for label, a in runs if label == target), None)
        if a is None:
            raise SystemExit(f"--at-throughput-of {spec}: no run is labelled {target!r}")
        pts = windows(served(a), [(r[0], r[1]) for r in a["rows"]], target)
        pins[prefix] = max(x for x, _, _ in pts) if pts else 0.0
    if args.cutoff:
        strict_tp = throughput_of(strict_ser, 0.0, args.cutoff / 60, args.cutoff / 60)
    elif strict_ser["tool_done"]:
        strict_tp = throughput_of(strict_ser, 0.0, float("inf"), strict_ser["tool_done"][-1])
    else:
        strict_tp = 0.0
    with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 1.1, "axes.spines.top": True, "axes.spines.right": True,
                         "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out"}):
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        if clock:
            ax.plot([strict_tp], [100], marker="o", markersize=8, color="#7F7F7F", linestyle="none", label=args.strict_label)
        lows, highs = [100.0], [strict_tp if clock else 0.0]
        groups: dict[str, tuple] = {}  # a --family group's colour, line style and points: one line through them
        points = [(args.strict_label, "", strict_tp, 100.0, 0)] if clock else []  # the figure's points, written to throughput-on-time.csv
        for i, (label, a) in enumerate(runs):
            pinned = next((pins[p] for p in pins if label.startswith(p)), None)
            if pinned is not None:  # its own clock does not count: the other run's throughput, its own on-time share
                calls = [r[1] for r in a["rows"]]
                pts = [(pinned, 100 * sum(1 for c in calls if c) / len(calls), len(calls))] if calls else []
            else:
                pts = windows(served(a), [(r[0], r[1]) for r in a["rows"]], label)
            if not pts:
                continue
            xs = np.array([x for x, _, _ in pts]); ys = np.array([y for _, y, _ in pts])
            points += [(label, group_of[i] or "", x, y, n) for x, y, n in pts]
            color, marker, ls = styles[i % len(styles)]
            lows.append(ys.min())
            highs.append(xs.max())
            if group_of[i] is not None:
                # One legend entry for the group; each point says which of its runs it is.
                g = group_of[i]
                ax.plot(xs, ys, marker=marker, markersize=6, color=color, linestyle="none", label=None if g in groups else g)
                pts_so_far = groups.setdefault(g, (color, ls, []))[2]
                for x, y in zip(xs, ys):
                    if pinned is not None:  # one x for the whole group: the legend names it, the points need no label
                        pts_so_far.append((x, y))
                        continue
                    # Labels alternate above and below the points, so that close runs stay readable.
                    dy = 4 if len(pts_so_far) % 2 == 0 else -9
                    ax.annotate(label[len(g):].strip("-_ ") or label, (x, y), xytext=(5, dy), textcoords="offset points", fontsize=7, color=color)
                    pts_so_far.append((x, y))
                continue
            if len(xs) >= 3 and xs.max() > xs.min():
                coef = np.polyfit(xs, ys, 2)
                grid = np.linspace(xs.min(), xs.max(), 100)
                ax.plot(grid, np.polyval(coef, grid), color=color, linestyle=ls, linewidth=1.8)
            else:
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8)
            ax.plot(xs, ys, marker=marker, markersize=6, color=color, linestyle="none", label=label)
        for color, ls, pts in groups.values():
            pts.sort()
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, linestyle=ls, linewidth=1.8)
        lo = max(0, (min(lows) // 10) * 10 - 10)
        ax.set_ylim(lo, 102)
        ax.set_xlim(0, max(highs) * 1.08 or None)  # room for the rightmost marker and its label
        ax.set_xlabel({"peak": f"Peak throughput (model calls/min/host, {args.peak_window:g}-min window)",
                       "bulk": "Throughput (model calls/hr/host)",
                       "busy": "Throughput (model calls/hr/host)",
                       "occupied": "Throughput (model calls/occupied host-hour)",
                       "rate": "Throughput (tool calls/hr/host)"}[args.throughput])
        ax.set_ylabel(f"On-time tool calls (%)\n(at most {slower(args.tolerance)})")
        ax.text(0.98, 0.03, "better →↑", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#7F7F7F")
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    style.save(fig, out / "throughput-on-time", formats)
    # The plotted points, for figures composed from several compares (one row per point: the run's
    # label, its --family group, throughput, on-time share, judged calls in that window).
    with open(out / "throughput-on-time.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "group", "throughput", "on_time_pct", "calls"])
        w.writerows(points)
    print(f"figures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
