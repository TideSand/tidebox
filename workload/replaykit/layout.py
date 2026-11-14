"""Where everything goes, the same for every benchmark.

    data/<bench>/<run>/<task>/                          a recording: tape.json, tape.telemetry.csv, meta.json, ...
    output/replay/<bench>/<run>/<name>/<unit>/          a replay of that run: <name> is the tideletd policy
                                                        (or what --name says), <unit> is <task>, or <task>__r<k>
                                                        for copy k of a --repeat run
    output/plots/<bench>/<run>/compare/<name>/          figures and tables comparing replay runs

Older replay runs laid their copies out as `rep<k>/<task>/`; `units()` reads both layouts, so the
plot code never looks at directory names itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from replaykit.paths import DATA_ROOT, OUTPUT_ROOT

REPLAY_ROOT = OUTPUT_ROOT / "replay"
PLOTS_ROOT = OUTPUT_ROOT / "plots"
_COPY = re.compile(r"__r(\d+)$")
_REP = re.compile(r"^rep(\d+)$")


def recording_dir(bench: str, run: str) -> Path:
    return DATA_ROOT / bench / run


def replay_dir(bench: str, run: str, name: str) -> Path:
    return REPLAY_ROOT / bench / run / name


@dataclass(frozen=True)
class Located:
    """What a path under data/ or output/replay/ is."""

    kind: str  # "recording" or "replay"
    bench: str
    run: str
    name: str | None = None  # the replay run's name (policy)
    task: str | None = None  # when the path goes down to a task directory

    @property
    def plots(self) -> Path:
        """Where this run's own figures go."""
        root = PLOTS_ROOT / self.bench / self.run
        return root / "recording" if self.kind == "recording" else root / "replay" / (self.name or "replay")

    def compare_plots(self, name: str = "policies") -> Path:
        return PLOTS_ROOT / self.bench / self.run / "compare" / name


def _is_unit(p: Path) -> bool:
    """A task's directory: it holds the replay's (or recording's) files, not other directories of them."""
    return any((p / f).exists() for f in ("meta.json", "status.json", "tape.json", "tape.json.steps.jsonl", "tape.telemetry.csv"))


def locate(path: Path | str) -> Located | None:
    """Parses a path under data/<bench>/<run>[/<task>] or output/replay/<bench>/<run>/<name>[/<unit>];
    None for a path elsewhere. A replay run's name may have several levels (`strict-ref/strict-speed1`),
    so when the path exists the disk says where the run ends and a unit begins (`_is_unit`); a path
    that does not exist yet is read as <name>[/<unit>]."""
    p = Path(path).resolve()
    for root, kind in ((DATA_ROOT.resolve(), "recording"), (REPLAY_ROOT.resolve(), "replay")):
        try:
            parts = p.relative_to(root).parts
        except ValueError:
            continue
        if kind == "recording" and len(parts) >= 2:
            return Located(kind, parts[0], parts[1], None, parts[2] if len(parts) > 2 else None)
        if kind == "replay" and len(parts) >= 3:
            name, task = parts[2:], None
            if len(parts) > 3:
                if p.exists() and _is_unit(p):
                    task = parts[-1]
                    name = parts[2:-2] if _REP.match(parts[-2]) and len(parts) > 4 else parts[2:-1]
                elif p.exists():
                    name = parts[2:]  # a run directory nested under another name
                else:
                    name, task = parts[2:3], parts[3]
            return Located(kind, parts[0], parts[1], "/".join(name), task)
    return None


def plots_for(path: Path | str, fallback: str = "misc") -> Path:
    """The figure directory for a recording or replay run (see `Located.plots`); for a path
    outside data/ and output/replay/, output/plots/<fallback>/<its name>."""
    loc = locate(path)
    return loc.plots if loc else PLOTS_ROOT / fallback / Path(path).resolve().name


def compare_plots_for(path: Path | str, name: str = "policies", fallback: str = "misc") -> Path:
    """Where figures comparing runs go, placed by one of the runs (a replay run, or the recording)."""
    loc = locate(path)
    return loc.compare_plots(name) if loc else PLOTS_ROOT / fallback / name


@dataclass(frozen=True)
class Unit:
    """One replayed task: its directory, the recording's task name and the copy number (None
    for a single pass). `rel` is the directory relative to the run, which analyses use as the
    unit's id ("algotune-fft", "algotune-fft__r1", "rep2/algotune-fft")."""

    dir: Path
    task: str
    copy: int | None

    @property
    def rel(self) -> str:
        parent = self.dir.parent.name
        return f"{parent}/{self.dir.name}" if _REP.match(parent) else self.dir.name


def unit_task(unit_dir: Path) -> str:
    """The recording a unit directory plays back: its record's `task` when it has one, else the
    directory name without the copy suffix."""
    for name in ("meta.json", "status.json"):
        try:
            task = json.loads((unit_dir / name).read_text()).get("task")
        except (OSError, ValueError):
            continue
        if task:
            return task
    return _COPY.sub("", unit_dir.name)


def unit_copy(unit_dir: Path) -> int | None:
    if m := _COPY.search(unit_dir.name):
        return int(m.group(1))
    if m := _REP.match(unit_dir.parent.name):
        return int(m.group(1))
    return None


def units(run: Path, require: str | None = None) -> list[Unit]:
    """The task directories of a replay run (or a recording run), in both layouts, sorted by
    their relative path. With `require`, only those that have that file (e.g. "tape.json")."""
    out: list[Unit] = []
    run = Path(run)
    if not run.is_dir():
        return out
    for d in sorted(run.iterdir()):
        if not d.is_dir():
            continue
        if _REP.match(d.name):
            dirs = sorted(x for x in d.iterdir() if x.is_dir())
        else:
            dirs = [d]
        for u in dirs:
            if require and not (u / require).exists() and not (u / (require + ".steps.jsonl")).exists():
                continue
            if not any((u / f).exists() for f in ("meta.json", "status.json", "tape.json", "tape.json.steps.jsonl", "tape.telemetry.csv")):
                continue
            out.append(Unit(u, unit_task(u), unit_copy(u)))
    return out


def recording_of(unit_dir: Path) -> Path | None:
    """For a replay unit, the recording's task directory (meta.json's `replay_of` names its tape),
    or None for a recording."""
    try:
        tape = json.loads((unit_dir / "meta.json").read_text()).get("replay_of")
    except (OSError, ValueError):
        return None
    return Path(tape).parent if tape else None


def reference_units(reference: Path, task: str) -> list[Path]:
    """The directories of `reference` that stand for `task`: the recording's own directory when
    `reference` is a recording run, else every copy of the task in that replay run."""
    direct = reference / task
    if (direct / "tape.json").exists() or (direct / "tape.json.steps.jsonl").exists():
        if locate(reference) is None or locate(reference).kind == "recording":
            return [direct]
    found = [u.dir for u in units(reference) if u.task == task]
    return found or ([direct] if direct.is_dir() else [])
