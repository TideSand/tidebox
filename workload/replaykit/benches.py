"""The benchmarks the replays come from, looked up by name or inferred from a path.

    from replaykit import benches
    bench = benches.get("algotune")
    bench = benches.infer(Path("data/algotune/run4-max-50"))      # from a path under data/ or output/replay/

A bench names its sandboxes (`<short>-rep-<task>`) and says where its task copies are: the task
directories (task.toml, tests/) that come with the images (images/<dir>/tasks-arm64/<task>),
whose `tests/test.sh` the replay's `--verify` runs in the sandbox.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from replaykit.layout import locate
from replaykit.paths import DATA_ROOT, IMAGES_ROOT


@dataclass(frozen=True)
class Bench:
    name: str  # data/<name>/<run>, output/replay/<name>/<run>/
    short: str  # sandbox names: <short>-rep-<task>
    images: str  # the image set's directory under images/
    default_run: str = "run1"

    @property
    def tasks_root(self) -> Path:
        return IMAGES_ROOT / self.images / "tasks-arm64"

    @property
    def default_data(self) -> Path:
        return DATA_ROOT / self.name / self.default_run


BENCHES = {b.name: b for b in (
    Bench("algotune", "alg", "algotune", default_run="run4-max-50"),
    Bench("terminal-bench-4.0", "tb4", "tb4", default_run="run2-medium"),
    Bench("terminal-bench-science", "tbsci", "tbsci", default_run="run1-rand50"),
)}


def names() -> list[str]:
    return list(BENCHES)


def get(name: str) -> Bench:
    if name not in BENCHES:
        raise KeyError(f"unknown benchmark {name!r}; one of {', '.join(BENCHES)}")
    return BENCHES[name]


def infer_name(path: Path | str | None = None, meta: dict | None = None) -> str | None:
    """The benchmark a path belongs to (data/<bench>/... or output/replay/<bench>/...), or the
    one a recording's meta.json names; None when neither says."""
    if path is not None:
        loc = locate(path)
        if loc and loc.bench in BENCHES:
            return loc.bench
        p = Path(path)
        for candidate in (p / "meta.json", *sorted(p.glob("*/meta.json"))[:1]):
            try:
                meta = meta or json.loads(candidate.read_text())
                break
            except (OSError, ValueError):
                continue
    if meta and meta.get("bench") in BENCHES:
        return meta["bench"]
    return None


def infer(path: Path | str | None = None, meta: dict | None = None) -> Bench | None:
    name = infer_name(path, meta)
    return get(name) if name else None
