"""`plot.py <figure> [options]`: the analysis of replay runs, for any benchmark. The benchmark and
the run are read off the paths given, and so is where the output goes:

    output/plots/<bench>/<run>/compare/<name>/       figures and tables comparing replay runs (`--out`)

    compare      policies against the strict reference over time: tasks completed, model calls, on-time tool
                 calls, on-time rate, pool resources, throughput against on-time rate (+ summary.csv,
                 throughput-on-time.csv, one on-time-<run>.csv per run)

`plot.py <figure> --help` has each figure's options.
"""

from __future__ import annotations

import importlib
import sys

FIGURES = {
    "compare": ("replaykit.plot.compare", "policies against the strict reference over time (figures, summary.csv)"),
}


def usage() -> str:
    width = max(len(k) for k in FIGURES)
    lines = ["usage: plot.py <figure> [options]", "", "figures:"]
    lines += [f"  {k:<{width}}  {desc}" for k, (_, desc) in FIGURES.items()]
    lines += ["", "plot.py <figure> --help has each figure's options; the module docstring (replaykit/plot/cli.py) says where figures go."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in FIGURES:
        print(f"unknown figure {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(FIGURES[name][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
