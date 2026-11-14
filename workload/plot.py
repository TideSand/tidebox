#!/usr/bin/env python3
"""The replay analysis, for any benchmark: `plot.py <figure> ...` (`plot.py --help` lists them).
The code is `replaykit.plot` (its `cli` module's docstring has the details)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # workload/: replaykit

from replaykit.plot.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
