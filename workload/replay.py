#!/usr/bin/env python3
"""Replay recorded tapes in Tidelet sandboxes, for any benchmark:
`replay.py --data data/<bench>/<run> [tasks | --all] [--policy P] [--repeat N] [--verify] ...`.
The driver is `replaykit.replay` (its docstring has the details)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # workload/: replaykit

from replaykit.replay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
