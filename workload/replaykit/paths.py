"""Paths in the TideBox checkout and small host helpers."""

from __future__ import annotations

import os
from pathlib import Path

WORKLOAD = Path(__file__).resolve().parents[1]  # workload/: the PYTHONPATH that holds `replaykit`
TIDEBOX = WORKLOAD.parent  # the TideBox checkout
DATA_ROOT = TIDEBOX / "data"  # the recordings (downloaded)
IMAGES_ROOT = TIDEBOX / "images"  # the task images and task copies (downloaded)
OUTPUT_ROOT = TIDEBOX / "output"  # replays and figures (downloaded results, or your own runs)
PYTHON = TIDEBOX / ".venv" / "bin" / "python"
TIDEBOX_CLI = TIDEBOX / ".venv" / "bin" / "tidebox"


def user_home() -> Path:
    """The invoking user's home, also under sudo (the replay driver re-executes itself as root
    for tideletd's socket; Claude Code's versions are the user's)."""
    return Path(os.path.expanduser("~" + os.environ.get("SUDO_USER", "")))
