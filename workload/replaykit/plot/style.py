"""Figure style for the paper: serif text at print sizes, column-width figures, colorblind-safe
colors (Okabe-Ito), no chart junk, vector output.

Widths: `SINGLE_COLUMN` (3.4 in) and `DOUBLE_COLUMN` (7.0 in) match a two-column article.
Figures carry no title by default; the caption says what they show.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE_COLUMN = 3.4
DOUBLE_COLUMN = 7.0

# Okabe-Ito
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
GREY = "#6E6E6E"
BLACK = "#000000"
PURPLE = "#CC79A7"  # Okabe-Ito reddish purple
SKY = "#56B4E9"  # Okabe-Ito sky blue

COLORS = {"cpu": BLUE, "memory": GREEN, "ratio": VERMILLION, "count": GREY, "limit": BLACK,
          # replays drawn over their recording: fixed hues in both panels, recording behind
          "recording": ORANGE, "replay": BLUE, "replay-llm": GREEN}

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.8,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed TrueType so the text stays editable and passes publisher checks
    "ps.fonttype": 42,
}


def apply() -> None:
    plt.rcParams.update(RC)


def legend_above(ax, ncol: int = 3) -> None:
    """A legend in the margin above the axes, so it never covers data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.0, handletextpad=0.5, columnspacing=1.5)


def save(fig, stem: Path, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Writes `<stem>.<fmt>` for each format and returns the paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        path = stem.with_name(f"{stem.name}.{fmt}")  # not with_suffix: a stem like run-v3.2 keeps its dot
        fig.savefig(path, format=fmt)
        out.append(path)
    return out


def parse_formats(text: str) -> tuple[str, ...]:
    return tuple(f.strip().lstrip(".") for f in text.split(",") if f.strip())
