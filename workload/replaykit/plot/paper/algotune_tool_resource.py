"""algotune_tool_resource.py [CSV]: the slowdown of tool calls under smaller sandbox allocations (fig 5 left of the motivation).
Ten AlgoTune tool calls, each replayed ten times at 2 CPUs / 4 GB, 4 CPUs / 8 GB and 8 CPUs / 16 GB (the recording's
allocation), from the variability experiment's measurements.csv (one row per repetition: wall_s, recorded_s, level).
One density line per allocation of wall_s / recorded_s (a Gaussian kernel, Scott's bandwidth), filled lightly."""
import sys, csv, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "output/replay/algotune/run2-max/variability/measurements.csv"
LEVELS = [("2c4g", "2 CPUs, 4 GB", "#D55E00"), ("4c8g", "4 CPUs, 8 GB", "#009E73"), ("8c16g", "8 CPUs, 16 GB", "#0072B2")]
rows = list(csv.DictReader(open(CSV)))


def kde(xs, grid):
    xs = np.asarray(xs); h = 1.06 * xs.std(ddof=1) * len(xs) ** (-1 / 5)   # Scott / Silverman bandwidth
    return np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / h) ** 2).sum(axis=1) / (len(xs) * h * math.sqrt(2 * math.pi))


with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 10, "axes.labelsize": 15.5, "axes.labelweight": "bold", "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out", "legend.fontsize": 15}):
    fig, ax = plt.subplots(figsize=(4.9, 3.5))
    ax.grid(color="#DDDDDD", linewidth=0.6, linestyle=":"); ax.set_axisbelow(True)
    for level, label, color in LEVELS:
        s = np.array([float(r["wall_s"]) / float(r["recorded_s"]) for r in rows if r["level"] == level])
        grid = np.linspace(s.min() - 0.1, s.max() + 0.1, 600); d = kde(s, grid)
        ax.plot(grid, d, color=color, linewidth=2, label=label); ax.fill_between(grid, 0, d, color=color, alpha=0.15, linewidth=0)
        print(f"{label}: n={len(s)} median {np.median(s):.2f}x p90 {np.percentile(s, 90):.2f}x max {s.max():.2f}x")
    ax.set_xscale("log"); ax.set_xticks([1, 1.5, 2, 3, 4, 6]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Slowdown over the recorded time"); ax.set_ylabel("Probability density"); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))  # few y ticks
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); out = "output/plots/paper/algotune-tool-resource.pdf"; fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print(out)
