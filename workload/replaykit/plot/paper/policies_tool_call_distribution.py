"""policies_tool_call_distribution.py [KEY]: the distribution of tool-call slowdown under three policies, one panel per
workload (AlgoTune, Terminal-Bench, Terminal-Bench Science; Claude Sonnet 5.5), from the same per-call data as the paper's
2x3 (the compare `<KEY>`, default bulk80-whole: on-time-<run>.csv, one row per tool call). Slowdown = the call's
execution time under the policy over its reference time (the slowest of five strict speed-0 copies), for the calls whose
reference is at least MIN_S seconds (default 1, the on-time definition's floor, so no floor applies). A violin per policy
(strict = the strict speed-1 run, overcommit = overcommit-0.8, TideSand = tidesand) over the calls up to YMAX (default
4x), the median as a black bar; the y axis is the normalised duration in % (100 = as recorded); the share of calls above YMAX is printed."""
import sys, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

key = sys.argv[1] if len(sys.argv) > 1 else "bulk80-whole"
YMAX = float(os.environ.get("YMAX", "4")); MIN_S = float(os.environ.get("MIN_S", "1"))
COLS = [("AlgoTune", "algotune/run4-max-50"), ("Terminal-Bench", "terminal-bench-4.0/run2-medium"), ("Terminal-Bench Science", "terminal-bench-science/run1-rand50")]
RUNS = [("Strict", "strict-speed-1", "#0072B2"), ("Overcommit", "overcommit-0.8", "#D55E00"), ("TideSand", "tidesand", "#009E73")]


def slowdowns(path):
    out = []
    for r in csv.DictReader(open(path)):
        try: ref = float(r["strict_ms"]); ex = float(r["overcommit_exec_ms"])
        except (KeyError, ValueError): continue
        if ref >= 1000 * MIN_S: out.append(ex / ref)
    return np.array(out)


with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 10, "axes.labelsize": 11, "axes.labelweight": "bold", "axes.titlesize": 12,
                     "axes.titleweight": "bold", "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "in"}):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)  # rollback_micro.py's width and font sizes
    for ax, (title, bench) in zip(axes, COLS):
        d = f"output/plots/{bench}/compare/{key}"
        for i, (label, run, color) in enumerate(RUNS):
            s = slowdowns(os.path.join(d, f"on-time-{run}.csv"))
            if not len(s): continue
            shown = 100 * s[s <= YMAX]
            v = ax.violinplot([shown], positions=[i], widths=0.8, showextrema=False, points=400, bw_method=0.08)
            for b in v["bodies"]: b.set_facecolor(color); b.set_edgecolor(color); b.set_linewidth(0.6); b.set_alpha(0.35)
            med, p95 = np.median(s), np.percentile(s, 95)
            ax.hlines(100 * med, i - 0.3, i + 0.3, color="black", linewidth=1.6)
            print(f"{title:24s} {label:10s} n={len(s):6d} median {np.median(s):5.2f} p95 {np.percentile(s, 95):6.2f} above {YMAX:g}x: {100 * (s > YMAX).mean():4.1f} %")
        ax.axhline(100, color="#AAAAAA", linewidth=0.8, linestyle=":")
        ax.set_title(title); ax.set_xticks(range(len(RUNS))); ax.set_xticklabels([l for l, _, _ in RUNS], fontsize=7); ax.set_xlim(-0.5, len(RUNS) - 0.5)  # upright policy names, smaller than the other ticks so they fit at 7 in
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Tool Call Delay (%)"); axes[0].set_ylim(50, 100 * YMAX + 5); axes[0].set_yticks([100, 200, 300, 400])
    fig.tight_layout(rect=(0, 0, 0.985, 1)); out = "output/plots/paper/policies-tool-call-distribution.pdf"; fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print(out)
