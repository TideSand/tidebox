"""workload_burstiness.py [OUT]: the CPU burstiness of each workload's strict run, per sandbox, in 1-s bins.
A burst is a run of consecutive seconds at >= ACTIVE cores in which at least one second reaches BURST cores.
(a) peak / mean CPU per sandbox (violins); (b) length of each burst (boxes). Distributions span the 5th-95th percentiles,
cut at the axis top (1.5x the largest upper quartile). Default OUT output/plots/paper/workload-burstiness.pdf."""
import sys, os, csv, glob, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "output/plots/paper/workload-burstiness.pdf"
ACTIVE = float(os.environ.get("ACTIVE", "0.5"))
BURST = float(os.environ.get("BURST", "3.5"))  # 4 cores, allowing for measurement jitter (3.99)
BENCHES = [("AlgoTune", "algotune"), ("Terminal-\nBench", "terminal-bench-4.0"), ("TB\nScience", "terminal-bench-science")]
MODELS = [("Sonnet 5.5", "#E69F00", {"algotune": "run4-max-50", "terminal-bench-4.0": "run2-medium", "terminal-bench-science": "run1-rand50"}),
          ("DeepSeek V4.1 Flash", "#56B4E9", {b: "run1-deepseek-v4.1-flash" for _, b in BENCHES})]
WIDTH = 0.36


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def unit_cpu(path):
    """Cores used in each second."""
    t, c = [], []
    for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
        if r.get("time"): t.append(ts(r["time"])); c.append(float(r.get("cpu_ms") or 0) / 1000)
    if len(t) < 8: return None
    edges = np.arange(t[0], t[-1], 1.0)
    return np.maximum(np.diff(np.interp(edges, t, c)), 0)


def workload(run_dir):
    """(peak/mean per sandbox, length of each burst)."""
    ratio, length = [], []
    for p in glob.glob(f"{run_dir}/*/tape.telemetry.csv"):
        c = unit_cpu(p)
        if c is None or c.mean() <= 0: continue
        ratio.append(c.max() / c.mean())
        d = np.diff(np.r_[0, (c >= ACTIVE).astype(int), 0]); starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        length += [b - a for a, b in zip(starts, ends) if c[a:b].max() >= BURST]
    return np.array(ratio), np.array(length)


def violin(ax, x, v, color, top):
    q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    body = ax.violinplot([v[(v >= q5) & (v <= min(q95, top))]], positions=[x], widths=WIDTH, showextrema=False)["bodies"][0]
    body.set_facecolor(color); body.set_edgecolor("black"); body.set_linewidth(0.9); body.set_alpha(1)
    ax.vlines(x, q25, q75, color="black", linewidth=3.5, zorder=3)
    ax.hlines(q50, x - WIDTH * 0.3, x + WIDTH * 0.3, color="black", linewidth=2, zorder=4)
    return q50, min(q95, top)


def box(ax, x, v, color, top):
    q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    ax.bxp([{"med": q50, "q1": q25, "q3": q75, "whislo": q5, "whishi": min(q95, top), "fliers": []}], positions=[x], widths=WIDTH,
           patch_artist=True, showfliers=False, boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 0.9},
           medianprops={"color": "black", "linewidth": 1.6}, whiskerprops={"linewidth": 0.9}, capprops={"linewidth": 0.9 if q95 <= top else 0})
    return q50, q75


data = {}
for _, color, recs in MODELS:
    for _, bench in BENCHES:
        data[(color, bench)] = workload(f"output/replay/{bench}/{recs[bench]}/strict")
        q = lambda v: "/".join(f"{x:.1f}" for x in np.percentile(v, [25, 50, 75]))
        r, l = data[(color, bench)]
        print(f"{bench:24s} {recs[bench]:26s} peak/mean {q(r):>16s}  bursts {len(l):4d} length s {q(l):>16s}")

PANELS = [("Peak / mean CPU", "{:.0f}×", "(a) Spikiness", violin), ("Burst length (s)", "{:.0f} s", "(b) Burst length", box)]
with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.linewidth": 1.1, "legend.fontsize": 12}):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for k, (ax, (ylabel, fmt, sub, draw)) in enumerate(zip(axes, PANELS)):
        top = 1.5 * max(np.percentile(v[k], 75) for v in data.values())
        for m, (_, color, _) in enumerate(MODELS):
            for i, (_, bench) in enumerate(BENCHES):
                x = i + (m - 0.5) * WIDTH * 1.15
                med, at = draw(ax, x, data[(color, bench)][k], color, top)
                ax.annotate(fmt.format(med), (x, at), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                            bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"}, zorder=5)
        ax.set_xticks(range(len(BENCHES))); ax.set_xticklabels([b for b, _ in BENCHES]); ax.set_xlim(-0.6, len(BENCHES) - 0.4)
        ax.set_ylim(0, top * 1.12); ax.set_ylabel(ylabel)
        ax.set_xlabel(sub, fontsize=13, labelpad=8)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.9) for _, c, _ in MODELS]
    fig.legend(handles, [l for l, _, _ in MODELS], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), columnspacing=2.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print(OUT)
