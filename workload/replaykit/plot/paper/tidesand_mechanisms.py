"""tidesand_mechanisms.py [CSV]: the mechanism ablation of TideSand on AlgoTune Sonnet (run4-max-50, x2 at 15 s), one panel
in the style of the paper 2x3 (throughput_2x3.py): TideSand (filled diamond) against each mechanism alone (hollow diamonds:
defer = the release and admission rule, speculate = swapping, rollback), overcommit 0.6 / 0.8 / 1.0 and strict, tool call
fidelity against throughput (tool calls/hr, bulk-80). Data: the compare `mechanisms` (scripts/analyze.sh): the 2x3's runs
plus the single-mechanism runs and AgentCgroup behind overcommit's gate,
output/plots/algotune/run4-max-50/compare/mechanisms/throughput-on-time.csv."""
import sys, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV = sys.argv[1] if len(sys.argv) > 1 else "output/plots/algotune/run4-max-50/compare/mechanisms/throughput-on-time.csv"
OUT = "output/plots/paper/tidesand-mechanisms.pdf"
MS, LABEL_FS, BETTER_FS, LEGEND_FS = 8.0, 8.8, 10.4, 11.0  # smaller points and legend than the 2x3 (10.8, 13.5); its label and "better" sizes
OVER, AGC, QS = "#D55E00", "#CC79A7", "#009E73"  # the 2x3's Okabe-Ito colours (no strict point here)
AGENTCGROUP = [("agentcgroup-g0.8", "0.8"), ("agentcgroup", "none")]
MECH = [("tidesand-defer", "defer only"), ("tidesand-speculate", "speculate only"), ("tidesand-rollback", "rollback only")]
pts = {r["run"]: (float(r["throughput"]), float(r["on_time_pct"])) for r in csv.DictReader(open(CSV))}


def place_labels(fig, ax, labels, fontsize=LABEL_FS):
    """throughput_2x3.py's place_labels: each label beside its point where it covers no marker, line, label or the edge."""
    renderer = fig.canvas.get_renderer(); px = fig.dpi / 72.0; radius = (MS / 2 + 1.5) * px
    obstacles = []
    for line in ax.get_lines():
        xy = ax.transData.transform(list(zip(line.get_xdata(), line.get_ydata())))
        if line.get_marker() not in ("", "None", None):
            obstacles += [(x, y, radius) for x, y in xy]
        if line.get_linestyle() not in ("", "None", "none", None) and len(xy) > 1:
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                n = max(2, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (3 * px)))
                obstacles += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 1.5 * px) for i in range(n + 1)]
    axbox = ax.get_window_extent(renderer)
    placed = [t.get_window_extent(renderer) for t in ax.texts]
    def overlap(bb):
        area = 0.0
        for x, y, r in obstacles:
            dx = max(bb.x0 - x, 0, x - bb.x1); dy = max(bb.y0 - y, 0, y - bb.y1)
            if dx * dx + dy * dy < r * r:
                area += (r - (dx * dx + dy * dy) ** 0.5) * r
        for o in placed:
            w = min(bb.x1, o.x1) - max(bb.x0, o.x0); h = min(bb.y1, o.y1) - max(bb.y0, o.y0)
            if w > 0 and h > 0:
                area += 4 * w * h
        out = max(0, axbox.x0 - bb.x0) + max(0, bb.x1 - axbox.x1) + max(0, axbox.y0 - bb.y0) + max(0, bb.y1 - axbox.y1)
        return area + 50 * out * px
    candidates = [((7, 3), "left", "bottom"), ((7, -3), "left", "top"), ((-7, 3), "right", "bottom"), ((-7, -3), "right", "top"),
                  ((0, 8), "center", "bottom"), ((0, -8), "center", "top"), ((10, 0), "left", "center"), ((-10, 0), "right", "center")]
    for x, y, text, color in labels:
        best = None
        for (dx, dy), ha, va in candidates:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color, ha=ha, va=va)
            bb = t.get_window_extent(renderer); score = overlap(bb)
            if best is None or score < best[0]:
                if best is not None: best[1].remove()
                best = (score, t, bb)
            else:
                t.remove()
            if score == 0: break
        placed.append(best[2])


with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 13.5, "axes.titleweight": "bold",
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.1, "xtick.direction": "out", "ytick.direction": "out"}):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = []
    oc = [(t, *pts[f"overcommit-{t}"]) for t in ("0.6", "0.8", "1.0")]
    ax.plot([x for _, x, _ in oc], [y for _, _, y in oc], color=OVER, linestyle="-", linewidth=1.8, zorder=2)
    ax.plot([x for _, x, _ in oc], [y for _, _, y in oc], marker="^", markersize=MS, color=OVER, linestyle="none", label="Overcommit", zorder=3)
    labels += [(x, y, t, OVER) for t, x, y in oc]
    ag = sorted((*pts[run], t) for run, t in AGENTCGROUP)
    ax.plot([x for x, _, _ in ag], [y for _, y, _ in ag], color=AGC, linestyle=":", linewidth=1.8, zorder=2)
    ax.plot([x for x, _, _ in ag], [y for _, y, _ in ag], marker="o", markersize=MS, color=AGC, linestyle="none", label="AgentCgroup", zorder=3)
    labels += [(x, y, t, AGC) for x, y, t in ag]
    for run, text in MECH:
        x, y = pts[run]
        ax.plot([x], [y], marker="D", markersize=MS, color=QS, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", zorder=3)
        labels.append((x, y, text, QS))
    x, y = pts["tidesand"]
    ax.plot([x], [y], marker="D", markersize=MS, color=QS, linestyle="none", label="TideSand (Ours)", zorder=3)
    ax.annotate("all", (x, y), xytext=(6, 3), textcoords="offset points", fontsize=LABEL_FS, color=QS, ha="left", va="bottom")  # the full policy: every mechanism
    xs = [x for _, x, _ in oc] + [x for x, _, _ in ag] + [pts[r][0] for r, _ in MECH] + [x]
    ys = [y for _, _, y in oc] + [y for _, y, _ in ag] + [pts[r][1] for r, _ in MECH]
    lo = (min(xs) * 0.9 // 500) * 500  # without strict, the axis starts below the leftmost point, not at 0
    ax.set_ylim(max(0, (min(ys) // 10) * 10 - 10), 102); ax.set_xlim(lo, max(xs) + (max(xs) - lo) * 0.12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.text(0.98, 0.03, "better →↑", transform=ax.transAxes, ha="right", va="bottom", fontsize=BETTER_FS, color="#808080")
    ax.set_xlabel("Throughput (tool calls/hr)"); ax.set_ylabel("Tool Call Fidelity (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=LEGEND_FS, markerscale=1.0,
               handletextpad=0.5, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.canvas.draw()
    place_labels(fig, ax, labels)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.02); print(OUT)
    for run, text in [("tidesand", "TideSand")] + MECH:
        print(f"{text:15s} {pts[run][0]:7.1f} calls/h  {pts[run][1]:5.1f} %")
