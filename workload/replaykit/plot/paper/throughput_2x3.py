"""throughput_2x3.py [KEY]: the paper's 2x3 throughput-fidelity figure (default KEY bulk80-whole), drawn from the points each
per-workload compare (scripts/analyze.sh) wrote to output/plots/<bench>/<rec>/compare/<KEY>/throughput-on-time.csv:
rows Claude Sonnet 5.5 / DeepSeek V4.1 Flash; columns AlgoTune, Terminal-Bench 4.0, Terminal-Bench Science.
KEY = <window> (rate) or <mode>-<window>; window = whole | first30; mode = peak | bulk[NN] | busy[NN] | occupied.
One y label per row (the model), one x label per column; no title text above the panels."""
import sys, os, csv, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
key = sys.argv[1] if len(sys.argv) > 1 else "bulk80-whole"
mode, _, window = key.rpartition("-") if "-" in key else ("rate", "", key)
base = re.match(r"[a-z]+", mode).group()
MS = 10.8  # marker size of the policy points (1.2x the first paper draft)
LABEL_FS = 8.8  # the point labels (0.6, 0.8, 1.0, none): 1.1x the first draft
BETTER_FS = 10.4  # the "better" note: 1.3x the first draft, in grey
PAPER = True  # the paper version: no strict (speed 1), no ablation runs, tidesand shown as TideSand
RENAME = {"tidesand": "TideSand", "strict-speed0": "strict", "agentcgroup-g0.8": "AgentCgroup-0.8", "agentcgroup": "AgentCgroup-none"}
GROUP = {"agentcgroup": "AgentCgroup"}  # the family's name (the run "agentcgroup" itself is the no-admission point)  # AgentCgroup: the paper's design (no admission) and with overcommit's rule at 0.8, one family  # the compare set: v6 (the final v6 figure) or v7 (v6 and v7 side by side)
d = "output/plots/%s/compare/" + key + "/throughput-on-time.csv"
cols = [("AlgoTune", "algotune"), ("Terminal-Bench 4.0", "terminal-bench-4.0"), ("Terminal-Bench Science", "terminal-bench-science")]
rows = [("Claude Sonnet 5.5", {"algotune": "run4-max-50", "terminal-bench-4.0": "run2-medium", "terminal-bench-science": "run1-rand50"}),
        ("DeepSeek V4.1 Flash", {b: "run1-deepseek-v4.1-flash" for _, b in cols})]
# The panel style of compare.py: a family keeps its colour/marker across figures.
# Okabe-Ito palette: vermillion (overcommit), blue (strict), bluish green (TideSand), reddish purple (AgentCgroup),
# orange, sky blue, yellow for anything else.
STYLES = [("#D55E00", "^", "-"), ("#0072B2", "s", "--"), ("#009E73", "D", "-."), ("#CC79A7", "o", ":"), ("#E69F00", "v", "-"), ("#56B4E9", "P", "--"), ("#F0E442", "X", "-.")]
FAMILY = {"overcommit": STYLES[0], "tidesand": STYLES[2], "TideSand": STYLES[2], "AgentCgroup": STYLES[3], "agentcgroup": STYLES[3]}
XLABEL = {"rate": "Throughput (tool calls/hr)", "peak": "Peak throughput (tool calls/min)", "bulk": "Throughput (tool calls/hr)",
          "busy": "Throughput (tool calls/hr)", "occupied": "Throughput (tool calls/occupied hour)"}[base]

def style_of(labels):
    """A style per (label, group): a family's own style for its first run, the group's style for the rest of a group,
    the remaining styles in order for everything else (compare.py's by_family)."""
    out, taken, shared = [], set(), {}
    spare = [st for st in STYLES if st not in FAMILY.values()]
    for label, group in labels:
        if group and group in shared: out.append(shared[group]); continue
        fam = next((f for f in FAMILY if label.startswith(f) and f not in taken), None)
        if fam: taken.add(fam); st = FAMILY[fam]
        else: st = spare.pop(0)
        out.append(st)
        if group: shared[group] = st
    return out

def panel(ax, path):
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes); return
    pts = list(csv.DictReader(open(path)))
    if PAPER:
        pts = [p for p in pts if p["run"] not in ("strict (speed 1)", "agentcgroup-g0.8") and not p["run"].startswith(("tidesand-speculate", "tidesand-defer", "tidesand-rollback"))]  # AgentCgroup: only its own design (no admission gate), one unlabelled point
        for p in pts:
            p["run"] = RENAME.get(p["run"], p["run"]); p["group"] = GROUP.get(p["group"], RENAME.get(p["group"], p["group"]))
    ax._labels = []  # (x, y, text, color): placed by place_labels once the layout is final
    strict = [p for p in pts if p["group"] == "" and p["calls"] == "0"]
    runs = [p for p in pts if not (p["group"] == "" and p["calls"] == "0")]
    seen = []  # (label, group) in first-appearance order
    for p in runs:
        if (p["run"], p["group"]) not in seen: seen.append((p["run"], p["group"]))
    styles = dict(zip(seen, style_of(seen)))
    lows, highs = [100.0], [0.0]
    for p in strict:
        x = float(p["throughput"]); highs.append(x)
        ax.plot([x], [100], marker="s", markersize=MS + 1, color="#0072B2", linestyle="none", label=p["run"], zorder=3)
    groups = {}  # group -> (color, ls, points); one line through a group's points, labelled unless all share one x (pinned)
    for label, group in seen:
        mine = [(float(p["throughput"]), float(p["on_time_pct"])) for p in runs if p["run"] == label and p["group"] == group]
        color, marker, ls = styles[(label, group)]
        xs = [x for x, _ in mine]; ys = [y for _, y in mine]
        lows.append(min(ys)); highs.append(max(xs))
        if group:
            gx = {round(float(p["throughput"])) for p in runs if p["group"] == group}
            ax.plot(xs, ys, marker=marker, markersize=MS, color=color, linestyle="none", label=None if group in groups else group, zorder=3)
            g = groups.setdefault(group, (color, ls, []))
            for x, y in mine:
                if len(gx) > 1:
                    ax._labels.append((x, y, label[len(group):].strip("-_ ") or label, color))
                g[2].append((x, y))
            continue
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8, zorder=2)
        ax.plot(xs, ys, marker=marker, markersize=MS, color=color, linestyle="none", label=label, zorder=3)
    for color, ls, p_ in groups.values():
        p_.sort(); ax.plot([x for x, _ in p_], [y for _, y in p_], color=color, linestyle=ls, linewidth=1.8, zorder=2)  # lines under the points
    ax.set_ylim(max(0, (min(lows) // 10) * 10 - 10), 102); ax.set_xlim(0, max(highs) * 1.08 or None)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))  # few x ticks, so the labels stay apart
    ax.text(0.98, 0.03, "better →↑", transform=ax.transAxes, ha="right", va="bottom", fontsize=BETTER_FS, color="#808080")

def place_labels(fig, ax, fontsize=LABEL_FS):
    """Put each point label beside its point where it covers no marker, line, label or the panel's edge.
    Candidates are tried right, then left, then above and below; the one with the least overlap wins."""
    if not getattr(ax, "_labels", None):
        return
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    radius = (MS / 2 + 1.5) * px
    # obstacles: every marker and every line of the axes, as display-space circles
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
    placed = [t.get_window_extent(renderer) for t in ax.texts]  # the "better" note
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
    for x, y, text, color in ax._labels:
        best = None
        for (dx, dy), ha, va in candidates:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color, ha=ha, va=va)
            bb = t.get_window_extent(renderer)
            score = overlap(bb)
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, bb)
            else:
                t.remove()
            if score == 0:
                break
        placed.append(best[2])

with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 13.5, "axes.titleweight": "bold",
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.1, "xtick.direction": "out", "ytick.direction": "out"}):
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 6.6))
    for r, (model, recs) in enumerate(rows):
        for c, (title, bench) in enumerate(cols):
            ax = axes[r][c]; panel(ax, d % f"{bench}/{recs[bench]}")
            if r == 0: ax.set_title(title, fontsize=14)
            if c == 0: ax.set_ylabel("Tool Call Fidelity (%)")
            if c == len(cols) - 1:  # the row's model, on the right of the row
                ax.text(1.04, 0.5, model, transform=ax.transAxes, rotation=270, ha="left", va="center", fontsize=13.5, fontweight="bold", clip_on=False)
            if r == 1: ax.set_xlabel(XLABEL)
    # One legend for the figure, above the panels, in a fixed order; the families come from whichever panels have them.
    handles = {}
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
    LEGEND = {"strict": "Strict", "overcommit": "Overcommit", "AgentCgroup": "AgentCgroup", "TideSand": "TideSand (Ours)"}  # display names, in this order, ours last
    order = [l for l in LEGEND if l in handles] + [l for l in handles if l not in LEGEND]
    fig.legend([handles[l] for l in order], [LEGEND.get(l, l) for l in order], loc="upper center", ncol=len(order), frameon=False, fontsize=13.5, markerscale=1.5, handletextpad=0.5, columnspacing=2.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 0.975, 0.95))
    fig.canvas.draw()
    for row in axes:
        for ax in row:
            place_labels(fig, ax)
    out = "output/plots/paper/throughput-fidelity-2x3.pdf" if PAPER else f"output/plots/throughput-on-time-2x3-{key}.png"; fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02); print(out)
