"""algotune_overcommit_run.py [N]: the final AlgoTune reward per task under strict reservation and under overcommit (fig 5
right of the motivation): one dot per policy and task on a log axis with a zero row, an arrow from strict to overcommit
labelled with the ratio, a cross where the agent earned no reward. The data are two recordings of the same ten tasks:
run2-max (strict, budget admission) and run3-max-loadgate (overcommit, load admission), each task's harbor_result.reward.
The N (default 10) tasks whose reward moved most (|log ratio|, a lost reward first) are shown, ordered by the strict reward."""
import sys, os, glob, json, math, statistics
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RUNS = {"strict": "data/algotune/run2-max", "overcommit": "data/algotune/run3-max-loadgate"}
OUT = "output/plots/paper/algotune-overcommit-run.pdf"
GOLD, BLUE = "#0072B2", "#D55E00"  # strict and overcommit in the paper 2x3 colours (Okabe-Ito blue, vermillion)
ZERO = 0.3  # where "no reward" sits on the log axis
DOT = 7  # marker size of the reward dots (pt); the crosses and the arrows scale with it


def task_of(unit):
    u = unit.split("__")[0]
    return u[len("algotune-"):] if u.startswith("algotune-") else u


rewards = {}
for run, d in RUNS.items():
    for m in glob.glob(f"{d}/*/meta.json"):
        j = json.load(open(m)); r = (j.get("harbor_result") or {}).get("reward")
        if r is None: continue
        rewards.setdefault(task_of(os.path.basename(os.path.dirname(m))), {}).setdefault(run, []).append(float(r))
both = {t: {k: statistics.mean(v) for k, v in rr.items()} for t, rr in rewards.items() if "strict" in rr and "overcommit" in rr}


def change(r):
    s, o = r["strict"], r["overcommit"]
    return math.inf if (s > 0) != (o > 0) else (abs(math.log(o / s)) if s > 0 else 0.0)


chosen = sorted(both, key=lambda t: -change(both[t]))[:N]
chosen.sort(key=lambda t: -both[t]["strict"])
ys = lambda v: v if v > 0 else ZERO

with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 10, "axes.labelsize": 15.5, "axes.labelweight": "bold", "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "in", "legend.fontsize": 15}):
    fig, ax = plt.subplots(figsize=(4.9, 3.5))
    ax.set_yscale("log"); ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle=":"); ax.set_axisbelow(True)
    ax.axhline(1, color="#999999", linewidth=0.7, linestyle="--")
    for i, t in enumerate(chosen):
        s, o = both[t]["strict"], both[t]["overcommit"]
        for v, c in ((s, GOLD), (o, BLUE)):
            if v > 0: ax.plot([i], [v], marker="o", markersize=DOT, color=c, linestyle="none", zorder=3)
            else: ax.plot([i], [ZERO], marker="x", markersize=DOT * 1.15, markeredgewidth=1.8, color=c, linestyle="none", zorder=3)
        if s != o and abs(math.log10(ys(o) / ys(s))) > 0.3:  # no arrow between dots that (nearly) touch: the ratio label says it
            ax.annotate("", (i, ys(o)), (i, ys(s)), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color="#808080", lw=1.4, shrinkA=DOT / 2, shrinkB=DOT / 2,  # end at the dot edge: no gap
                         joinstyle="miter", capstyle="butt"), zorder=2)
        if s > 0 and o > 0:
            left = t == "cumulative-simpson-multid"  # its ratio label goes left of the arrow, clear of the next label / arrow
            high = t == "sylvester-solver"  # between its dots it would sit on kd-tree's arrow: right of the upper dot, above eigcplx's dot
            ax.text(i - 0.26 if left else i + 0.26, s * 1.6 if high else math.sqrt(s * o), f"{o / s:.2g}x", fontsize=11, color="#555555", ha="right" if left else "left", va="center")
    SHORT = {"discrete-log": "dlog", "lti-simulation": "lti-sim", "wasserstein-dist": "wstein", "cumulative-simpson-multid": "cumsimp",
             "l1-pruning": "l1prune", "kd-tree": "kd-tree", "sylvester-solver": "sylv", "eigenvectors-complex": "eigcplx",
             "delaunay": "delaun", "btsp": "btsp"}  # tick labels of at most 7 characters
    short = lambda t: SHORT.get(t, t[:7])
    ax.set_xticks(range(len(chosen))); ax.set_xticklabels([short(t) for t in chosen], rotation=40, ha="right", fontsize=10)
    ax.set_xlim(-0.6, len(chosen) - 0.4)
    top = max(max(both[t].values()) for t in chosen)
    ax.set_ylim(ZERO / 1.6, top * 3)
    ticks = [ZERO] + [10 ** k for k in range(0, int(math.log10(top)) + 1)]
    ax.set_yticks(ticks); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == ZERO else f"$10^{{{int(round(math.log10(v)))}}}$"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_ylabel("Reward")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", markersize=DOT, color=GOLD, linestyle="none", label="Strict"),
               plt.Line2D([], [], marker="o", markersize=DOT, color=BLUE, linestyle="none", label="Overcommit")]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=1, handletextpad=0.3, markerscale=1.2)  # stacked
    fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print(OUT)
    print("tasks:", ", ".join(f"{t} {both[t]['strict']:.3g}->{both[t]['overcommit']:.3g}" for t in chosen))
