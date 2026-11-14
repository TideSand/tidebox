"""rollback_micro.py [OUT.pdf]: the selective-rollback micro figure (rollback-micro.pdf): per workload, (b) the share of
model calls rolled back and (c) the CPU overhead of the mechanism as a share of all tool execution in the run, stacked:
checkpoint = the time sandboxes were paused in CRIU dumps, rollback = the execution discarded by rollbacks (release to stop
of each rolled-back call, redone after the restore). One colour per model (Okabe-Ito orange for Sonnet 5.5, sky blue for
DeepSeek V4.1 Flash); a checkpoint segment is the greyed version of its model's colour.
Sources, per workload's `tidesand` run: tideletd.log (checkpoint tookMs, "tidesand release", rollback), metrics.txt
(tidelet_model_calls_total), and the compare `bulk80-whole` (scripts/analyze.sh): on-time-tidesand.csv, whose
overcommit_exec_ms summed is the run's tool execution. The numbers also go to output/plots/rollback-micro.json."""
import sys, json, csv, re, colorsys
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

out = sys.argv[1] if len(sys.argv) > 1 else "output/plots/paper/rollback-micro.pdf"
WS = [("AlgoTune", "Sonnet", "algotune/run4-max-50"), ("AlgoTune", "DeepSeek", "algotune/run1-deepseek-v4.1-flash"),
      ("Terminal-\nBench", "Sonnet", "terminal-bench-4.0/run2-medium"), ("Terminal-\nBench", "DeepSeek", "terminal-bench-4.0/run1-deepseek-v4.1-flash"),
      ("TB\nScience", "Sonnet", "terminal-bench-science/run1-rand50"), ("TB\nScience", "DeepSeek", "terminal-bench-science/run1-deepseek-v4.1-flash")]


def ts(line):
    return datetime.fromisoformat(re.match(r"time=(\S+)", line)[1].replace("Z", "+00:00")).timestamp()


rows = []
for bench, model, w in WS:
    d = f"output/replay/{w}/tidesand"
    calls = int(float(re.search(r"tidelet_model_calls_total (\S+)", open(f"{d}/metrics.txt").read())[1]))
    dumps, rel, lost, rolls = [], {}, [], 0
    with open(f"{d}/tideletd.log", errors="replace") as fh:
        for line in fh:
            if "msg=checkpoint " in line:
                m = re.search(r"tookMs=(\d+)", line)
                if m: dumps.append(int(m[1]) / 1000)
            elif "tidesand release" in line:
                m = re.search(r"call=(\S+)", line); rel[m[1]] = ts(line)
            elif "msg=rollback " in line:
                m = re.search(r"call=(\S+)", line); rolls += 1
                if m and m[1] in rel: lost.append(ts(line) - rel[m[1]])
    exec_s = sum(float(r["overcommit_exec_ms"]) for r in csv.DictReader(open(f"output/plots/{w}/compare/bulk80-whole/on-time-tidesand.csv"))) / 1000
    rows.append({"bench": bench, "model": model, "calls": calls, "insured": len(dumps), "rolled_back": rolls, "rb_pct": 100 * rolls / calls,
                 "ckpt_pct": 100 * sum(dumps) / exec_s, "rb_over_pct": 100 * sum(lost) / exec_s, "exec_h": exec_s / 3600})
    r = rows[-1]
    print(f"{bench.replace(chr(10), ' '):16s} {model:9s} calls {calls:6d} checkpointed {len(dumps):4d} rolled back {rolls:3d} ({r['rb_pct']:.2f} %)  "
          f"overhead checkpoint {r['ckpt_pct']:.2f} % + rollback {r['rb_over_pct']:.2f} % = {r['ckpt_pct'] + r['rb_over_pct']:.2f} %")
json.dump(rows, open("output/plots/rollback-micro.json", "w"), indent=1)
MODELS = [("Sonnet", "Sonnet 5.5", "#E69F00"), ("DeepSeek", "DeepSeek V4.1 Flash", "#56B4E9")]

def grey(c, sat=0.22, light=0.72):
    """The greyed version of a colour: its hue, little saturation, a light grey tone."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(c))
    return colorsys.hls_to_rgb(h, light, s * sat)

benches = []
for r in rows:
    if r["bench"] not in benches: benches.append(r["bench"])
val = {(r["bench"], r["model"]): r for r in rows}
W = 0.36
with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "axes.labelweight": "bold", "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.0}):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for k, (m, _, c) in enumerate(MODELS):
        for i, bench in enumerate(benches):
            r = val[(bench, m)]; x = i + (k - 0.5) * W
            a.bar(x, r["rb_pct"], W, color=c, edgecolor="white", linewidth=0.6)
            a.text(x, r["rb_pct"] + 0.012, f"{r['rb_pct']:.2f}%", ha="center", va="bottom", fontsize=6.8)
            ck, rb = r["ckpt_pct"], r["rb_over_pct"]
            b.bar(x, ck, W, color=grey(c), edgecolor="white", linewidth=0.6)
            b.bar(x, rb, W, bottom=ck, color=c, edgecolor="white", linewidth=0.6)
            b.text(x, ck + rb + 0.05, f"{ck + rb:.1f}%", ha="center", va="bottom", fontsize=6.8)
    for ax, lab, title in ((a, "Calls rolled back (%)", "(b) Rollback frequency"), (b, "CPU overhead (%)", "(c) Rollback overhead")):
        ax.set_xticks(range(len(benches))); ax.set_xticklabels(benches)
        ax.set_ylabel(lab); ax.set_xlabel(title, fontsize=11, labelpad=6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    a.set_ylim(0, max(r["rb_pct"] for r in rows) * 1.25)
    b.set_ylim(0, max(r["ckpt_pct"] + r["rb_over_pct"] for r in rows) * 1.3)
    fig.legend(handles=[Patch(facecolor=c, label=lab) for _, lab, c in MODELS], frameon=False, fontsize=9.5, loc="upper center",
               ncol=len(MODELS), bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)  # the model legend, on top of both panels
    # (c): one entry per segment kind, its swatch split into the two models' colours (the models are named in (b)).
    from matplotlib.legend_handler import HandlerTuple
    b.legend([tuple(Patch(facecolor=c) for _, _, c in MODELS), tuple(Patch(facecolor=grey(c)) for _, _, c in MODELS)],
             ["Rollback", "Checkpoint"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
             frameon=False, fontsize=8.5, loc="upper left", handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print(out)
