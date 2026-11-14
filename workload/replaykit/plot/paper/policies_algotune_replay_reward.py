"""policies_algotune_replay_reward.py: the AlgoTune replay-reward experiment in one bar per policy. The recorded trace of
every task is replayed under a policy and the solution re-scored by the task's verifier in the sandbox (runs
overcommit-0.8-verify, tidesand-verify and strict-verify of algotune/run4-max-50, two copies of each task). For every
task the strict rerun scored above 0, the bar is the mean over tasks of the policy's reward (mean of its two copies) against
the strict rerun's, in %; the error bar is the standard deviation over tasks. The third bar is the strict rerun's second
copy against its first: the verifier's own noise floor."""
import json, glob, os, statistics
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BR = "output/replay/algotune/run4-max-50"
STRICT = f"{BR}/strict-verify"
RUNS = [("overcommit", f"{BR}/overcommit-0.8-verify", "#D55E00"), ("TideSand", f"{BR}/tidesand-verify", "#009E73"), ("strict", STRICT, "#0072B2")]
OUT = "output/plots/paper/policies-algotune-replay-reward.pdf"


def task_of(u):
    u = u.split("__")[0]
    return u[len("alg-rep-"):] if u.startswith("alg-rep-") else u


per_run = {}
for name, d, _ in RUNS:
    per = per_run.setdefault(name, {})
    for m in glob.glob(f"{d}/*/meta.json"):
        j = json.load(open(m)); rw = j.get("reward")
        if rw is None: rw = (j.get("verifier") or {}).get("reward")
        if rw is not None: per.setdefault(task_of(os.path.basename(os.path.dirname(m))), []).append(float(rw))
strict = per_run["strict"]
ref = {t: statistics.mean(v) for t, v in strict.items() if statistics.mean(v) > 0}
stats = []
for name, _, color in RUNS:
    if name == "strict": changes = [100 * (strict[t][1] / strict[t][0] - 1) for t in sorted(ref) if len(strict[t]) == 2 and strict[t][0] > 0]
    else: changes = [100 * (statistics.mean(per_run[name][t]) / ref[t] - 1) for t in sorted(ref) if t in per_run[name]]
    stats.append((name, color, statistics.mean(changes), statistics.stdev(changes) if len(changes) > 1 else 0, len(changes)))
    print(f"{name:12s} n={len(changes):2d} tasks  mean {stats[-1][2]:+7.1f} %  std {stats[-1][3]:6.1f} %  median {statistics.median(changes):+6.1f} %")

with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 10, "axes.labelsize": 12, "axes.labelweight": "bold", "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "in"}):
    fig, ax = plt.subplots(figsize=(4.9, 2.6))  # the 4.9-in width of the other single-column figures, shorter
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle=":"); ax.set_axisbelow(True)
    for i, (name, color, mean, sd, n) in enumerate(stats):
        ax.bar(i, mean, width=0.55, color=color, linewidth=0, yerr=sd, capsize=4, error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
        # the value sits on the bar itself (its mean), on a white patch so the error bar does not cross the digits
        ax.text(i, mean + (5 if mean >= 0 else -5), f"{mean:+.0f} %", ha="center", va="bottom" if mean >= 0 else "top", fontsize=8.5, zorder=4,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_xticks(range(len(stats))); ax.set_xticklabels([{"strict": "Strict", "overcommit": "Overcommit"}.get(n, n) for n, *_ in stats])
    ax.set_ylabel("Task Reward Error (%)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print(OUT)
