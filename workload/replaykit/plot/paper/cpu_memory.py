"""cpu_memory.py RUN_DIR OUT.pdf UNIT...: the task-usage panels of several units side by side, one CPU axis on the left
and one memory axis on the right (a common scale across the panels), one shared legend above."""
import sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from usage import RC, load, draw, legend_handles

run, out, units = sys.argv[1], sys.argv[2], sys.argv[3:]
data = [load(os.path.join(run, u)) for u in units]
cpu_top = max(max(d[1]) for d in data) * 1.15; mem_top = max(max(d[2] + d[3]) for d in data) * 1.15
with plt.rc_context({**RC, "axes.labelsize": 15.5, "axes.labelweight": "bold", "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13}):
    fig, axes = plt.subplots(1, len(units), figsize=(4.9 * len(units), 3.5), sharey=True)
    any_swap = False
    for k, (ax, unit, (x, cpu, mem, swap, calls)) in enumerate(zip(axes, units, data)):
        b = ax.twinx(); any_swap |= draw(ax, b, x, cpu, mem, swap, calls)
        ax.set_ylim(0, cpu_top); b.set_ylim(0, mem_top)
        if k == 0: ax.set_ylabel("CPU (cores)")
        if k == len(units) - 1: b.set_ylabel("Memory (GiB)")
        else: b.tick_params(axis="y", labelright=False, right=False)
        ax.text(0.02, 0.97, unit.replace("algotune-", ""), transform=ax.transAxes, ha="left", va="top", fontsize=12.5)
    fig.legend(handles=legend_handles(any_swap), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.subplots_adjust(wspace=0.06); fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print(out)
