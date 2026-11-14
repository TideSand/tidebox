"""tidesand_ablation.py [CSV]: the parameter sensitivity of TideSand on AlgoTune Sonnet (run4-max-50, x2 at 15 s), 1x3 (the
mechanism ablation is tidesand_mechanisms.py): the CPU budget fraction alpha (0.55 / 0.7 / 0.85), the bandwidth budget
fraction beta (off / 0.6 / 0.8 / 1.0) and the rollback pressure threshold (off / 0.7 / 0.85 / 1.0), one option moved at a
time from the default configuration (alpha 0.7, beta 0.8, pressure 0.85). Each panel plots tool call fidelity against raw
throughput (tool calls/hr, bulk-90), the points labelled with the option's value and the default configuration filled.
Data: the compare `sensitivity` (scripts/analyze.sh), output/plots/algotune/run4-max-50/compare/sensitivity/throughput-on-time.csv."""
import sys, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "output/plots/algotune/run4-max-50/compare/sensitivity/throughput-on-time.csv"  # bulk-90
XLABEL = "Throughput\n(tool calls/hr)"  # two lines, so each panel carries its own

FINAL = "tidesand"
SWEEPS = [("CPU Threshold", [("0.55", "tidesand-t0.55"), ("0.7", FINAL), ("0.85", "tidesand-t0.85")]),
          ("Bandwidth Threshold", [("off", "tidesand-bw-off"), ("0.6", "tidesand-bw0.6"), ("0.8", FINAL), ("1.0", "tidesand-bw1.0")]),
          ("Rollback Threshold", [("off", "tidesand-rb-off"), ("0.7", "tidesand-rb0.7"), ("0.85", FINAL), ("1.0", "tidesand-rb1.0")])]
COLOR = "#009E73"  # TideSand: the paper 2x3 colour (Okabe-Ito bluish green)
OFFSET = {("CPU", "0.85"): (6, 4), ("Bandwidth", "off"): (6, -4), ("Bandwidth", "0.6"): (-26, -4), ("Bandwidth", "1.0"): (6, -11),
          ("Rollback", "off"): (-24, -11), ("Rollback", "1.0"): (6, -11)}  # label placements that avoid the neighbours; default (6, 4)

with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 10, "axes.labelsize": 11, "axes.labelweight": "bold", "axes.titlesize": 11, "axes.titleweight": "bold",
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "in"}):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0))  # a single-column figure at rollback_micro.py's width and font sizes; the mechanisms are tidesand_mechanisms.py
    rows = list(csv.DictReader(open(SWEEP)))
    raw = {r["run"]: float(r["throughput"]) for r in rows}; S = raw["strict (speed 1)"]
    fidelity = {r["run"]: float(r["on_time_pct"]) for r in rows}
    for ax, (title, items) in zip(axes, SWEEPS):
        xs = [raw[r] for _, r in items]; ys = [fidelity[r] for _, r in items]
        num = [(x, y) for (label, _), x, y in zip(items, xs, ys) if label != "off"]  # the line joins the numeric settings; "off" stands alone
        ax.plot([x for x, _ in num], [y for _, y in num], color=COLOR, linewidth=1.0, zorder=1)
        for (label, run), x, y in zip(items, xs, ys):
            final = run == FINAL
            ax.plot([x], [y], marker="o", markersize=7 if final else 5.5, color=COLOR, markerfacecolor=COLOR if final else "white", markeredgewidth=1.2, linestyle="none", zorder=3)
            ax.annotate(label, (x, y), xytext=OFFSET.get((title.split()[0], label), (6, 4)), textcoords="offset points", fontsize=8.5, color="#333333")
            print(f"{title:24s} {label:5s} {run:22s} {x:8.1f} calls/h  {y:5.1f} %")
        ax.set_title(title); ax.grid(color="#DDDDDD", linewidth=0.6, linestyle=":"); ax.set_axisbelow(True)
        ax.set_xlabel(XLABEL); ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Tool Call Fidelity (%)")
    # each panel on its own scale, so the flat sweeps are readable: a margin of 30 % of the panel's own span, at least 0.15x / 0.5 pt
    for ax, (_, items) in zip(axes, SWEEPS):
        xs = [raw[r] for _, r in items]; ys = [fidelity[r] for _, r in items]
        mx = max(0.3 * (max(xs) - min(xs)), 0.15 * S); my = max(0.3 * (max(ys) - min(ys)), 0.5)
        if ax is axes[1]:  # the bandwidth sweep: zoomed onto its points (the floors above would leave them in a corner)
            mx = 0.8 * (max(xs) - min(xs)); my = 0.3 * (max(ys) - min(ys))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))  # one decimal
        ax.set_xlim(min(xs) - mx, max(xs) + mx * 1.6); ax.set_ylim(min(ys) - my, max(ys) + my)
    fig.tight_layout(); out = "output/plots/paper/tidesand-ablation.pdf"; fig.savefig(out, bbox_inches="tight", pad_inches=0.02); print(out)
