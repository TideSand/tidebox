"""policies_resources.py [RUN_BASE] [OUT]: the host's CPU, DRAM and swap use over wall-clock time under three policies on
one workload: strict, overcommit-0.8 and TideSand (tidesand). Default RUN_BASE output/replay/terminal-bench-science/
run1-rand50 (Terminal-Bench Science Sonnet, x2 at 15 s), OUT output/plots/paper/policies-tbs-resources.pdf; the AlgoTune
Sonnet version is RUN_BASE output/replay/algotune/run4-max-50 with OUT policies-algotune-resources.pdf. Every unit's sandbox
telemetry (250-ms samples, absolute time) is binned to BIN seconds: CPU as the growth of the cgroup's cumulative CPU time
per bin, memory and swap as the samples' mean; the bins are summed over the run's units and divided by the host's sandbox
capacity (56 cores, 112 GiB; swap in GiB). Three panels, one line per policy, from each run's start.
UNITS=abs draws CPU in cores and DRAM in GB (10^9 bytes) instead of % of capacity; LINES=solid draws every policy solid
(all.sh sets both for the AlgoTune version)."""
import sys, os, csv, glob, bisect, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "output/replay/terminal-bench-science/run1-rand50"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/plots/paper/policies-tbs-resources.pdf"
RUNS = [("Strict", "strict", "#0072B2", "-", "o"), ("Overcommit", "overcommit-0.8", "#D55E00", "-", "s"), ("TideSand", "tidesand", "#009E73", "-", "^")]
CORES, GIB, BIN = 56.0, 112.0, float(os.environ.get("BIN", "15"))
SMOOTH = int(os.environ.get("SMOOTH", "4"))  # bins of running mean on the drawn line (0 = raw bins)
CPU_SMOOTH = int(os.environ.get("CPU_SMOOTH", "20"))  # bins of running mean on the CPU line (20 x 15 s = 5 min; all.sh keeps 4 for AlgoTune)
UNITS = os.environ.get("UNITS", "pct")  # pct = % of the capacity; abs = cores and GB
LINES = os.environ.get("LINES", "")  # solid = every policy's line solid
GB = 2**30 / 1e9  # GiB -> GB
XMAX = float(os.environ.get("XMAX", "0"))  # minutes shown (0 = up to the end of the longest loaded run); strict at x1 can run much longer


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def unit_series(path):
    """(times, cumulative cpu seconds, memory GiB, swap GiB) of one unit's telemetry."""
    t, c, m, w = [], [], [], []
    for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
        if not r.get("time"): continue
        t.append(ts(r["time"])); c.append(float(r.get("cpu_ms") or 0) / 1000); m.append(float(r.get("memory_bytes") or 0) / 2**30); w.append(float(r.get("swap_bytes") or 0) / 2**30)
    return np.array(t), np.array(c), np.array(m), np.array(w)


def pool(run_dir):
    """(minutes, cores, memory GiB, swap GiB) per bin, summed over the run's units; t0 = the earliest sample."""
    units = [unit_series(p) for p in glob.glob(f"{run_dir}/*/tape.telemetry.csv")]
    units = [u for u in units if len(u[0]) > 1]
    t0 = min(u[0][0] for u in units); end = max(u[0][-1] for u in units)
    n = int((end - t0) // BIN) + 1; cpu = np.zeros(n); mem = np.zeros(n); swp = np.zeros(n)
    edges = t0 + np.arange(n + 1) * BIN
    for t, c, m, w in units:
        # CPU: the counter's value at each bin edge (interpolated), differenced
        lo, hi = bisect.bisect_left(edges, t[0]), bisect.bisect_right(edges, t[-1])
        if hi - lo < 2: continue
        at = np.interp(edges[lo:hi], t, c); cpu[lo:hi - 1] += np.diff(at) / BIN
        # memory: the mean of the samples in each bin
        idx = ((t - t0) // BIN).astype(int); ok = (idx >= 0) & (idx < n)
        cnt = np.maximum(np.bincount(idx[ok], minlength=n), 1)
        mem += np.bincount(idx[ok], weights=m[ok], minlength=n) / cnt; swp += np.bincount(idx[ok], weights=w[ok], minlength=n) / cnt
    return np.arange(n) * BIN / 60, cpu, mem, swp


def smooth(y, w):
    if w <= 1: return y
    k = np.ones(w); return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")  # the mean of the bins present, so the ends do not dip


with plt.rc_context({"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 10, "axes.labelsize": 12, "axes.labelweight": "bold", "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "in", "legend.fontsize": 10}):
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(4.9, 5.0), sharex=True)
    xmax = 0
    for label, name, color, ls, marker in RUNS:
        x, cpu, mem, swp = pool(os.path.join(BASE, name))
        cpu = np.minimum(cpu, CORES)  # cap each bin at the host (56 cores = 100 %): a bin above it is a telemetry catch-up, not load
        if LINES == "solid": ls = "-"
        cy, my = (cpu, mem * GB) if UNITS == "abs" else (100 * cpu / CORES, 100 * mem / GIB)
        a.plot(x, smooth(cy, CPU_SMOOTH), color=color, linestyle=ls, linewidth=1.3, label=label)
        b.plot(x, smooth(my, SMOOTH), color=color, linestyle=ls, linewidth=1.3, label=label)
        c.plot(x, smooth(swp, SMOOTH), color=color, linestyle=ls, linewidth=1.3, label=label)
        if name != "strict": xmax = max(xmax, x[-1])
        print(f"{label:10s} {name:16s} {x[-1]:5.0f} min  mean CPU {100 * cpu.mean() / CORES:4.0f} %  mean DRAM {100 * mem.mean() / GIB:4.0f} %  peak DRAM {100 * mem.max() / GIB:4.0f} %  peak swap {swp.max():5.1f} GiB  mean swap {swp.mean():5.1f} GiB")
    if UNITS == "abs":  # the dotted line marks the sandboxes' capacity: 56 cores, 112 GiB = 120 GB
        for ax, yl, cap, ticks in ((a, "CPU (cores)", CORES, [0, 14, 28, 42, 56]), (b, "DRAM (GB)", GIB * GB, [0, 30, 60, 90, 120])):
            ax.axhline(cap, color="#AAAAAA", linewidth=0.7, linestyle=":"); ax.set_ylim(0, cap * 1.08); ax.set_yticks(ticks); ax.set_ylabel(yl)
    else:
      for ax, yl in ((a, "CPU (%)"), (b, "DRAM (%)")):
        ax.axhline(100, color="#AAAAAA", linewidth=0.7, linestyle=":"); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel(yl)
    c.set_ylabel("Swap (GiB)"); c.set_ylim(bottom=0)
    for ax in (a, b, c):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    c.set_xlabel("Time (min)"); c.set_xlim(0, XMAX or xmax * 1.02)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.align_ylabels((a, b, c)); fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02); print(OUT)
