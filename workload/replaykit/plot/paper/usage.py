"""The sandbox CPU and memory series of one replay unit, for the task-usage figures.

CPU is the average of the cgroup's cumulative CPU time over a centred window (WINDOW s, default 10): an average over
the sampler's 250-ms samples, never clipped, so a sandbox at its limit shows a flat plateau and a tool burst its real
size. Memory and swap are the raw samples. Model calls are the request-to-last-byte spans of the tape."""
import os, csv, json, bisect, datetime as dt

WINDOW = float(os.environ.get("WINDOW", "10"))  # seconds of CPU time averaged into each point

GREEN, BLUE, BAND = "#009E73", "#0072B2", "#EBEBEB"  # Okabe-Ito, as the paper 2x3
RC = {"font.family": "sans-serif", "font.weight": "bold", "mathtext.default": "bf",  # the fonts of throughput_2x3.py (DejaVu Sans Bold)
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
      "xtick.direction": "in", "ytick.direction": "in", "legend.fontsize": 8}


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def window_rate(t, c, w):
    """Average rate of the cumulative counter c over a centred window of w (t units): (c[b] - c[a]) / (t[b] - t[a])."""
    h = w / 2; out = []
    for i, ti in enumerate(t):
        a = bisect.bisect_left(t, ti - h); b = min(len(t) - 1, bisect.bisect_right(t, ti + h) - 1)
        out.append((c[b] - c[a]) / (t[b] - t[a]) if t[b] > t[a] else 0.0)
    return out


def load(unit_dir):
    """(minutes, cores, memory GiB, swap GiB, [(call start, call end) in minutes]) of one unit."""
    rows = [r for r in csv.DictReader(open(os.path.join(unit_dir, "tape.telemetry.csv"))) if r["cpu_percent"]]
    t0 = ts(rows[0]["time"])
    sec = [(ts(r["time"]) - t0).total_seconds() for r in rows]
    cpu = window_rate(sec, [float(r["cpu_ms"]) / 1000 for r in rows], WINDOW)
    x = [s / 60 for s in sec]
    mem = [float(r["memory_bytes"]) / 2**30 for r in rows]
    swap = [float(r["swap_bytes"]) / 2**30 for r in rows]
    tp = json.load(open(os.path.join(unit_dir, "tape.json")))
    off = (ts(tp["meta"]["started_at"]) - t0).total_seconds() / 60
    calls = [(off + tm["request_at"] / 60000, off + tm["last_byte_at"] / 60000) for th in tp["threads"] for s in th["steps"]
             for tm in [s.get("timing") or {}] if "request_at" in tm and tm.get("last_byte_at") is not None]
    return x, cpu, mem, swap, calls


def draw(ax, b, x, cpu, mem, swap, calls):
    """One panel: CPU on ax (left axis), memory and swap on b (its twin). Returns whether swap was drawn."""
    for s, e in calls:
        ax.axvspan(s, e, color=BAND, linewidth=0, zorder=0)
    ax.plot(x, cpu, color=GREEN, linewidth=0.7)
    b.plot(x, mem, color=BLUE, linewidth=0.9, linestyle="--")
    drew = max(swap) > 0
    if drew:
        b.plot(x, swap, color=BLUE, linewidth=0.9, linestyle=":")
    ax.set_xlabel("Time (min)"); ax.set_xlim(0, x[-1])
    return drew


def legend_handles(swap):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    h = [plt.Line2D([], [], color=GREEN, linewidth=0.7, label="CPU (cores)"), plt.Line2D([], [], color=BLUE, linewidth=0.9, linestyle="--", label="Memory (GiB)")]
    if swap:
        h.append(plt.Line2D([], [], color=BLUE, linewidth=0.9, linestyle=":", label="Swap (GiB)"))
    h.append(Patch(facecolor=BAND, edgecolor="#999999", linewidth=0.5, label="Model call"))
    return h
