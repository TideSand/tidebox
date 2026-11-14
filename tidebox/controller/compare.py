"""How far a replay got and where it differed from the recording.

Both inputs are tapes: the recording, and the replay tape the controller wrote with
`--output`. Reports steps replayed vs. recorded per thread, the first step whose tool calls or
results differ, per-tool-call outcome (same / different output / error where there was none),
and recorded vs. replayed durations side by side. Used by `tidebox compare`.
"""

from __future__ import annotations

from tidebox.api.tape import Step, Tape, Thread


def _duration(thread: Thread) -> int | None:
    if not thread.steps:
        return None
    first = thread.steps[0].timing.request_at
    last = thread.steps[-1].timing
    end = last.next_request_at if last.next_request_at is not None else last.last_byte_at
    return None if end is None else end - first


def _compare_step(recorded: Step, replayed: Step) -> dict:
    calls = {}
    rec_results = {r.tool_call_id: r for r in recorded.tool_results}
    rep_results = {r.tool_call_id: r for r in replayed.tool_results}
    for call in recorded.tool_calls:
        rep_call = next((c for c in replayed.tool_calls if c.id == call.id), None)
        a, b = rec_results.get(call.id), rep_results.get(call.id)
        if rep_call is None:
            outcome = "not run"
        elif b is None:
            outcome = "no result"
        elif a is None:
            outcome = "unrecorded"
        elif a.text() == b.text() and a.is_error == b.is_error:
            outcome = "same"
        elif b.is_error and not a.is_error:
            outcome = "error"
        else:
            outcome = "different"
        calls[call.id] = {
            "name": call.name,
            "outcome": outcome,
            "recorded_ms": a.duration_ms if a else None,
            "replayed_ms": b.duration_ms if b else None,
        }
    return {
        "index": recorded.index,
        "same": all(c["outcome"] == "same" for c in calls.values()),
        "calls": calls,
        "recorded_gap_ms": None if recorded.timing.next_request_at is None or recorded.timing.last_byte_at is None else recorded.timing.next_request_at - recorded.timing.last_byte_at,
        "replayed_gap_ms": None if replayed.timing.next_request_at is None or replayed.timing.last_byte_at is None else replayed.timing.next_request_at - replayed.timing.last_byte_at,
    }


def compare(recording: Tape, replay: Tape) -> dict:
    replay_threads = {t.id: t for t in replay.threads}
    threads = {}
    for rec in recording.threads:
        rep = replay_threads.get(rec.id)
        steps = [_compare_step(a, b) for a, b in zip(rec.steps, rep.steps)] if rep else []
        first_diff = next((s["index"] for s in steps if not s["same"]), None)
        threads[rec.id] = {
            "recorded_steps": len(rec.steps),
            "replayed_steps": len(rep.steps) if rep else 0,
            "complete": bool(rep) and len(rep.steps) == len(rec.steps),
            "first_difference": first_diff,
            "recorded_ms": _duration(rec),
            "replayed_ms": _duration(rep) if rep else None,
            "steps": steps,
        }
    recorded_total = sum(len(t.steps) for t in recording.threads)
    replayed_total = sum(len(t.steps) for t in replay.threads)
    return {
        "recording": recording.meta.tape_id,
        "replay": replay.meta.tape_id,
        "recorded_steps": recorded_total,
        "replayed_steps": replayed_total,
        "complete": replayed_total == recorded_total and all(t["complete"] for t in threads.values()),
        "identical": all(s["same"] for t in threads.values() for s in t["steps"]) and replayed_total == recorded_total,
        "threads": threads,
    }


def summarize(report: dict) -> str:
    lines = [f"recording {report['recording']}  replay {report['replay']}"]
    lines.append(f"steps: {report['replayed_steps']}/{report['recorded_steps']} replayed, "
                 + ("identical" if report["identical"] else "complete" if report["complete"] else "incomplete"))
    for tid, t in report["threads"].items():
        rec = f"{t['recorded_ms']/1000:.1f}s" if t["recorded_ms"] is not None else "?"
        rep = f"{t['replayed_ms']/1000:.1f}s" if t["replayed_ms"] is not None else "?"
        diff = "" if t["first_difference"] is None else f", first difference at step {t['first_difference']}"
        lines.append(f"  {tid}: {t['replayed_steps']}/{t['recorded_steps']} steps, {rec} recorded vs {rep} replayed{diff}")
        for s in t["steps"]:
            for cid, c in s["calls"].items():
                if c["outcome"] != "same":
                    lines.append(f"    step {s['index']} {c['name']} ({cid}): {c['outcome']}")
    return "\n".join(lines)
