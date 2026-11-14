"""Command line entry point: `tidebox <command>`.

    tidebox replay    --llm --tape run.json [--listen :8081] [--output replay.json] [--speed N]
                      [--telemetry --container tidelet:NAME]
    tidebox inspect   run.json               summary: threads, steps, tool calls, timing
    tidebox compare   run.json replay.json   how far the replay got and where it diverged

`replay --llm` answers a real harness's model calls from the tape; the harness runs in the sandbox and
performs the tool calls itself. `--telemetry` samples the sandbox's CPU and memory every 250 ms into
`<output stem>.telemetry.csv`, next to the replay tape.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("tidebox")


def _listen(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    return (host or "0.0.0.0"), int(port)


def _sampler(args: argparse.Namespace, output: str | None, container: str | None, clock, client=None):
    """The telemetry sampler the flags ask for, or None. `client` overrides the Docker stats
    source (a Tidelet sandbox is sampled through tideletd)."""
    if not args.telemetry:
        return None
    if not output:
        raise SystemExit("--telemetry needs --output: the CSV is written next to the tape")
    if not container:
        raise SystemExit("--telemetry needs to know the container: --container NAME (tidelet:NAME or docker:NAME)")
    from tidebox.telemetry import Sampler, telemetry_path

    if container.startswith("tidelet:"):  # an --llm replay names the sandbox to sample this way
        from tidebox.telemetry import TideletStats

        container, client = container[len("tidelet:"):], TideletStats()
    elif container.startswith("docker:"):
        container = container[len("docker:"):]
    path = telemetry_path(output)
    return Sampler(container, path, clock, client=client)


def cmd_replay(args: argparse.Namespace) -> int:
    from tidebox.api.io import validate
    from tidebox.api.lazy import LazyTape

    # Read on demand: one pass indexes the steps, each is parsed only when it is played, so a
    # controller's memory does not grow with the tape (see tidebox.api.lazy).
    tape = LazyTape(args.tape)
    import atexit

    atexit.register(tape.close)
    for problem in validate(tape):
        log.warning("tape: %s", problem)
    steps = sum(len(t.steps) for t in tape.threads)
    output = Path(args.output) if args.output else None
    from tidebox.controller.server import ControllerServer

    host, port = _listen(args.listen or ":8081")
    server = ControllerServer(tape, args.speed, output, host, port)
    server.sampler = _sampler(args, args.output, args.container, server.now)
    log.info("replaying %s (%d steps) on %s:%d, speed %g%s", tape.meta.tape_id, steps, host, port, args.speed, f", recording to {output}" if output else "")
    server.run()
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from tidebox.api.io import read_tape, validate

    tape = read_tape(args.tape)
    m = tape.meta
    print(f"tape {m.tape_id}: {m.harness} {m.harness_version or ''} on {m.provider}/{m.model}, started {m.started_at}" + (f", replay of {m.replay_of}" if m.replay_of else ""))
    if m.task:
        print("task: " + ", ".join(f"{k}={v}" for k, v in m.task.items()))
    for problem in validate(tape):
        print(f"problem: {problem}")
    for thread in tape.threads:
        calls = sum(len(s.tool_calls) for s in thread.steps)
        first = thread.steps[0].timing.request_at if thread.steps else 0
        last = thread.steps[-1].timing if thread.steps else None
        end = (last.next_request_at if last and last.next_request_at is not None else last.last_byte_at if last else None) or first
        parent = f" (child of {thread.parent[0]} via {thread.parent[1]})" if thread.parent else ""
        print(f"thread {thread.id}{parent}: {len(thread.steps)} steps, {calls} tool calls, {(end - first) / 1000:.1f}s")
        if args.verbose:
            for step in thread.steps:
                t = step.timing
                latency = (t.last_byte_at - t.request_at) if t.last_byte_at is not None else 0
                gap = (t.next_request_at - t.last_byte_at) if t.next_request_at is not None and t.last_byte_at is not None else None
                names = ", ".join(c.name for c in step.tool_calls) or "(text)"
                print(f"  step {step.index:3d} @{t.request_at / 1000:8.1f}s  model {latency / 1000:6.1f}s  tools {'' if gap is None else f'{gap / 1000:.1f}s':>7}  {names}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import json

    from tidebox.api.lazy import LazyTape
    from tidebox.controller.compare import compare, summarize

    with LazyTape(args.recording) as recording, LazyTape(args.replay) as replay:  # steps are read one at a time
        report = compare(recording, replay)
    print(json.dumps(report, indent=2) if args.json else summarize(report))
    return 0 if report["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tidebox", description="replay a recorded agent run in a sandbox without a model")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("replay", help="replay a tape: serve it as the LLM to a real harness (--llm)")
    r.add_argument("--tape", required=True)
    r.add_argument("--llm", action="store_true", default=True,
                   help="LLM replay (the only mode): the real harness runs, pointed at --listen, and its model calls are answered from the tape")
    r.add_argument("--listen", default=None, help="host:port to serve the provider API on (default :8081)")
    r.add_argument("--output", default=None, help="record the replay as a tape to this file")
    r.add_argument("--speed", type=float, default=1.0, help="1 = recorded pace, 2 = twice as fast, 0 = no waiting")
    r.add_argument("--telemetry", action="store_true", help="sample the sandbox every 250 ms into <output stem>.telemetry.csv")
    r.add_argument("--container", default=None, help="the sandbox to sample: tidelet:NAME, or a Docker container NAME / docker:NAME")
    r.set_defaults(func=cmd_replay)

    i = sub.add_parser("inspect", help="summarize a tape")
    i.add_argument("tape")
    i.set_defaults(func=cmd_inspect)

    c = sub.add_parser("compare", help="diff a replay tape against its recording")
    c.add_argument("recording")
    c.add_argument("replay")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
