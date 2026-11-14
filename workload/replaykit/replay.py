#!/usr/bin/env python3
"""Replay recorded TideBox tapes in Tidelet sandboxes under a tideletd policy, for any benchmark.

    replay.py --data data/algotune/run4-max-50 --all --policy strict --repeat 2 --proxy http://10.88.0.1:18800
    replay.py --data data/terminal-bench-4.0/run2-medium some-task other-task --speed 0

Each task's recording (`<data>/<task>/`) is played back inside a sandbox that runs the task's
environment image (the recording's meta.json names it) with the limits the recording had, the
real Claude Code (the version that made the recording, bind-mounted) inside it, and its model
calls answered from the tape by TideBox's replay controller (`tidebox replay --llm`); the
sandbox's footprint is the harness plus its tools, as in the recording. The output is the same
for every benchmark:

    output/replay/<bench>/<run>/<name>/            <name>: the policy (--name to change)
        run.json                                   the setup: data, policy, repeat, speed, revisions
        replay.log                                 this driver's log
        <task>/  or  <task>__r<k>/                 one directory per replay; copy k with --repeat N
            tape.json                              the replay tape
            tape.telemetry.csv                     CPU and memory of the sandbox every 250 ms, on the replay's clock
            meta.json                              the sandbox as created, admission, timings, the harness, the
                                                   compare summary against the recording (replay_of, task, copy)
            compare.txt, compare.json              `tidebox compare` against the recording
            claude.log, controller.log             Claude Code's stream-json output, the controller's log
            verifier/                              --verify: the task's own test.sh output and its reward

While a replay runs, `status.json` holds what `meta.json` will, as far as it is known, with a
`state` (rejected, queued, starting, running, done, error), and the controller appends completed
steps to `tape.json.steps.jsonl`, so a run can be watched and plotted while it goes. A task with a
complete replay in the output directory is skipped unless `--redo`.

Arrivals are the workload's: the driver asks for one new sandbox every `--interval` seconds (15 by
default; 3 for the Terminal-Bench runs), in the selected order, and the daemon's policy answers
each request at once: start it now, queue it in the daemon and start it later (tidesand, while a
boot is in progress), or reject it (ResourceExhausted: strict with no room, the load gates with the
pool over their threshold), in which case the same sandbox is offered again at the next interval
and the ones behind it wait. So the policy, not this script, decides how many replays run at once;
the interval decides how fast they are offered. meta.json's `admission` block records the
requests (rejections, the queue wait) and the node state at the start. `--repeat N` offers N
copies of every task (`<task>__r<k>`), so the pool carries N times the recorded demand, and
`--order random` shuffles the copies together. `--proxy` sends Claude Code's model calls through
tideletd's LLM proxy, which forwards each to its sandbox's controller (the sandbox's
`tidelet.io/llm-upstream` label), so the policies that schedule on model calls see them. The
driver re-executes itself with sudo for tideletd's socket. `--verify` runs the task's own verifier
(`tests/test.sh` from the benchmark's task copy) in the sandbox after the agent, so the reward is
measured under the same policy.

Claude Code's API and stream timeouts are raised to an hour so that a response a policy holds in
the proxy is not mistaken for a dead model; the CLI launch's interactive tools that the
recordings' Agent SDK launch did not offer are disallowed (`--disallowed-tools`), so the controller
pairs each opening request with its recording by tool count.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import resource
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from replaykit import benches, layout
from replaykit.paths import TIDEBOX, TIDEBOX_CLI, user_home

log = logging.getLogger("replay")

HOME = user_home()
TIDELET_API = os.environ.get("TIDELET_API", "http://127.0.0.1:7474")
# LLM mode: Claude Code bind-mounted into the sandbox, as Harbor did for the recording, and the
# address the sandbox reaches the controller at (the host's end of the tidelet bridge).
CLAUDE_VERSIONS = HOME / ".local" / "share" / "claude" / "versions"
BRIDGE_IP = os.environ.get("TIDELET_BRIDGE_IP", "10.88.0.1")
CONTROLLER_PORT_BASE = 18200
PORT_SPAN = 2000  # ports tried from the base: one controller per sandbox, hundreds at once with --repeat
DEFAULT_INTERVAL = 15.0  # s between requests to start a sandbox (see the module docstring)
REJECT_RETRY = 5.0  # s before a rejected request is repeated when --interval is 0
# Tools the CLI launch offers that the recordings' Agent SDK launch did not (see the docstring).
DISALLOWED_TOOLS = "DesignSync,Monitor,PushNotification"
# Claude Code's model-call timeouts, raised so a response held in the proxy is waited for.
# API_FORCE_IDLE_TIMEOUT=0 turns off the HTTP client's own idle timeout (Bun's fetch gives up on a
# response silent for 5-6 minutes), which Claude Code leaves on for any base URL but
# api.anthropic.com: without it a response held that long ends the run with "The operation timed out".
HARNESS_TIMEOUT_ENV = {"API_TIMEOUT_MS": "3600000", "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": "3600000",
                       "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "3600000", "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "3600000",
                       "API_FORCE_IDLE_TIMEOUT": "0"}
HARNESS = {"disallowed_tools": DISALLOWED_TOOLS, "env": dict(HARNESS_TIMEOUT_ENV)}  # set from the command line in main()
# A controller indexes its tape when it starts (one pass over the file, a few MB of memory);
# a few may do so at once.
_controller_start_lock = threading.Semaphore(4)
_compare_lock = threading.Semaphore(4)  # `tidebox compare` reads both tapes step by step
CONTROLLER_LOAD_TIMEOUT = 1800.0
CONTROLLER_FINALIZE_TIMEOUT = 1800.0  # assembling a large replay tape on a saturated host
RESULT_GRACE = 300  # seconds an agent may live on after its final result before the driver ends it
POOL_CGROUP = os.environ.get("POOL_CGROUP", "/sys/fs/cgroup/moby")  # the sandboxes' cgroup, killed as one at a cutoff


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- the backend: how a sandbox is created, entered and watched ------------------------------------


INSPECT_RETRY_S = float(os.environ.get("INSPECT_RETRY_S", "300"))  # how long a failing `tidelet inspect` is retried (a daemon stall under IO pressure can last minutes)


class TideletBackend:
    """A Tidelet sandbox through the `tidelet` CLI (TIDELET_BIN, default <tidelet checkout>/bin/tidelet;
    the checkout is TideBox's parent) and tideletd's REST API (TIDELET_API)."""

    name = "tidelet"
    needs_root = True

    def __init__(self) -> None:
        self.cli = Path(os.environ.get("TIDELET_BIN", str(TIDEBOX.parent / "bin" / "tidelet")))

    def run(self, *args: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.cli), *args], check=check, capture_output=True, text=True, **kw)

    def inspect(self, name: str) -> dict:
        """The sandbox's record. A CLI failure that is not NotFound (tideletd stalled under IO
        pressure, a refused or reset connection) is retried with backoff for INSPECT_RETRY_S, so
        a queued sandbox is not given up on because the daemon paused for a minute."""
        deadline = time.time() + INSPECT_RETRY_S
        delay = 1.0
        while True:
            r = self.run("inspect", "--json", name, check=False)
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout)
            err = (r.stderr or "").strip()
            if "NotFound" in err or "not found" in err.lower() or time.time() > deadline:
                raise RuntimeError(f"tidelet inspect {name}: exit {r.returncode}: {err[-300:]}")
            time.sleep(delay)
            delay = min(delay * 2, 15.0)

    def node(self) -> dict | None:
        return json.loads(self.run("node", "--json").stdout)

    def policy(self) -> str | None:
        try:
            return (self.node() or {}).get("policy")
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            detail = getattr(e, "stderr", "") or str(e)
            raise SystemExit(f"tideletd is not reachable through {self.cli} (is it running? TIDELET_API={TIDELET_API}): {detail.strip()[-300:]}") from None

    def stats(self, name: str) -> dict | None:
        r = self.run("stats", "--json", name, check=False)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None

    def remove(self, name: str) -> None:
        self.run("rm", "-f", name, check=False)

    def submit(self, name: str, image: str, cpus: float, memory_mb: int, mounts: list[str], workdir: str, labels: dict[str, str],
               env: dict[str, str]) -> tuple[str, dict | None, str]:
        """One request to start the sandbox. Returns (outcome, the sandbox as inspected, the
        daemon's words): outcome "run" (created, starting), "queued" (accepted into tideletd's queue,
        to be started when the policy says) or "rejected" (the policy has no room for it now, the
        CLI's ResourceExhausted; nothing was created, and the caller may ask again later)."""
        run_args = ["run", "-d", "--name", name, "--image", image, "--cpus", f"{cpus:g}", "--memory", f"{memory_mb}Mi"]
        for k, v in labels.items():
            run_args += ["--label", f"{k}={v}"]
        for m in mounts:
            run_args += ["-v", m]
        for k, v in env.items():
            run_args += ["--env", f"{k}={v}"]
        run_args += ["-w", workdir, "--", "sleep", "infinity"]
        run = self.run(*run_args, check=False)
        if run.returncode != 0 and "already exists" in run.stderr:
            # A sandbox of this name survived an earlier run (a daemon restart keeps containerd's
            # containers): it is not this unit's, so it goes, and the request is made once more.
            log.warning("%s: a stale sandbox of that name exists; removing it and asking again", name)
            self.remove(name)
            run = self.run(*run_args, check=False)
        if run.returncode != 0:
            err = run.stderr.strip()
            if "ResourceExhausted" in err:
                return "rejected", None, err.rsplit("rejected by policy:", 1)[-1].strip()[-300:]
            raise RuntimeError(f"tidelet run: {err[-500:]}")
        sb = self.inspect(name)
        status = sb.get("status") or {}
        return ("queued" if status.get("phase") == "PHASE_QUEUED" else "run"), sb, status.get("message") or ""

    def wait_running(self, name: str, deadline: float, queued: bool) -> dict:
        """Polls until the sandbox runs; `deadline` is an epoch time after which it is given up on."""
        sb = self.inspect(name)
        while (sb.get("status") or {}).get("phase") not in ("PHASE_RUNNING", "PHASE_EXITED", "PHASE_FAILED"):
            if time.time() > deadline:
                raise RuntimeError(f"sandbox not running by the admission timeout (phase {(sb.get('status') or {}).get('phase')})")
            time.sleep(5.0 if queued else 0.5)
            sb = self.inspect(name)
        if (sb.get("status") or {}).get("phase") != "PHASE_RUNNING":
            raise RuntimeError(f"sandbox is {(sb.get('status') or {}).get('phase')}: {(sb.get('status') or {}).get('message')}")
        return sb

    def exec_cmd(self, name: str, cmd: list[str], env: dict[str, str], workdir: str | None) -> list[str]:
        out = [str(self.cli), "exec", "-i"]
        if workdir:
            out += ["-w", workdir]
        out += [f"--env={k}={v}" for k, v in env.items()]
        return [*out, name, "--", *cmd]

    def telemetry_target(self, name: str) -> str:
        return f"tidelet:{name}"

    def controller_env(self) -> dict[str, str]:
        return {"TIDELET_API": TIDELET_API}

    def limits_of(self, created: dict, requested: dict) -> dict:
        """The limits as tideletd applied them, in the shape of the recordings' meta.json."""
        limits = (((created.get("status") or {}).get("allocated") or {}).get("limits") or {})
        cpus = int(limits.get("cpu_millicores") or 0) / 1000 or requested["cpus"]
        memory_mb = int(limits.get("memory_bytes") or 0) // 2**20 or requested["memory_mb"]
        return {"cpus": cpus, "memory_mb": memory_mb}


# -- small helpers -------------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    """Writes beside and renames, so a reader watching the run never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_status(dest: Path, result: dict, state: str) -> None:
    """`status.json`: the replay's record so far, while it runs (see the module docstring)."""
    _write_json(dest / "status.json", {**result, "state": state})


def write_meta(dest: Path, result: dict) -> None:
    """`meta.json`, the record of a replay that has ended, and its final `status.json`."""
    _write_json(dest / "meta.json", result)
    write_status(dest, result, "error" if result.get("error") else "done")


def git_rev(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def write_run_json(out: Path, info: dict) -> None:
    """`run.json` at the run's root: the setup, merged into what a launcher (replay_policies.sh)
    already wrote there."""
    path = out / "run.json"
    try:
        current = json.loads(path.read_text())
    except (OSError, ValueError):
        current = {}
    current.update(info)
    _write_json(path, current)


def agent_seconds(meta: dict) -> float:
    ae = (meta.get("harbor_result") or {}).get("agent_execution") or {}
    try:
        return (dt.datetime.fromisoformat(ae["finished_at"]) - dt.datetime.fromisoformat(ae["started_at"])).total_seconds()
    except (KeyError, ValueError):
        return float("inf")


def quickest(data: Path, n: int) -> list[str]:
    metas = {p.parent.name: json.loads(p.read_text()) for p in data.glob("*/meta.json") if (p.parent / "tape.json").exists()}
    return sorted(metas, key=lambda t: agent_seconds(metas[t]))[:n]


def instruction_of(first_step) -> str:
    """The task instruction as the harness received it: the first user message's last text
    block that is not a system reminder, from the recording's first step."""
    first = first_step.request.messages[0]
    texts = [b.text for b in first.content if getattr(b, "text", None) and "<system-reminder>" not in b.text]
    return texts[-1]


def claude_bin_for(meta: dict) -> Path:
    """The Claude Code to run: CLAUDE_BIN when set; else the version that made the recording
    (`claude_code_version` in its meta.json) from the versions directory; else the newest version
    installed there; else `~/.local/bin/claude`."""
    if "CLAUDE_BIN" in os.environ:
        return Path(os.environ["CLAUDE_BIN"])
    version = meta.get("claude_code_version") or Path(meta.get("claude_binary") or "").name
    if version and (CLAUDE_VERSIONS / str(version)).exists():
        return CLAUDE_VERSIONS / str(version)
    installed = sorted((p for p in CLAUDE_VERSIONS.glob("*") if p.is_file()), key=lambda p: [int(x) if x.isdigit() else x for x in p.name.split(".")])
    if installed:
        if version:
            log.warning("Claude Code %s (the recording's) is not installed; using %s", version, installed[-1].name)
        return installed[-1]
    return HOME / ".local" / "bin" / "claude"


# -- controller ports: one per sandbox, reserved across every driver on the host -----------------

_ports_lock = threading.Lock()
_ports_taken: set[int] = set()
PORT_LOCKS = Path(os.environ.get("TIDEBOX_PORT_LOCKS", "/tmp/tidebox-ports"))  # shared by every driver on the host


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def free_port(host: str, start: int) -> int:
    """A controller port no replay holds, in this process or any other driver on the host, and
    nothing listens on. A controller only listens once it has indexed its tape, so a bind test
    alone is not enough: a lock file per port, created atomically and holding the driver's
    pid, reserves it across processes until `release_port`."""
    PORT_LOCKS.mkdir(parents=True, exist_ok=True)
    with _ports_lock:
        for port in range(start, start + PORT_SPAN):
            if port in _ports_taken:
                continue
            lock = PORT_LOCKS / f"{host}:{port}"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = int(lock.read_text().strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if owner and _pid_alive(owner):
                    continue
                lock.unlink(missing_ok=True)  # stale: its driver is gone
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            with socket.socket() as sock:
                # A finished controller leaves its port in TIME_WAIT for a minute; that is not a
                # listener, and SO_REUSEADDR lets the bind test see through it (the controller
                # binds the same way). A port something else listens on is skipped, and its lock
                # dropped so it is tried again later: a kept lock would leak the port for the
                # driver's whole life, which at x16 (1,632 sandboxes) exhausted the 2,000-port pool.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((host, port))
                except OSError:
                    lock.unlink(missing_ok=True)
                    continue
            _ports_taken.add(port)
            return port
    raise RuntimeError("no free port")


def release_port(host: str, port: int) -> None:
    with _ports_lock:
        _ports_taken.discard(port)
        (PORT_LOCKS / f"{host}:{port}").unlink(missing_ok=True)


def wait_controller(url: str, proc: subprocess.Popen, seconds: float, tape_id: str | None = None) -> bool:
    """True once *our* controller answers at `url` (its state names `tape_id`); False if it
    exits first or `seconds` pass. Another task's controller on the port is not ours."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    state = json.load(resp)
                    if tape_id is None or state.get("tape") == tape_id:
                        return True
                    log.warning("controller at %s serves tape %s, not ours (%s); waiting", url, state.get("tape"), tape_id)
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    return False


def controller_state(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.load(resp)
    except (OSError, ValueError):
        return None


class Pacer:
    """The workload's arrival clock: one request to start a sandbox per `interval` seconds.
    `wait` sleeps until the next request may go and books it; `rejected` says the request just made
    was rejected, so that the next (the same sandbox again) waits the interval even when it is 0."""

    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self.next_at = 0.0

    def wait(self) -> None:
        at = max(time.time(), self.next_at)
        if at > time.time():
            time.sleep(at - time.time())
        self.next_at = at + self.interval

    def rejected(self) -> None:
        if self.interval <= 0:
            self.next_at = time.time() + REJECT_RETRY


# -- one replay ------------------------------------------------------------------------------------


# RUN_STOPPED is set when the run is cut off or interrupted: a sandbox the policy dispatches from
# its queue after that is removed, not started (the cut-off stops the sandboxes it finds running,
# which frees room, and the admission queue then dispatched hundreds more into a run that was over:
# a SIGTERM stop of a large run did not converge in 10 minutes).
RUN_STOPPED = threading.Event()


class Replay:
    """One task's replay: its recording, its sandbox, its output directory."""

    def __init__(self, task: str, copy: int | None, args, backend, bench) -> None:
        self.task, self.copy, self.args, self.backend, self.bench = task, copy, args, backend, bench
        self.unit = task if copy is None else f"{task}__r{copy}"
        self.task_dir = args.data / task
        self.meta = json.loads((self.task_dir / "meta.json").read_text())
        self.tape = self.task_dir / "tape.json"
        self.dest = args.out / self.unit
        self.name = sandbox_name(bench.short if bench else None, self.meta.get("bench") or "bench", task, copy)
        limits = (self.meta.get("container") or {}).get("limits") or {}
        self.cpus = float(limits.get("cpus") or 1)
        self.memory_mb = int(limits.get("memory_mb") or 1024)
        self.workdir = (self.meta.get("container") or {}).get("workdir") or "/app"
        self.image = image_of(self.meta, task, args.image_prefix)
        self.result: dict = {
            "bench": self.meta.get("bench") or (bench.name if bench else None), "task": task, "copy": copy, "unit": self.unit,
            "policy": args.policy, "name": args.run_name, "mode": "llm", "backend": backend.name, "replay_of": str(self.tape),
            "sandbox": {"name": self.name, "image": self.image, "requested": {"cpus": self.cpus, "memory_mb": self.memory_mb}, "workdir": self.workdir},
            "replay": {"speed": args.speed}, "tape_mb": round(self.tape.stat().st_size / 2**20),
        }

    # -- the sandbox ---------------------------------------------------------------------------

    def prepare(self) -> None:
        """Before the first request: a clean output directory, no stale sandbox of this name, and
        what the request carries (the harness mount, the proxy label)."""
        args, backend, dest = self.args, self.backend, self.dest
        dest.mkdir(parents=True, exist_ok=True)
        for stale in ("tape.json", "tape.telemetry.csv", "claude.log", "controller.log", "compare.txt", "compare.json", "meta.json", "tape.json.steps.jsonl"):
            (dest / stale).unlink(missing_ok=True)
        backend.remove(self.name)
        self.mount = f"{claude_bin_for(self.meta)}:/usr/local/bin/claude:ro"
        self.labels: dict[str, str] = {}
        if args.proxy:
            # The proxy forwards this sandbox's model calls to its own controller: reserve the
            # controller's port now so the label can name it.
            port = free_port(BRIDGE_IP, CONTROLLER_PORT_BASE)
            self.result["replay"]["controller_port"] = port
            self.labels["tidelet.io/llm-upstream"] = f"http://{BRIDGE_IP}:{port}"
        self.env: dict[str, str] = {}
        self.submitted = time.time()
        self.result["admission"] = {"policy": args.policy, "interval": args.interval, "submitted_at": utc(self.submitted), "rejections": 0}
        log.info("%s: asking for %s %s from %s (%g cpus, %d MB)%s", self.unit, backend.name, self.name, self.image, self.cpus, self.memory_mb,
                 " through the llm proxy" if self.labels else "")

    def submit(self) -> str:
        """One request to start the sandbox, the workload's arrival (see the module docstring).
        Returns the policy's answer, "run", "queued" or "rejected" (offer it again later), or
        "error" when the request itself failed, which ends the unit with an error meta.json."""
        args, backend, result = self.args, self.backend, self.result
        adm = result["admission"]
        try:
            outcome, sb, words = backend.submit(self.name, self.image, self.cpus, self.memory_mb, [self.mount], self.workdir, self.labels, self.env)
        except Exception as e:  # noqa: BLE001
            return self.fail(f"{backend.name}: {e}")
        if outcome == "rejected":
            adm["rejections"] += 1
            adm["last_rejection"] = words
            level = log.info if adm["rejections"] == 1 or adm["rejections"] % 20 == 0 else log.debug
            level("%s: rejected by the %s policy (%s); offered again in %g s (%d rejections so far)", self.unit, args.policy, words,
                  args.interval or REJECT_RETRY, adm["rejections"])
            write_status(self.dest, result, "rejected")
            if time.time() - self.submitted > args.admission_timeout:
                return self.fail(f"not admitted within {args.admission_timeout:.0f} s: {adm['rejections']} rejections, the last: {words}")
            return "rejected"
        adm.update(accepted_at=utc(), queued=outcome == "queued")
        if words:
            adm["reason"] = words
        result["sandbox"]["created"] = sb
        log.info("%s: %s by the %s policy%s%s", self.unit, "queued" if outcome == "queued" else "started", args.policy, f" ({words})" if words else "",
                 f" after {adm['rejections']} rejections" if adm["rejections"] else "")
        write_status(self.dest, result, "queued" if outcome == "queued" else "starting")
        return outcome

    def fail(self, error: str) -> str:
        """Ends the unit before it ran: the error goes into meta.json, nothing is left behind."""
        self.result["error"] = error
        log.error("%s: %s", self.unit, error)
        self.backend.remove(self.name)
        if port := self.result["replay"].get("controller_port"):
            release_port(BRIDGE_IP, port)
        write_meta(self.dest, self.result)
        return "error"

    # -- the harness --------------------------------------------------------------------------

    def run_llm(self) -> int:
        """LLM replay: the controller serves the tape as the model; Claude Code runs in the
        sandbox and performs the tools itself. Returns Claude Code's exit code."""
        from tidebox.api.lazy import read_head

        args, backend, dest = self.args, self.backend, self.dest
        # Only the tape's meta and its first step are read here; the driver must not hold tapes in
        # memory, since it runs one thread per task.
        tape_meta, _, first_step = read_head(self.tape)
        effort = tape_meta.labels.get("effort") or self.meta.get("effort") or "max"
        model = tape_meta.model
        instruction = instruction_of(first_step)
        listen_ip = BRIDGE_IP
        port = self.result["replay"].get("controller_port") or free_port(listen_ip, CONTROLLER_PORT_BASE)
        self.result["replay"]["controller_port"] = port
        base_url = f"http://{listen_ip}:{port}"
        state_url = f"http://{listen_ip}:{port}/tidebox/state"
        ctrl_cmd = [str(TIDEBOX_CLI), "replay", "--llm", "--tape", str(self.tape), "--listen", f"{listen_ip}:{port}", "--output", str(dest / "tape.json"),
                    "--telemetry", "--container", backend.telemetry_target(self.name), "--speed", str(args.speed)]
        ctrl_log = open(dest / "controller.log", "w", encoding="utf-8")  # noqa: SIM115 - closed after the controller exits
        # Indexing a tape takes CPU for a while, so only a few controllers start at once; the sandbox
        # is already running meanwhile, and the wait is recorded.
        lock_wait_started = time.time()
        with _controller_start_lock:
            lock_wait = time.time() - lock_wait_started
            load_started = time.time()
            controller = subprocess.Popen(ctrl_cmd, stdout=ctrl_log, stderr=subprocess.STDOUT, env={**os.environ, **backend.controller_env()})
            up = wait_controller(state_url, controller, CONTROLLER_LOAD_TIMEOUT, tape_meta.tape_id)
        self.result["replay"].update(controller_lock_wait_sec=round(lock_wait, 1), controller_load_sec=round(time.time() - load_started, 1))
        if not up:
            controller.terminate()
            ctrl_log.close()
            raise RuntimeError(f"controller did not come up on {base_url} (exit {controller.poll()}); see {dest / 'controller.log'}")
        # A few task images run as a non-root user, for whom /root is not writable: use the
        # sandbox user's own home for Claude Code's config, or /tmp when that is not writable either.
        probe = subprocess.run(backend.exec_cmd(self.name, ["sh", "-c", 'd="${HOME:-/tmp}"; if mkdir -p "$d/.claude" 2>/dev/null; then echo "$d"; else mkdir -p /tmp/.claude && echo /tmp; fi'], {}, None),
                               capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
        home = (probe.stdout.strip().splitlines() or ["/tmp"])[-1] if probe.returncode == 0 else "/tmp"
        # With tideletd's LLM proxy in front (--proxy), Claude Code talks to the proxy and the proxy
        # forwards to this task's controller, named by the sandbox's upstream label at creation.
        proxy = args.proxy
        env = {
            "ANTHROPIC_BASE_URL": proxy or base_url, "ANTHROPIC_API_KEY": "tidebox-replay", "ANTHROPIC_MODEL": model,
            "IS_SANDBOX": "1", "CLAUDE_CONFIG_DIR": f"{home}/.claude", "HOME": home,
            "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            **HARNESS["env"],
        }
        # Harbor's command, with the model named explicitly since the tape's model is what the controller serves.
        disallowed = f" --disallowedTools {HARNESS['disallowed_tools']}" if HARNESS["disallowed_tools"] else ""
        harness = ("mkdir -p \"$CLAUDE_CONFIG_DIR\" && exec claude --verbose --output-format=stream-json "
                   f"--effort {effort} --permission-mode=bypassPermissions --model {model}{disallowed} --print")
        cmd = backend.exec_cmd(self.name, ["sh", "-c", harness], env, self.workdir)
        timeout = self.deadline()
        self.result["harness"] = {"binary": str(claude_bin_for(self.meta)), "command": harness, "env": {k: v for k, v in env.items() if k != "ANTHROPIC_API_KEY"},
                                  "controller": base_url, "proxy": proxy, "timeout_sec": timeout}
        log.info("%s: Claude Code in %s against %s (effort %s), log in %s", self.unit, self.name, base_url, effort, dest / "claude.log")
        with open(dest / "claude.log", "w", encoding="utf-8") as claude_log:
            code = self.run_agent(cmd, instruction, claude_log, dest / "claude.log", timeout)
        self.result["harness"]["state"] = controller_state(state_url)
        time.sleep(3)  # the harness exits as soon as it has the last response; let the controller finish recording that step
        # On SIGINT the controller assembles the replay tape from its spill file, which for a long
        # tape is over a gigabyte and, on a saturated host, takes minutes: wait for it.
        controller.send_signal(signal.SIGINT)
        t_fin = time.time()
        try:
            controller.wait(timeout=CONTROLLER_FINALIZE_TIMEOUT)
        except subprocess.TimeoutExpired:
            log.error("%s: controller still finalizing after %.0f s; killed (replay tape incomplete)", self.unit, CONTROLLER_FINALIZE_TIMEOUT)
            controller.kill()
            controller.wait()
        ctrl_log.close()
        self.result["harness"]["controller_exit_code"] = controller.returncode
        self.result["replay"]["controller_finalize_sec"] = round(time.time() - t_fin, 1)
        spill = dest / "tape.json.steps.jsonl"
        if spill.exists():  # left behind only when the controller did not finish assembling
            self.result["harness"]["spill_left_mb"] = round(spill.stat().st_size / 2**20)
            spill.unlink()
        release_port(listen_ip, port)
        return code

    def run_agent(self, cmd: list[str], instruction: str, claude_log, log_path: Path, timeout: float) -> int:
        """Runs the harness to its end, the deadline, or RESULT_GRACE seconds after its final result.
        The deadline: the replay fell hopelessly behind the recording, an outcome of the host's
        contention, recorded as such (the compare shows how far it got), not retried. The grace: an
        agent that has printed its final `result` event is done with the tape, but a background tool
        loop can keep it alive (a poll on a task id the replay cannot reproduce, a `pgrep -f` that
        matches itself, a job that failed at once): six such units once held a run open for six hours."""
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=claude_log, stderr=subprocess.STDOUT, text=True)
        try:
            proc.stdin.write(instruction)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        started = time.time()
        result_at: float | None = None
        pos = 0
        while True:
            try:
                return proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            if now - started > timeout:
                log.error("%s: Claude Code exceeded %.0f s; killed", self.unit, timeout)
                self.result["harness"]["timed_out"] = True
                proc.kill()
                proc.wait()
                return -9
            if result_at is None:
                try:
                    with open(log_path, encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    if '"type":"result"' in chunk or '"type": "result"' in chunk:
                        result_at = now
                except OSError:
                    pass
            elif now - result_at > RESULT_GRACE:
                log.warning("%s: agent still running %.0f s after its final result; ended (a tool process outlived the replay)", self.unit, now - result_at)
                self.result["harness"]["ended_after_result"] = True
                proc.kill()
                proc.wait()
                return -15

    def deadline(self) -> float:
        """The harness deadline: the policies overcommit the pool several times over, so a CPU-bound
        task may legitimately run several times slower than its recording, and a holding policy can
        keep each of its model responses back for a while; a task is only given up on when it falls
        far behind even that (--deadline overrides)."""
        if self.args.deadline:
            return float(self.args.deadline)
        steps = int((self.meta.get("tape_summary") or {}).get("steps") or 0)
        rec = agent_seconds(self.meta)
        return max(4 * rec, 1800) + 600 + 60 * steps if rec != float("inf") else 4 * 3600

    # -- the verifier --------------------------------------------------------------------------

    def verify(self) -> None:
        """Runs the task's own verifier (`tests/test.sh` from the benchmark's task copy) in the
        sandbox, as the recording did, and records its reward (`/logs/verifier/reward.txt`)."""
        tests = (self.bench.tasks_root / self.task / "tests") if self.bench and self.bench.tasks_root else None
        if tests is None or not (tests / "test.sh").exists():
            self.result["verifier"] = {"error": f"no tests/test.sh for {self.task} under the bench's task copies ({tests})"}
            log.warning("%s: %s", self.unit, self.result["verifier"]["error"])
            return
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(tests, arcname=".")
        subprocess.run(self.backend.exec_cmd(self.name, ["sh", "-c", "mkdir -p /tests /logs/verifier && tar -C /tests -xf -"], {}, None), input=buf.getvalue(), check=True, capture_output=True)
        verifier_timeout = float((self.meta.get("task_spec") or {}).get("verifier_timeout_sec") or 7200)
        (self.dest / "verifier").mkdir(exist_ok=True)
        t_v = time.time()
        with open(self.dest / "verifier" / "test-stdout.txt", "w", encoding="utf-8") as vlog:
            try:
                v = subprocess.run(self.backend.exec_cmd(self.name, ["bash", "-c", "bash /tests/test.sh"], {}, None), check=False, text=True, stdin=subprocess.DEVNULL,
                                   stdout=vlog, stderr=subprocess.STDOUT, timeout=verifier_timeout)
                vcode = v.returncode
            except subprocess.TimeoutExpired:
                log.warning("%s: verifier exceeded %.0f s; stopped", self.unit, verifier_timeout)
                vcode = -9
        reward_txt = subprocess.run(self.backend.exec_cmd(self.name, ["cat", "/logs/verifier/reward.txt"], {}, None), capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL).stdout.strip()
        (self.dest / "verifier" / "reward.txt").write_text(reward_txt + "\n")
        try:
            reward = float(reward_txt)
        except ValueError:
            reward = None
        self.result["verifier"] = {"exit_code": vcode, "seconds": round(time.time() - t_v, 1), "reward": reward, "timed_out": vcode == -9}
        self.result["reward"] = reward
        log.info("%s: verifier exit %s in %.0f s, reward %s", self.unit, vcode, time.time() - t_v, reward)

    # -- the whole flow --------------------------------------------------------------------------

    def run(self) -> dict:
        """The replay of an accepted sandbox: wait until it runs, then the harness, the verifier
        and the compare against the recording."""
        args, backend, dest, result = self.args, self.backend, self.dest, self.result
        adm = result["admission"]
        try:
            created = backend.wait_running(self.name, self.submitted + args.admission_timeout, adm["queued"])
        except Exception as e:  # noqa: BLE001
            if result.get("cutoff"):
                # The run's time was up while this sandbox still waited to start: no error, nothing ran.
                log.info("%s: cut off before it started (%s)", self.unit, e)
                self.backend.remove(self.name)
                if port := result["replay"].get("controller_port"):
                    release_port(BRIDGE_IP, port)
                write_meta(self.dest, result)
                return result
            self.fail(f"{backend.name}: {e}")
            return result
        if RUN_STOPPED.is_set():
            # Dispatched into a run that is over: nothing starts, the sandbox goes.
            result["cutoff"] = True
            log.info("%s: dispatched after the run was stopped: not started", self.unit)
            self.backend.remove(self.name)
            if port := result["replay"].get("controller_port"):
                release_port(BRIDGE_IP, port)
            write_meta(self.dest, result)
            return result
        now = time.time()
        adm.update(running_at=utc(now), wait_sec=round(now - self.submitted, 1), node=backend.node())
        if adm["queued"]:
            log.info("%s: dispatched after %.0f s in the queue", self.unit, now - self.submitted)
        result["sandbox"]["created"] = created
        result["node_before"] = adm["node"]
        started = time.time()
        result["replay"]["started_at"] = utc(started)
        write_status(dest, result, "running")
        try:
            exit_code = self.run_llm()
            finished = time.time()
            result["replay"].update(finished_at=utc(finished), seconds=round(finished - started, 1), exit_code=exit_code)
            if args.verify:
                self.verify()
        except Exception as e:  # noqa: BLE001
            if result.get("cutoff"):
                # The cutoff stopped it while it was starting or running (a unit dispatched from the
                # queue in the cutoff's last second never gets its controller up): what it recorded
                # counts, and it is no error of the unit's. An error here would deny the sample its
                # .complete marker and its upload.
                log.info("%s: cut off while it ran (%s)", self.unit, e)
            else:
                result["error"] = str(e)
                log.error("%s: replay failed: %s", self.unit, e, exc_info=True)
        if not result.get("cutoff"):
            try:
                result["sandbox"]["final_stats"] = backend.stats(self.name)
                result["sandbox"]["final"] = backend.inspect(self.name)
            except Exception as e:  # noqa: BLE001 - a sandbox removed from outside (a cutoff, an operator) is not this unit's error
                log.warning("%s: final stats unavailable: %s", self.unit, e)
        backend.remove(self.name)
        result["container"] = {"name": self.name, "image": self.image, "workdir": self.workdir,
                               "limits": backend.limits_of(result["sandbox"].get("created") or {}, result["sandbox"]["requested"])}
        tape_out = dest / "tape.json"
        if tape_out.exists():
            with _compare_lock:  # compare reads both tapes step by step; a few at a time
                cmp_txt = subprocess.run([str(TIDEBOX_CLI), "compare", str(self.tape), str(tape_out)], check=False, capture_output=True, text=True)
                cmp_json = subprocess.run([str(TIDEBOX_CLI), "compare", "--json", str(self.tape), str(tape_out)], check=False, capture_output=True, text=True)
            (dest / "compare.txt").write_text(cmp_txt.stdout + cmp_txt.stderr)
            try:
                report = json.loads(cmp_json.stdout)
                (dest / "compare.json").write_text(cmp_json.stdout)
                result["compare"] = {k: v for k, v in report.items() if not isinstance(v, (list, dict))}
                result["tape_summary"] = {"steps": report.get("replayed_steps"), "recorded_steps": report.get("recorded_steps")}
            except ValueError:
                result["compare"] = {"error": cmp_json.stderr.strip()[-500:]}
            result["compare"]["exit_code"] = cmp_txt.returncode
        telemetry = dest / "tape.telemetry.csv"
        result["telemetry_rows"] = 0
        if telemetry.exists():
            with open(telemetry, encoding="utf-8") as f:
                result["telemetry_rows"] = max(sum(1 for _ in f) - 1, 0)
        write_meta(dest, result)
        log.info("%s: done in %s s, replay exit %s, compare %s, %d telemetry rows", self.unit, result["replay"].get("seconds"),
                 result["replay"].get("exit_code"), (result.get("compare") or {}).get("complete"), result["telemetry_rows"])
        return result


def sandbox_name(short: str | None, bench: str, task: str, copy: int | None) -> str:
    """`<short>-rep-<task>[-r<k>]`, within tideletd's name rules (63 characters, [a-zA-Z0-9_.-])."""
    short = short or "".join(c for c in bench if c.isalnum())[:8]
    name = f"{short}-rep-{task}" + ("" if copy is None else f"-r{copy}")
    name = "".join(c if c.isalnum() or c in "_.-" else "-" for c in name)
    if len(name) > 63:
        raise ValueError(f"sandbox name {name} is longer than 63 characters")
    return name


def image_of(meta: dict, task: str, image_prefix: str | None) -> str:
    """The task's environment image: `<prefix>/<task>:env` when --image-prefix is given, else the
    image the recording's container ran. containerd wants the registry in the name (docker.io/ for
    a bare name)."""
    if image_prefix:
        image = f"{image_prefix}/{task}:env"
    else:
        image = ((meta.get("container") or {}).get("image")) or ((meta.get("task_spec") or {}).get("image")) or f"tb4/{task}:env"
    first = image.split("/", 1)[0]
    return image if "/" in image and ("." in first or ":" in first or first == "localhost") else f"docker.io/{image}"


def recorded_tasks(data: Path, single_container: bool = False) -> list[str]:
    """Tasks with a complete recording: a tape and a meta.json. With `single_container`, only
    those whose task declares no service containers (sidecars): a sandbox is one container here."""
    tasks = sorted(p.parent.name for p in data.glob("*/meta.json") if (p.parent / "tape.json").exists())
    if single_container:
        tasks = [t for t in tasks if not (json.loads((data / t / "meta.json").read_text()).get("task_spec") or {}).get("sidecars")]
    return tasks


def already_replayed(out: Path, unit: str) -> bool:
    meta = out / unit / "meta.json"
    if not meta.exists():
        return False
    m = json.loads(meta.read_text())
    if m.get("error"):
        return False
    if (m.get("harness") or {}).get("timed_out"):
        return True  # ran out of time under contention: a result of the policy, kept as is
    # Done when every recorded step was replayed; the harness's exit code may legitimately be
    # non-zero when the recording itself ended in an error (an output-token overrun, say).
    # A replay that stopped early for any other reason is redone on the next run.
    return bool((m.get("compare") or {}).get("complete"))


def chown_back(path: Path) -> None:
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid and path.exists():
        for p in [path, *path.rglob("*")]:
            try:
                os.chown(p, int(uid), int(gid))
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", help="task names (default: --quickest, or every recorded task with --all)")
    ap.add_argument("--data", type=Path, default=None, help="the recordings, data/<bench>/<run> (one directory per task); or give --bench and --run")
    ap.add_argument("--bench", choices=benches.names(), default=None, help="the benchmark (default: inferred from --data)")
    ap.add_argument("--run", default=None, help="with --bench: the recording run under data/<bench>/ (default: the bench's run1)")
    ap.add_argument("--all", action="store_true", help="replay every task that has a recording")
    ap.add_argument("--single-container", action="store_true", help="with --all: only tasks without service containers (sidecars)")
    ap.add_argument("--quickest", type=int, default=5, help="without tasks or --all: replay the N tasks with the shortest recorded agent phase (default 5)")
    ap.add_argument("--policy", default="strict", help="the tideletd policy in use (checked against the daemon); names the output directory")
    ap.add_argument("--name", default=None, help="the run's name under output/replay/<bench>/<run>/ (default: the policy)")
    ap.add_argument("--out", type=Path, default=None, help="the run's directory (default output/replay/<bench>/<run>/<name>)")
    ap.add_argument("--speed", type=float, default=1.0, help="the model's pace: 1 = recorded (default), 0 = no waiting")
    ap.add_argument("--repeat", type=int, default=1, help="offer N copies of every selected task, each its own sandbox and controller (<task>__r<k>); default 1")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help=f"seconds between requests to start a sandbox: one is offered per interval, in order, and a rejected one is offered again at the "
                         f"next (default {DEFAULT_INTERVAL:g}; 0: as fast as the daemon answers, a rejected one again after {REJECT_RETRY:g} s)")
    ap.add_argument("--order", choices=("given", "random"), default="given", help="order of the tasks: as listed/quickest-first, or shuffled (--seed)")
    ap.add_argument("--seed", type=int, default=None, help="--order random: seed (default from the clock, logged)")
    ap.add_argument("--redo", action="store_true", help="replay tasks that already have a complete replay in the output directory")
    ap.add_argument("--proxy", default=None, metavar="URL",
                    help="tideletd's LLM proxy (e.g. http://10.88.0.1:18800); Claude Code calls it and it forwards to each task's "
                         "controller, named by the sandbox's tidelet.io/llm-upstream label, so the policies that schedule on model calls see them")
    ap.add_argument("--admission-timeout", type=float, default=6 * 3600, help="give up on a task whose sandbox is not running this many seconds after its first request")
    ap.add_argument("--deadline", type=float, default=None, help="seconds the harness may run (default: four times the recorded agent time, at least 30 min, plus a minute per step)")
    ap.add_argument("--cutoff", type=float, default=None, help="end the run this many seconds after the first sandbox is offered: no more are offered and the "
                    "ones running are stopped (their meta.json gets \"cutoff\": true, no error); for a run that samples a policy rather than finishes the recording")
    ap.add_argument("--port-base", type=int, default=CONTROLLER_PORT_BASE, help=f"first controller port (default {CONTROLLER_PORT_BASE}); ports are reserved across drivers through lock files")
    ap.add_argument("--image-prefix", default=None, help="images as <prefix>/<task>:env (default: the image the recording's container ran, from its meta.json)")
    ap.add_argument("--disallowed-tools", default=DISALLOWED_TOOLS,
                    help=f"tools Claude Code is told not to offer, so its opening request matches the recording's (default {DISALLOWED_TOOLS}; '' for none)")
    ap.add_argument("--harness-env", action="append", default=[], metavar="KEY=VALUE", help="extra environment for Claude Code (repeatable), on top of the raised API and stream timeouts")
    ap.add_argument("--verify", action="store_true", help="after the agent, run the task's own verifier (tests/test.sh from the bench's task copy) in the sandbox and record its reward")
    return ap


def raise_open_files() -> None:
    """Lifts the open-files limit to the hard limit: at Terminal-Bench x8 the driver holds
    streams for hundreds of sandboxes at once, and sudo resets the soft limit to 1024."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = hard if hard != resource.RLIM_INFINITY else 1048576
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ValueError, OSError) as e:
        log.warning("could not raise the open-files limit: %s", e)


def main(argv: list[str] | None = None) -> int:
    raise_open_files()
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.data is None:
        if not args.bench:
            ap.error("give --data data/<bench>/<run>, or --bench (and --run)")
        bench0 = benches.get(args.bench)
        args.data = bench0.default_data if not args.run else bench0.default_data.with_name(args.run)
    args.data = args.data.resolve()
    if not args.data.is_dir():
        ap.error(f"{args.data} is not a directory")
    bench = benches.get(args.bench) if args.bench else benches.infer(args.data)
    if bench is None:
        log.warning("%s: unknown benchmark; sandbox names use the recordings' bench name and --verify is unavailable", args.data)
    HARNESS["disallowed_tools"] = args.disallowed_tools
    for kv in args.harness_env:
        k, _, v = kv.partition("=")
        HARNESS["env"][k] = v
    global CONTROLLER_PORT_BASE
    CONTROLLER_PORT_BASE = args.port_base
    backend = TideletBackend()
    if backend.needs_root and os.geteuid() != 0:
        os.execvp("sudo", ["sudo", "-E", "env", f"PATH={os.environ.get('PATH', '')}", sys.executable, *sys.argv])
    name = args.name or args.policy
    args.run_name = name  # the run's name as meta.json and run.json record it, whatever --out says
    bench_name = bench.name if bench else (benches.infer_name(args.data) or args.data.parent.name)
    out = args.out.resolve() if args.out else layout.replay_dir(bench_name, args.data.name, name)
    args.out = out
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "replay.log")])
    tasks = args.tasks or (recorded_tasks(args.data, args.single_container) if args.all else quickest(args.data, args.quickest))
    missing = [t for t in tasks if not (args.data / t / "tape.json").exists()]
    if missing:
        ap.error(f"no recording for {', '.join(missing)} under {args.data}")
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    if args.interval < 0:
        ap.error("--interval must not be negative")
    if not args.tasks and not tasks:
        ap.error(f"no recordings under {args.data}")
    units: list[tuple[str, int | None]] = [(t, None) for t in tasks] if args.repeat == 1 else [(t, k) for k in range(args.repeat) for t in tasks]
    if args.order == "random":
        seed = args.seed if args.seed is not None else int(time.time())
        random.Random(seed).shuffle(units)
        log.info("task order: random, seed %d", seed)
    if not args.redo:
        skipped = [u for u in units if already_replayed(out, u[0] if u[1] is None else f"{u[0]}__r{u[1]}")]
        if skipped:
            log.info("already replayed, skipping (--redo to repeat): %s", ", ".join(u[0] if u[1] is None else f"{u[0]}__r{u[1]}" for u in skipped))
        units = [u for u in units if u not in skipped]
    daemon_policy = backend.policy()
    if daemon_policy and daemon_policy != args.policy:
        log.warning("--policy %s names the output directory but tideletd runs %s", args.policy, daemon_policy)
    workers = max(1, len(units))
    log.info("tideletd policy %s; replaying %d sandboxes (%d tasks x %d), one offered every %g s, speed %g, into %s",
             daemon_policy, len(units), len(tasks), args.repeat, args.interval, args.speed, out)
    write_run_json(out, {"bench": bench_name, "data": str(args.data), "policy": args.policy, "name": name, "backend": "tidelet", "mode": "llm",
                         "speed": args.speed, "repeat": args.repeat, "interval": args.interval, "tasks": len(tasks), "sandboxes": len(units), "proxy": args.proxy,
                         "verify": args.verify, "seed": args.seed if args.order == "random" else None,
                         "tidebox_rev": git_rev(TIDEBOX), "tidelet_rev": git_rev(TIDEBOX.parent), "driver_started_at": utc()})
    failures = 0
    lock = threading.Lock()
    slots = threading.Semaphore(workers)  # one per sandbox: the policy, not the driver, decides how many run

    def failed(u: tuple[str, int | None], e: Exception) -> None:
        """A unit that failed outside its own record keeping: an error meta.json so the run's
        results stay complete, and no sandbox left running."""
        task, copy = u
        unit = task if copy is None else f"{task}__r{copy}"
        log.error("%s: replay failed: %s", unit, e, exc_info=True)
        try:
            backend.remove(sandbox_name(bench.short if bench else None, bench_name, task, copy))
        except Exception:  # noqa: BLE001
            pass
        meta_path = out / unit / "meta.json"
        if not meta_path.exists():
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            write_meta(meta_path.parent, {"bench": bench_name, "task": task, "copy": copy, "unit": unit, "policy": args.policy, "mode": "llm",
                                          "backend": "tidelet", "replay_of": str(args.data / task / "tape.json"), "error": str(e)})

    def finish(replay: Replay) -> None:
        """An accepted sandbox's replay, in its own thread."""
        nonlocal failures
        try:
            r = replay.run()
        except Exception as e:  # noqa: BLE001 - one task's failure must not stop the others
            failed((replay.task, replay.copy), e)
            r = {"error": str(e)}
        finally:
            slots.release()
            with lock:
                running.discard(replay)
        if r.get("error") or (r.get("replay") or {}).get("exit_code"):
            with lock:
                failures += 1

    pacer = Pacer(args.interval)
    running: set[Replay] = set()
    offered = 0
    cutoff_timer: threading.Timer | None = None
    first_offer: float | None = None
    cut = threading.Event()

    def cut_off() -> None:
        """The run's time is up: the sandboxes still running are stopped; each records what it got done."""
        cut.set()
        RUN_STOPPED.set()
        with lock:
            rs = list(running)
        log.info("cutoff after %.0f s: stopping the %d sandboxes still running", args.cutoff, len(rs))
        for r in rs:
            r.result["cutoff"] = True
        # Every sandbox of the run lives in the pool cgroup: one write ends them all at once, so the
        # removes below find exited containers. Stopping them one by one through the daemon took 20-50
        # minutes on a CPU-saturated host (each delete waited minutes on containerd), and the sandboxes
        # still alive meanwhile kept the host saturated.
        kill = Path(POOL_CGROUP) / "cgroup.kill"
        try:
            kill.write_text("1")
            log.info("cutoff: killed every process of the pool cgroup %s", POOL_CGROUP)
        except OSError as e:  # not root, or another cgroup layout: the removes do the stopping
            log.info("cutoff: pool cgroup not killed as one (%s); stopping the sandboxes one by one", e)

        def stop(r: Replay) -> None:
            try:
                backend.remove(r.name)
            except Exception:  # noqa: BLE001
                pass

        # In parallel: a stop waits out the grace period of an agent that ignores SIGTERM, and
        # one after another 140 of them took a quarter of an hour.
        with ThreadPoolExecutor(max_workers=32, thread_name_prefix="cutoff") as stoppers:
            list(stoppers.map(stop, rs))

    interrupted = threading.Event()

    def on_signal(signum: int, _frame: object) -> None:
        """SIGTERM or SIGINT: the run ends like a cutoff, so no sandbox is left behind (a driver
        killed with 470 sandboxes up once left their tasks, veths and CNI leases on the host)."""
        if interrupted.is_set():
            return
        interrupted.set()
        RUN_STOPPED.set()
        log.warning("signal %d: stopping the run; the sandboxes still running are stopped and recorded", signum)
        threading.Thread(target=cut_off, name="interrupt", daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)

    try:
        # The workload's arrivals: one request per interval, in order; the policy answers each at
        # once (start, queue, reject), and a rejected sandbox is offered again at the next interval
        # while the ones behind it wait. An accepted sandbox's replay runs in its own thread.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replay") as pool:
            for u in units:
                if interrupted.is_set():
                    log.info("interrupted: %d of %d sandboxes were offered; the rest are not", offered, len(units))
                    break
                if args.cutoff and first_offer is not None and time.time() >= first_offer + args.cutoff:
                    log.info("cutoff: %d of %d sandboxes were offered; the rest are not", offered, len(units))
                    break
                slots.acquire()
                try:
                    replay = Replay(u[0], u[1], args, backend, bench)
                    replay.prepare()
                except Exception as e:  # noqa: BLE001
                    failed(u, e)
                    failures += 1
                    slots.release()
                    continue
                outcome = "rejected"
                while True:
                    if args.cutoff and first_offer is not None and time.time() >= first_offer + args.cutoff:
                        break  # the run's time is up while this one was being refused: it is not offered again
                    pacer.wait()
                    if cut.is_set() or interrupted.is_set():
                        break  # the cutoff fired during the wait: nothing starts after it
                    outcome = replay.submit()
                    if outcome != "rejected":
                        break
                    pacer.rejected()
                if outcome == "rejected":
                    log.info("cutoff: %s was not admitted in time; left out", replay.unit)
                    slots.release()
                    break
                if outcome == "error":
                    failures += 1
                    slots.release()
                    continue
                offered += 1
                if first_offer is None:
                    first_offer = time.time()
                    if args.cutoff:
                        cutoff_timer = threading.Timer(args.cutoff, cut_off)
                        cutoff_timer.daemon = True
                        cutoff_timer.start()
                with lock:
                    running.add(replay)
                if cut.is_set():
                    # Offered in the moment the cutoff fired: it is stopped like the others.
                    replay.result["cutoff"] = True
                    try:
                        backend.remove(replay.name)
                    except Exception:  # noqa: BLE001
                        pass
                pool.submit(finish, replay)
    finally:
        if cutoff_timer:
            cutoff_timer.cancel()
        write_run_json(out, {"driver_finished_at": utc(), "failures": failures, "offered": offered, "cutoff": args.cutoff, "interrupted": interrupted.is_set()})
        chown_back(out)
    log.info("done: %d sandboxes, %d with errors or non-zero exit", len(units), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
