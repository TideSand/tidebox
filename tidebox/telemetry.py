"""Resource telemetry: CPU and memory of the sandbox container, sampled every 250 ms through
the Docker Engine API and written as a CSV next to the tape.

    tidebox replay --llm --tape run.json --output replay.json --telemetry --container tidelet:NAME  -> replay.telemetry.csv

Each sample is one `GET /containers/<id>/stats?stream=false&one-shot=true` (a few ms). CPU
percent is computed from the deltas between consecutive samples the way `docker stats` does:
container CPU time / system CPU time * online CPUs * 100, so 100 means one core busy. Memory
is what `docker stats` shows: cgroup usage minus inactive file cache.

CSV columns:
    t_ms            milliseconds since the tape's clock started (the recording's or the replay's)
    time            the sample's wall-clock time, UTC
    cpu_percent     100 = one core busy over the interval since the previous sample
    cpu_ms          cumulative container CPU time, milliseconds
    memory_bytes    memory in use (usage minus inactive file cache)
    memory_limit_bytes
    swap_bytes      swap the container's cgroup holds; empty when it cannot be read
    pids            processes in the container

The Docker endpoint comes from DOCKER_HOST, else the active `docker context`, else
/var/run/docker.sock. Unix sockets and tcp:// hosts are supported; nothing beyond the standard
library is used.

Docker's stats endpoint does not report swap, so on the Docker host it is read from the
container's cgroup: `/containers/<id>/json` gives the init process, `/proc/<pid>/cgroup` its
cgroup, and `memory.swap.current` under /sys/fs/cgroup (cgroup v2; on v1 memsw usage minus
memory usage) the figure. Sampling a remote Docker leaves the column empty.

A Tidelet sandbox is sampled the same way through tideletd's REST gateway
(`GET /v1alpha1/sandboxes/<name>/stats`, TIDELET_API or http://127.0.0.1:7474); its stats are
presented in the Docker shape so the rows and the CSV are identical.
"""

from __future__ import annotations

import csv
import http.client
import json
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("tidebox.telemetry")

COLUMNS = ["t_ms", "time", "cpu_percent", "cpu_ms", "memory_bytes", "memory_limit_bytes", "swap_bytes", "pids"]


def telemetry_path(tape_path: Path | str) -> Path:
    """`run.json` -> `run.telemetry.csv`, next to the tape."""
    p = Path(tape_path)
    return p.with_name(p.stem + ".telemetry.csv")


# -- the Docker Engine API ---------------------------------------------------------------------


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_path)


def docker_host() -> str:
    host = os.environ.get("DOCKER_HOST")
    if host:
        return host
    try:
        out = subprocess.run(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return "unix:///var/run/docker.sock"


class DockerStats:
    """A thin client for the stats endpoint."""

    def __init__(self, host: str | None = None, timeout: float = 5.0) -> None:
        self.host = host or docker_host()
        self.timeout = timeout
        self._swap_readers: dict[str, CgroupSwap] = {}

    def _connection(self) -> http.client.HTTPConnection:
        if self.host.startswith("unix://"):
            return _UnixConnection(self.host[len("unix://"):], self.timeout)
        if self.host.startswith("tcp://") or self.host.startswith("http://"):
            hostport = self.host.split("://", 1)[1].rstrip("/")
            return http.client.HTTPConnection(hostport, timeout=self.timeout)
        raise ValueError(f"unsupported DOCKER_HOST {self.host!r}")

    def get(self, path: str) -> Any:
        conn = self._connection()
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"docker API {path}: HTTP {resp.status}: {body[:200].decode(errors='replace')}")
            return json.loads(body)
        finally:
            conn.close()

    def stats(self, container: str) -> dict[str, Any]:
        s = self.get(f"/containers/{container}/stats?stream=false&one-shot=true")
        swap = self._swap(container)
        if swap is not None:
            s.setdefault("memory_stats", {}).setdefault("stats", {})["swap"] = swap
        return s

    def _swap(self, container: str) -> int | None:
        """Swap held by the container's cgroup, read on the Docker host; None when it cannot be."""
        if not self.host.startswith("unix://"):
            return None  # a remote Docker: no cgroup to read
        reader = self._swap_readers.get(container)
        if reader is None:
            try:
                pid = int((self.get(f"/containers/{container}/json").get("State") or {}).get("Pid") or 0)
            except Exception:  # noqa: BLE001
                return None
            if pid <= 0:
                return None
            reader = CgroupSwap.for_pid(pid)
            if reader is None:
                return None
            self._swap_readers[container] = reader
        swap = reader.read()
        if swap is None:
            self._swap_readers.pop(container, None)  # the container restarted; resolve again next time
        return swap


class CgroupSwap:
    """Reads the swap usage of one cgroup from /sys/fs/cgroup."""

    def __init__(self, current: Path, usage: Path | None = None) -> None:
        self.current = current  # memory.swap.current (v2) or memory.memsw.usage_in_bytes (v1)
        self.usage = usage  # v1 only: memory.usage_in_bytes, subtracted from memsw

    @classmethod
    def for_pid(cls, pid: int, root: Path = Path("/sys/fs/cgroup"), proc: Path = Path("/proc")) -> CgroupSwap | None:
        try:
            lines = (proc / str(pid) / "cgroup").read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            hierarchy, controllers, path = parts
            rel = path.lstrip("/")
            if hierarchy == "0" and controllers == "":  # cgroup v2, unified hierarchy
                current = root / rel / "memory.swap.current"
                if current.exists():
                    return cls(current)
            elif "memory" in controllers.split(","):  # cgroup v1 memory controller
                base = root / "memory" / rel
                if (base / "memory.memsw.usage_in_bytes").exists():
                    return cls(base / "memory.memsw.usage_in_bytes", base / "memory.usage_in_bytes")
        return None

    def read(self) -> int | None:
        try:
            value = int(self.current.read_text(encoding="utf-8").strip())
            if self.usage is not None:
                value = max(value - int(self.usage.read_text(encoding="utf-8").strip()), 0)
            return value
        except (OSError, ValueError):
            return None


class TideletStats:
    """tideletd's stats endpoint, answering in the Docker stats shape."""

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        self.base_url = (base_url or os.environ.get("TIDELET_API") or "http://127.0.0.1:7474").rstrip("/")
        self.timeout = timeout

    def stats(self, sandbox: str) -> dict[str, Any]:
        hostport = self.base_url.split("://", 1)[1]
        conn = http.client.HTTPConnection(hostport, timeout=self.timeout)
        try:
            conn.request("GET", f"/v1alpha1/sandboxes/{sandbox}/stats")
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"tidelet API stats {sandbox}: HTTP {resp.status}: {body[:200].decode(errors='replace')}")
            s = json.loads(body).get("stats") or {}
        finally:
            conn.close()
        usage = int(s.get("memoryUsageBytes") or 0)
        working_set = int(s.get("memoryWorkingSetBytes") or 0)
        return {
            "read": s.get("timestamp", ""),
            "cpu_stats": {
                "cpu_usage": {"total_usage": int(s.get("cpuUsageNanos") or 0)},
                "system_cpu_usage": _host_cpu_nanos(),
                "online_cpus": os.cpu_count() or 1,
            },
            "memory_stats": {"usage": usage, "limit": int(s.get("memoryLimitBytes") or 0),
                             "stats": {"inactive_file": max(usage - working_set, 0), "swap": int(s.get("memorySwapUsageBytes") or 0)}},
            "pids_stats": {"current": int(s.get("pids") or 0)},
        }


def _host_cpu_nanos() -> int:
    """Total CPU time of the host in nanoseconds, as Docker reports system_cpu_usage."""
    with open("/proc/stat", encoding="utf-8") as f:
        fields = f.readline().split()[1:]
    return int(sum(int(x) for x in fields) * (1_000_000_000 // os.sysconf("SC_CLK_TCK")))


# -- samples -----------------------------------------------------------------------------------


def memory_bytes(stats: dict[str, Any]) -> int:
    mem = stats.get("memory_stats") or {}
    usage = int(mem.get("usage") or 0)
    inner = mem.get("stats") or {}
    cache = inner.get("inactive_file", inner.get("total_inactive_file", 0)) or 0  # cgroup v2, then v1
    return max(usage - int(cache), 0)


def swap_bytes(stats: dict[str, Any]) -> int | None:
    """Swap the container holds, or None when the sample does not say (the CSV cell is empty)."""
    inner = (stats.get("memory_stats") or {}).get("stats") or {}
    swap = inner.get("swap", inner.get("total_swap"))
    return int(swap) if swap is not None else None


def cpu_percent(previous: dict[str, Any] | None, current: dict[str, Any]) -> float | None:
    """As `docker stats` computes it, between two consecutive samples; None for the first."""
    if previous is None:
        return None
    cpu, prev = current.get("cpu_stats") or {}, previous.get("cpu_stats") or {}
    cpu_delta = int((cpu.get("cpu_usage") or {}).get("total_usage") or 0) - int((prev.get("cpu_usage") or {}).get("total_usage") or 0)
    system_delta = int(cpu.get("system_cpu_usage") or 0) - int(prev.get("system_cpu_usage") or 0)
    if cpu_delta <= 0 or system_delta <= 0:
        return 0.0
    online = int(cpu.get("online_cpus") or len((cpu.get("cpu_usage") or {}).get("percpu_usage") or []) or 1)
    return round(cpu_delta / system_delta * online * 100, 2)


def row(t_ms: int, previous: dict[str, Any] | None, current: dict[str, Any]) -> list[Any]:
    cpu = current.get("cpu_stats") or {}
    return [
        t_ms,
        current.get("read", ""),
        cpu_percent(previous, current),
        int((cpu.get("cpu_usage") or {}).get("total_usage") or 0) // 1_000_000,
        memory_bytes(current),
        int((current.get("memory_stats") or {}).get("limit") or 0),
        swap if (swap := swap_bytes(current)) is not None else "",
        int((current.get("pids_stats") or {}).get("current") or 0),
    ]


class Sampler:
    """Samples one container on a background thread until stopped."""

    def __init__(self, container: str, path: Path | str, clock: Callable[[], int], interval: float = 0.25, client: DockerStats | None = None) -> None:
        self.container = container
        self.path = Path(path)
        self.clock = clock  # milliseconds on the tape's clock
        self.interval = interval
        self.client = client or DockerStats()
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tidebox-telemetry", daemon=True)

    def start(self) -> Sampler:
        # Fail now, not on the thread, if the container is wrong (an HTTP error); a daemon that is
        # slow to answer is not a wrong container: on a host at its memory limit tideletd took 5-26 s
        # per stats call and two units of a run were lost to this probe's 5 s timeout.
        for attempt in range(12):
            try:
                self.client.stats(self.container)
                break
            except (TimeoutError, OSError) as e:
                if attempt == 11:
                    raise
                time.sleep(5)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        log.info("telemetry: sampling %s every %d ms to %s", self.container, int(self.interval * 1000), self.path)
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval * 4 + self.client.timeout)
        log.info("telemetry: %d samples, %d errors", self.samples, self.errors)

    def _run(self) -> None:
        previous: dict[str, Any] | None = None
        next_at = time.monotonic()
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            f.flush()
            while not self._stop.is_set():
                try:
                    current = self.client.stats(self.container)
                except Exception as e:  # noqa: BLE001 - keep sampling through transient API errors
                    self.errors += 1
                    if self.errors <= 3:
                        log.warning("telemetry: %s", e)
                    current = None
                if current is not None and str(current.get("read", "")).startswith("0001-01-01"):
                    current = None  # the container exists but is not running: Docker answers with zeroed stats
                if current is not None:
                    writer.writerow(row(self.clock(), previous, current))
                    f.flush()
                    previous = current
                    self.samples += 1
                next_at += self.interval
                delay = next_at - time.monotonic()
                if delay < 0:  # fell behind: resynchronize rather than burst
                    next_at = time.monotonic()
                    delay = 0
                if self._stop.wait(delay):
                    break
