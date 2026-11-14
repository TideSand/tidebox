import csv
import itertools
import shutil
import subprocess
import time

import pytest

from tidebox.telemetry import COLUMNS, DockerStats, Sampler, cpu_percent, memory_bytes, row, telemetry_path


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _stats(total_ns, system_ns, cpus=4, usage=1000, inactive=200, pids=3, read="2026-01-01T00:00:00Z"):
    return {
        "read": read,
        "cpu_stats": {"cpu_usage": {"total_usage": total_ns}, "system_cpu_usage": system_ns, "online_cpus": cpus},
        "memory_stats": {"usage": usage, "limit": 4000, "stats": {"inactive_file": inactive}},
        "pids_stats": {"current": pids},
    }


def test_cpu_and_memory_computation():
    a = _stats(1_000_000_000, 100_000_000_000)
    b = _stats(1_500_000_000, 101_000_000_000)  # 0.5 s of CPU over 1 s of system time on 4 cores
    assert cpu_percent(None, b) is None
    assert cpu_percent(a, b) == 200.0
    assert cpu_percent(b, a) == 0.0  # counters went backwards: no guess
    assert memory_bytes(b) == 800
    assert memory_bytes({"memory_stats": {"usage": 500, "stats": {"total_inactive_file": 100}}}) == 400  # cgroup v1
    r = row(1234, a, b)
    assert dict(zip(COLUMNS, r)) == {"t_ms": 1234, "time": "2026-01-01T00:00:00Z", "cpu_percent": 200.0, "cpu_ms": 1500, "memory_bytes": 800, "memory_limit_bytes": 4000, "swap_bytes": "", "pids": 3}
    b["memory_stats"]["stats"]["swap"] = 4096  # the sampler fills this in from the cgroup; Docker's stats lack it
    assert dict(zip(COLUMNS, row(1234, a, b)))["swap_bytes"] == 4096
    assert telemetry_path("tapes/run.json").as_posix() == "tapes/run.telemetry.csv"


def test_sampler_writes_rows_on_schedule(tmp_path):
    class Fake(DockerStats):
        def __init__(self):
            super().__init__(host="unix:///nonexistent")
            self.n = 0

        def stats(self, container):
            self.n += 1
            return _stats(self.n * 100_000_000, self.n * 1_000_000_000, cpus=1, read=f"t{self.n}")

    clock_t0 = time.monotonic()
    sampler = Sampler("box", tmp_path / "x.telemetry.csv", lambda: int((time.monotonic() - clock_t0) * 1000), interval=0.05, client=Fake())
    sampler.start()
    time.sleep(0.42)
    sampler.stop()
    rows = _rows(tmp_path / "x.telemetry.csv")
    assert 6 <= len(rows) <= 10 and rows[0]["cpu_percent"] == "" and rows[1]["cpu_percent"] == "10.0"
    assert [r["time"] for r in rows[:3]] == ["t2", "t3", "t4"]  # t1 was the start-up probe
    assert int(rows[-1]["t_ms"]) > int(rows[0]["t_ms"]) and sampler.errors == 0


def _docker_ready() -> bool:
    return shutil.which("docker") is not None and subprocess.run(["docker", "image", "inspect", "debian:bookworm-slim"], check=False, capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_ready(), reason="docker with debian:bookworm-slim not available")
def test_sampler_against_docker(tmp_path):
    container = subprocess.run(["docker", "run", "-d", "--rm", "debian:bookworm-slim", "sh", "-c", "while :; do :; done"], check=True, capture_output=True, text=True).stdout.strip()
    try:
        t0 = time.monotonic()
        sampler = Sampler(container, tmp_path / "run.telemetry.csv", lambda: int((time.monotonic() - t0) * 1000)).start()
        time.sleep(1.2)
        sampler.stop()
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
    rows = _rows(tmp_path / "run.telemetry.csv")
    assert len(rows) >= 3 and sampler.errors == 0
    assert all(int(r["memory_bytes"]) > 0 and int(r["pids"]) >= 1 for r in rows)
    busy = [float(r["cpu_percent"]) for r in rows[1:]]
    assert max(busy) > 50, busy  # one core spinning
    gaps = [int(b["t_ms"]) - int(a["t_ms"]) for a, b in itertools.pairwise(rows)]
    assert all(150 <= g <= 600 for g in gaps), gaps
