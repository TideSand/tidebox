"""The controller as a server: LLM replay (--llm), for a real harness pointed at it.

    /v1/..., /r/<run>/...   the provider's API, answered from the tape
    GET /tidebox/state      where every replay is
"""

from __future__ import annotations

import time
from pathlib import Path

from aiohttp import web

from tidebox.api.tape import Tape, TapeMeta
from tidebox.controller.pacing import Pacer
from tidebox.controller.simulator import Simulator
from tidebox.telemetry import Sampler


class ControllerServer:
    def __init__(self, tape: Tape, speed: float = 1.0, output: Path | None = None, host: str = "0.0.0.0", port: int = 8081, sampler: Sampler | None = None) -> None:
        self.tape = tape
        self.host, self.port = host, port
        self.simulator = Simulator(tape, Pacer(speed), output)
        self.sampler = sampler
        self.t0 = time.monotonic()
        self.app = web.Application(client_max_size=256 * 1024 * 1024)
        self.app.router.add_get("/tidebox/state", self.handle_state)
        self.app.add_routes(self.simulator.routes())
        self.app.on_startup.append(self._startup)
        self.app.on_cleanup.append(self._cleanup)

    def now(self) -> int:
        """Milliseconds since the server started: the clock telemetry is stamped with."""
        return int((time.monotonic() - self.t0) * 1000)

    async def _startup(self, app: web.Application) -> None:
        self.t0 = time.monotonic()
        if self.sampler is not None:
            self.sampler.start()

    async def handle_state(self, request: web.Request) -> web.Response:
        return web.json_response({"tape": self.tape.meta.tape_id, "simulator": self.simulator.state()})

    async def _cleanup(self, app: web.Application) -> None:
        # A harness exits the moment it has the last response, and the controller is then told
        # to stop while that response's step may still be being recorded: let in-flight
        # requests finish before the tape is assembled, else the final step is lost.
        import asyncio

        deadline = time.monotonic() + 30
        while self.simulator.in_flight and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if self.sampler is not None:
            self.sampler.stop()
        self.simulator.finalize()

    def run(self) -> None:
        web.run_app(self.app, host=self.host, port=self.port, print=None, access_log=None)


def replay_meta(tape: Tape, mode: str) -> TapeMeta:
    return TapeMeta(
        tape_id=f"{tape.meta.tape_id}-replay",
        harness=tape.meta.harness, provider=tape.meta.provider, model=tape.meta.model,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        harness_version=tape.meta.harness_version, task=dict(tape.meta.task), replay_of=tape.meta.tape_id,
        labels={**tape.meta.labels, "mode": mode},
    )
