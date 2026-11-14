"""Replaying time.

The recorded response arrived in chunks at recorded offsets after the request. Pacing turns
those into delays for the replay: `--speed 1` reproduces them, `--speed 2` halves them,
`--speed 0` removes them.
"""

from __future__ import annotations

from tidebox.api.tape import Step, StreamChunk


class Pacer:
    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed

    def scale(self, ms: int | None) -> float:
        """Seconds to wait for a recorded interval of `ms` milliseconds."""
        if not ms or ms < 0 or self.speed <= 0:
            return 0.0
        return ms / 1000.0 / self.speed

    def chunk_delays(self, step: Step, chunks: list[StreamChunk]) -> list[tuple[float, str]]:
        """(seconds to wait before sending, chunk data) in order, relative to the previous chunk."""
        out: list[tuple[float, str]] = []
        previous = step.timing.request_at
        for chunk in chunks:
            out.append((self.scale(chunk.at_ms - previous), chunk.data))
            previous = max(previous, chunk.at_ms)
        return out

    def response_delay(self, step: Step) -> float:
        """Seconds the whole response took to arrive, for non-streamed replies."""
        if step.timing.last_byte_at is None:
            return 0.0
        return self.scale(step.timing.last_byte_at - step.timing.request_at)

    def tool_gap(self, step: Step) -> float:
        if step.timing.next_request_at is None or step.timing.last_byte_at is None:
            return 0.0
        return self.scale(step.timing.next_request_at - step.timing.last_byte_at)
