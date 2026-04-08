"""Throughput metric utilities for per-stream and aggregate FPS."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class ThroughputStats:
    """Throughput summary for all streams."""

    elapsed_sec: float
    total_count: int
    total_fps: float
    per_stream_count: dict[str, int]
    per_stream_fps: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_sec": float(self.elapsed_sec),
            "total_count": int(self.total_count),
            "total_fps": float(self.total_fps),
            "per_stream_count": {k: int(v) for k, v in self.per_stream_count.items()},
            "per_stream_fps": {k: float(v) for k, v in self.per_stream_fps.items()},
        }


def compute_throughput_from_counts(counts: Mapping[str, int], *, elapsed_sec: float) -> ThroughputStats:
    """Compute per-stream and total FPS from counts over elapsed seconds."""
    elapsed = max(float(elapsed_sec), 1e-6)
    per_stream_count = {str(stream_id): int(count) for stream_id, count in counts.items()}
    total_count = int(sum(per_stream_count.values()))
    per_stream_fps = {
        stream_id: float(count / elapsed) for stream_id, count in per_stream_count.items()
    }
    total_fps = float(total_count / elapsed)
    return ThroughputStats(
        elapsed_sec=elapsed,
        total_count=total_count,
        total_fps=total_fps,
        per_stream_count=per_stream_count,
        per_stream_fps=per_stream_fps,
    )


class ThroughputTracker:
    """Mutable throughput tracker that can be sampled at runtime."""

    def __init__(self, *, start_ts: float | None = None) -> None:
        self.start_ts = float(start_ts if start_ts is not None else time.time())
        self._counts: dict[str, int] = {}

    def mark(self, stream_id: str, count: int = 1) -> None:
        if count <= 0:
            return
        key = str(stream_id)
        self._counts[key] = int(self._counts.get(key, 0)) + int(count)

    def snapshot(self, *, now_ts: float | None = None) -> ThroughputStats:
        now = float(now_ts if now_ts is not None else time.time())
        elapsed = max(now - self.start_ts, 1e-6)
        return compute_throughput_from_counts(self._counts, elapsed_sec=elapsed)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)
