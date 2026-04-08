"""Latency metric utilities for benchmark and runtime analytics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class LatencyStats:
    """Summary statistics of latency values (milliseconds)."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "mean_ms": float(self.mean_ms),
            "p50_ms": float(self.p50_ms),
            "p95_ms": float(self.p95_ms),
            "min_ms": float(self.min_ms),
            "max_ms": float(self.max_ms),
        }


def percentile(values: list[float], p: float) -> float:
    """Compute percentile with linear interpolation on sorted values."""
    if not values:
        return 0.0
    if p <= 0:
        return float(values[0])
    if p >= 100:
        return float(values[-1])

    pos = (len(values) - 1) * (p / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    if upper == lower:
        return float(values[lower])

    weight = pos - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def compute_latency_stats(latencies_ms: Iterable[float]) -> LatencyStats:
    """Compute count/mean/p50/p95/min/max from latency sequence."""
    values = [float(v) for v in latencies_ms]
    if not values:
        return LatencyStats(count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, min_ms=0.0, max_ms=0.0)

    values.sort()
    count = len(values)
    mean_ms = sum(values) / count
    return LatencyStats(
        count=count,
        mean_ms=float(mean_ms),
        p50_ms=percentile(values, 50.0),
        p95_ms=percentile(values, 95.0),
        min_ms=float(values[0]),
        max_ms=float(values[-1]),
    )


class RollingLatencyTracker:
    """Fixed-window rolling tracker for latency stats."""

    def __init__(self, *, window_size: int = 512) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.window_size = int(window_size)
        self._values: deque[float] = deque(maxlen=self.window_size)

    def add(self, latency_ms: float) -> None:
        self._values.append(float(latency_ms))

    def extend(self, latencies_ms: Iterable[float]) -> None:
        for value in latencies_ms:
            self.add(float(value))

    def snapshot(self) -> LatencyStats:
        return compute_latency_stats(list(self._values))

    def values(self) -> list[float]:
        return list(self._values)
