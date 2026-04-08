from __future__ import annotations

from src.metrics.fairness import compute_fairness, jains_fairness_index
from src.metrics.latency import RollingLatencyTracker, compute_latency_stats
from src.metrics.throughput import ThroughputTracker, compute_throughput_from_counts


def test_compute_latency_stats_and_p95() -> None:
    stats = compute_latency_stats([10, 20, 30, 40, 50])
    assert stats.count == 5
    assert abs(stats.mean_ms - 30.0) < 1e-6
    assert abs(stats.p50_ms - 30.0) < 1e-6
    assert stats.p95_ms >= 45.0


def test_rolling_latency_tracker_window() -> None:
    tracker = RollingLatencyTracker(window_size=3)
    tracker.extend([5, 10, 15, 20])
    stats = tracker.snapshot()
    assert stats.count == 3
    assert abs(stats.mean_ms - 15.0) < 1e-6


def test_throughput_tracker_and_compute() -> None:
    tracker = ThroughputTracker(start_ts=100.0)
    tracker.mark("a")
    tracker.mark("a")
    tracker.mark("b")
    snap = tracker.snapshot(now_ts=102.0)
    assert snap.total_count == 3
    assert abs(snap.total_fps - 1.5) < 1e-6
    assert abs(snap.per_stream_fps["a"] - 1.0) < 1e-6

    direct = compute_throughput_from_counts({"x": 4, "y": 2}, elapsed_sec=2.0)
    assert abs(direct.total_fps - 3.0) < 1e-6


def test_fairness_metrics() -> None:
    balanced = jains_fairness_index([5, 5, 5, 5])
    skewed = jains_fairness_index([10, 0, 0, 0])
    assert balanced > skewed

    stats = compute_fairness([2.0, 2.0, 2.0, 2.0])
    assert abs(stats.jain_index - 1.0) < 1e-6
    assert abs(stats.std) < 1e-6
