from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts import benchmark_multi, export_results, visualize_results
from src.metrics.recorder import ExperimentRecorder


def _temp_root(tag: str) -> Path:
    root = Path("outputs") / "test_tmp" / f"{tag}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_benchmark_multi_compute_run_metrics() -> None:
    predictions = [
        {"stream_id": "room_001", "timestamp": 100.0, "latency_ms": 10.0, "detections": [{"id": 1}]},
        {"stream_id": "room_002", "timestamp": 100.1, "latency_ms": 20.0, "detections": []},
        {"stream_id": "room_001", "timestamp": 101.0, "latency_ms": 30.0, "detections": [{"id": 2}]},
        {"stream_id": "room_002", "timestamp": 101.2, "latency_ms": 40.0, "detections": [{"id": 3}]},
    ]
    summary = {
        "status_history": [
            {"elapsed_sec": 0.5, "stream_id": "room_001", "buffer_size": 3, "drop_count": 0, "read_frame_count": 10},
            {"elapsed_sec": 1.5, "stream_id": "room_001", "buffer_size": 4, "drop_count": 1, "read_frame_count": 20},
            {"elapsed_sec": 1.6, "stream_id": "room_002", "buffer_size": 2, "drop_count": 2, "read_frame_count": 25},
        ],
        "stream_state_final": {
            "room_001": {"drop_count": 1, "read_frame_count": 20},
            "room_002": {"drop_count": 2, "read_frame_count": 25},
        },
    }

    metrics = benchmark_multi._compute_run_metrics(
        predictions=predictions,
        summary=summary,
        expected_stream_ids=["room_001", "room_002"],
        duration_sec=2.0,
        warmup_sec=0.5,
    )

    assert metrics["total_fps"] > 0
    assert metrics["latency_ms"]["p95_ms"] >= metrics["latency_ms"]["mean_ms"]
    assert metrics["fairness"]["jain_index"] >= 0
    assert "room_001" in metrics["per_stream_fps"]


def test_recorder_and_export_normalization() -> None:
    root = _temp_root("export")
    recorder = ExperimentRecorder(experiment_name="multi_stream_benchmark", output_dir=root)
    recorder.add_record(
        {
            "strategy": "qos",
            "stream_count": 4,
            "status": "completed",
            "metrics": {
                "total_fps": 12.5,
                "latency_ms": {"mean_ms": 42.0, "p95_ms": 70.0},
                "effective_detection_score": 0.8,
                "drop_rate_avg": 0.05,
                "buffer_avg_backlog": 6.0,
                "fairness": {"jain_index": 0.92},
            },
            "gpu": {"gpu_utilization_pct": 45.0, "gpu_memory_allocated_mb": 1024.0},
        }
    )
    multi_path = recorder.write_json(output_path=root / "multi_stream_benchmark_test.json")

    payload = json.loads(multi_path.read_text(encoding="utf-8"))
    rows = export_results._normalize_multi_rows(payload)
    assert len(rows) == 1
    assert rows[0]["strategy"] == "qos"
    assert rows[0]["total_fps"] == 12.5


def test_visualize_grouping_and_plot_output() -> None:
    root = _temp_root("viz")
    rows = [
        {"strategy": "static_low", "stream_count": 1, "status": "completed", "total_fps": 8.0},
        {"strategy": "static_low", "stream_count": 2, "status": "completed", "total_fps": 14.0},
        {"strategy": "qos", "stream_count": 1, "status": "completed", "total_fps": 9.0},
        {"strategy": "qos", "stream_count": 2, "status": "completed", "total_fps": 16.0},
    ]
    grouped = visualize_results._group_metric(rows, "total_fps")
    assert set(grouped.keys()) == {"static_low", "qos"}

    output = root / "throughput.png"
    visualize_results._plot_line_chart(
        grouped=grouped,
        title="Test Throughput",
        xlabel="Streams",
        ylabel="FPS",
        output_path=output,
    )
    assert output.exists()
    assert output.stat().st_size > 0
