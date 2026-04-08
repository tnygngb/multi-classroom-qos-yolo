from __future__ import annotations

from pathlib import Path

from src.detector.schemas import Detection, FramePrediction
from src.edge.result_sink import ResultSink


def test_result_sink_writes_jsonl_and_csv() -> None:
    out_dir = Path("outputs/test_result_sink")
    sink = ResultSink(reports_dir=out_dir, run_name="unit_test", write_csv=True)
    prediction = FramePrediction(
        stream_id="room_101",
        timestamp=1.23,
        detections=[Detection(label="head", confidence=0.9, bbox=[1, 2, 3, 4])],
        latency_ms=11.2,
        frame_index=1,
        source="demo.mp4",
        image_width=640,
        image_height=480,
    )
    sink.write_prediction(prediction)
    sink.write_summary({"ok": True})
    sink.close()

    assert sink.jsonl_path.exists()
    assert sink.csv_path.exists()
    assert sink.summary_path.exists()
