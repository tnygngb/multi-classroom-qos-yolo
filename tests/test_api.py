from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from src.api.routers import ReportApi, create_api_app


def _temp_reports_dir() -> Path:
    root = Path("outputs") / "test_tmp" / f"api_{uuid.uuid4().hex[:8]}"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports.resolve()


def test_report_api_latest_summary_and_predictions() -> None:
    reports_dir = _temp_reports_dir()

    summary_path = reports_dir / "edge_demo_summary.json"
    summary_payload = {"stats": {"total_infer_count": 3}, "qos_enabled": True}
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    jsonl_path = reports_dir / "edge_demo.jsonl"
    rows = [
        {"stream_id": "room_101", "timestamp": 1.0, "latency_ms": 11.0, "detections": []},
        {"stream_id": "room_102", "timestamp": 2.0, "latency_ms": 12.0, "detections": [{"id": 1}]},
        {"stream_id": "room_101", "timestamp": 3.0, "latency_ms": 13.0, "detections": []},
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    api = ReportApi(reports_dir=reports_dir)

    health = api.health()
    assert health.status == "ok"

    summary = api.latest_summary()
    assert summary.path.endswith("summary.json")
    assert summary.summary["stats"]["total_infer_count"] == 3

    predictions = api.latest_predictions(limit=2)
    assert predictions.returned == 2
    assert predictions.records[-1].stream_id == "room_101"

    room_101 = api.latest_predictions(limit=10, stream_id="room_101")
    assert room_101.returned == 2
    assert all(record.stream_id == "room_101" for record in room_101.records)


def test_create_api_app_registers_routes() -> None:
    reports_dir = _temp_reports_dir()
    # Ensure any report file has older timestamp than route check moment
    (reports_dir / "dummy_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    time.sleep(0.01)

    app = create_api_app(reports_dir=reports_dir)
    route_paths = {route.path for route in app.routes}

    assert "/health" in route_paths
    assert "/reports/latest-summary" in route_paths
    assert "/reports/latest-predictions" in route_paths
