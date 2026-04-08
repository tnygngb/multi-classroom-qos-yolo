from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import yaml

from src.cloud.app import create_app


def _create_temp_root() -> Path:
    root = Path("outputs") / "test_tmp" / f"cloud_service_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _write_cloud_config(root: Path) -> str:
    reports_dir = root / "reports"
    sample_bank_dir = root / "sample_bank"
    analytics_path = reports_dir / "analytics_summary.json"

    config = {
        "output_dir": str(root / "outputs"),
        "cloud_service": {
            "host": "127.0.0.1",
            "port": 18081,
            "request_timeout_sec": 2,
            "max_retry": 1,
        },
        "event_engine": {
            "enabled": True,
            "review_api_base_url": "http://127.0.0.1:18081",
        },
        "sample_bank": {
            "root_dir": str(sample_bank_dir),
        },
        "analytics": {
            "report_path": str(analytics_path),
        },
    }

    path = root / "cloud_test.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def test_cloud_service_routes_and_review_pipeline() -> None:
    temp_root = _create_temp_root()
    config_path = _write_cloud_config(temp_root)
    app = create_app(config_path)

    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths
    assert "/review/frame" in route_paths
    assert "/review/clip" in route_paths
    assert "/metrics/summary" in route_paths

    review_service = app.state.review_service

    image_b64 = base64.b64encode(b"fake-jpeg-binary").decode("utf-8")
    frame_result = review_service.review_frame(
        {
            "stream_id": "room_101",
            "timestamp": 1711111111.123,
            "image": image_b64,
            "event_type": "low_confidence_detection",
            "edge_prediction": {
                "stream_id": "room_101",
                "timestamp": 1711111111.123,
                "detections": [
                    {
                        "label": "head",
                        "confidence": 0.25,
                        "bbox": [1, 2, 3, 4],
                    }
                ],
            },
        }
    )
    assert frame_result["action"] in {"manual_review", "keep", "ignore", "escalate"}

    clip_result = review_service.review_clip(
        {
            "stream_id": "room_101",
            "start_ts": 1711111111.100,
            "end_ts": 1711111112.300,
            "event_type": "abnormal_activity_suspected",
            "clip": [image_b64, image_b64],
            "edge_prediction": {
                "stream_id": "room_101",
                "timestamp": 1711111111.123,
                "detections": [],
            },
        }
    )
    assert clip_result["event_type"] in {"abnormal_activity_suspected", "dense_crowd_frame"}

    metrics_result = review_service.ingest_metrics_summary(
        {
            "node_id": "edge_1",
            "stats": {"total_infer_count": 5},
        }
    )
    assert metrics_result["ack"] is True

    sample_bank_file = temp_root / "sample_bank" / "sample_bank.jsonl"
    assert sample_bank_file.exists()
    lines = [line for line in sample_bank_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2

    analytics_file = temp_root / "reports" / "analytics_summary.json"
    assert analytics_file.exists()
    analytics_payload = json.loads(analytics_file.read_text(encoding="utf-8"))
    assert "event_counts" in analytics_payload
    assert "activity" in analytics_payload

    review_service.close()
