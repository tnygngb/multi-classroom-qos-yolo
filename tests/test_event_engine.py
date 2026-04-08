from __future__ import annotations

import time

import numpy as np

from src.detector.schemas import Detection, FramePrediction
from src.edge.event_engine import EventEngine


def _prediction(*, confidence: float, cloud_review: bool = True) -> FramePrediction:
    return FramePrediction(
        stream_id="room_101",
        timestamp=time.time(),
        detections=[Detection(label="head", confidence=confidence, bbox=[0, 0, 10, 10])],
        latency_ms=12.5,
        cloud_review=cloud_review,
        mode="ALERT",
    )


def test_event_engine_triggers_low_confidence_frame_review(monkeypatch) -> None:
    callbacks: list[tuple[dict, dict]] = []

    def fake_post_json(self: EventEngine, endpoint: str, payload: dict) -> dict:  # type: ignore[override]
        assert endpoint == "/review/frame"
        assert payload["stream_id"] == "room_101"
        return {
            "refined_prediction": payload.get("edge_prediction", {}),
            "confidence": 0.81,
            "action": "manual_review",
        }

    monkeypatch.setattr(EventEngine, "_post_json", fake_post_json)
    engine = EventEngine(
        config_path="configs/cloud/cloud_review.yaml",
        on_review_result=lambda event, review: callbacks.append((event, review)),
    )

    try:
        engine.on_prediction(_prediction(confidence=0.2), frame=np.zeros((12, 12, 3), dtype=np.uint8))

        deadline = time.time() + 1.0
        while time.time() < deadline and not callbacks:
            time.sleep(0.01)

        assert callbacks
        event_payload, review_payload = callbacks[0]
        assert event_payload["event"]["event_type"] == "low_confidence_detection"
        assert review_payload["action"] == "manual_review"

        stats = engine.stats()
        assert stats["upload_attempted"] >= 1
        assert stats["upload_success"] >= 1
    finally:
        engine.close()


def test_event_engine_network_failure_does_not_raise(monkeypatch) -> None:
    def fake_post_json_failure(self: EventEngine, endpoint: str, payload: dict) -> dict:  # type: ignore[override]
        _ = endpoint, payload
        raise RuntimeError("simulated network down")

    monkeypatch.setattr(EventEngine, "_post_json", fake_post_json_failure)
    engine = EventEngine(config_path="configs/cloud/cloud_review.yaml")

    try:
        engine.on_prediction(_prediction(confidence=0.2), frame=np.zeros((8, 8, 3), dtype=np.uint8))

        deadline = time.time() + 1.0
        while time.time() < deadline:
            if engine.stats()["upload_failed"] >= 1:
                break
            time.sleep(0.01)

        stats = engine.stats()
        assert stats["upload_attempted"] >= 1
        assert stats["upload_failed"] >= 1
    finally:
        engine.close()
