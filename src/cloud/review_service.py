"""Cloud review business logic for stage 6."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from src.cloud.analytics import CloudAnalytics
from src.cloud.sample_bank import SampleBank


class CloudReviewService:
    """Provide frame/clip review and persistence hooks."""

    def __init__(
        self,
        *,
        sample_bank: SampleBank,
        analytics: CloudAnalytics,
        logger: logging.Logger | None = None,
    ) -> None:
        self.sample_bank = sample_bank
        self.analytics = analytics
        self.logger = logger or logging.getLogger(__name__)

    def review_frame(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Review a single frame payload and return refined result."""
        edge_prediction = payload.get("edge_prediction", {})
        detections = edge_prediction.get("detections", []) if isinstance(edge_prediction, Mapping) else []
        confidences = [
            float(item.get("confidence", 0.0))
            for item in detections
            if isinstance(item, Mapping)
        ]

        detect_count = len(confidences)
        avg_conf = (sum(confidences) / detect_count) if detect_count else 0.0
        boost = 0.08 if avg_conf < 0.45 else 0.03

        refined_detections: list[dict[str, Any]] = []
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            item = dict(detection)
            item["confidence"] = min(1.0, float(item.get("confidence", 0.0)) + boost)
            refined_detections.append(item)

        if detect_count == 0:
            action = "ignore"
        elif avg_conf < 0.35:
            action = "manual_review"
        elif detect_count >= 20:
            action = "escalate"
        else:
            action = "keep"

        response = {
            "refined_prediction": {
                "stream_id": str(payload.get("stream_id", "unknown")),
                "timestamp": float(payload.get("timestamp", time.time())),
                "detections": refined_detections,
            },
            "confidence": min(1.0, avg_conf + boost),
            "action": action,
        }

        record = self.sample_bank.add_frame_review(request_payload=payload, response_payload=response)
        self.analytics.record_sample_review(record)
        self.analytics.write_report()
        return response

    def review_clip(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Review a short clip payload and return event-level result."""
        clip_frames = payload.get("clip", [])
        clip_count = len(clip_frames) if isinstance(clip_frames, list) else 0
        duration = max(
            float(payload.get("end_ts", 0.0)) - float(payload.get("start_ts", 0.0)),
            0.0,
        )

        event_type = str(payload.get("event_type", "")).strip()
        if not event_type:
            event_type = "dense_crowd_frame" if clip_count >= 8 else "abnormal_activity_suspected"

        confidence = min(0.99, 0.45 + min(float(clip_count), 20.0) * 0.02 + min(duration, 5.0) * 0.03)

        response = {
            "event_type": event_type,
            "confidence": confidence,
            "notes": f"clip_frames={clip_count}; duration_sec={duration:.3f}",
        }

        record = self.sample_bank.add_clip_review(request_payload=payload, response_payload=response)
        self.analytics.record_sample_review(record)
        self.analytics.write_report()
        return response

    def ingest_metrics_summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Ingest edge metrics summary and acknowledge receipt."""
        self.analytics.record_metrics_summary(payload)
        self.analytics.write_report()
        return {"ack": True}

    def close(self) -> None:
        self.sample_bank.close()
