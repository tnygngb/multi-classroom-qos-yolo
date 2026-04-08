"""Cloud-side lightweight analytics for stage 6."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from src.common.io_utils import write_json


class CloudAnalytics:
    """Aggregate event count and activity statistics by stream."""

    def __init__(self, *, report_path: str | Path, logger: logging.Logger | None = None) -> None:
        self.report_path = Path(report_path)
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

        self._event_counts: dict[str, dict[str, int]] = defaultdict(dict)
        self._activity_accumulator: dict[str, dict[str, float]] = defaultdict(dict)
        self._metrics_summary_count: int = 0

    def record_sample_review(self, record: Mapping[str, Any]) -> None:
        """Ingest one sample-bank record into in-memory analytics."""
        stream_id = str(record.get("stream_id", "unknown"))
        event_type = str(record.get("event_type", "unknown"))

        with self._lock:
            event_map = self._event_counts.setdefault(stream_id, {})
            event_map[event_type] = int(event_map.get(event_type, 0)) + 1

            edge_prediction = record.get("edge_prediction", {})
            detections = edge_prediction.get("detections", []) if isinstance(edge_prediction, Mapping) else []
            detect_count = float(len(detections) if isinstance(detections, list) else 0)

            activity = self._activity_accumulator.setdefault(
                stream_id,
                {
                    "frames": 0.0,
                    "total_detect_count": 0.0,
                },
            )
            activity["frames"] = float(activity.get("frames", 0.0)) + 1.0
            activity["total_detect_count"] = float(activity.get("total_detect_count", 0.0)) + detect_count

    def record_metrics_summary(self, summary_payload: Mapping[str, Any]) -> None:
        """Count metrics summary ingestion for observability."""
        _ = summary_payload
        with self._lock:
            self._metrics_summary_count += 1

    def build_report(self) -> dict[str, Any]:
        """Generate current analytics report payload."""
        with self._lock:
            event_counts = {
                stream_id: dict(sorted(event_map.items()))
                for stream_id, event_map in sorted(self._event_counts.items())
            }

            activity_stats: dict[str, dict[str, float]] = {}
            for stream_id, data in sorted(self._activity_accumulator.items()):
                frames = max(float(data.get("frames", 0.0)), 1.0)
                total_detect = float(data.get("total_detect_count", 0.0))
                activity_stats[stream_id] = {
                    "frames": float(data.get("frames", 0.0)),
                    "total_detect_count": total_detect,
                    "avg_detect_count": total_detect / frames,
                }

            summary_count = int(self._metrics_summary_count)

        return {
            "generated_at": time.time(),
            "event_counts": event_counts,
            "activity": activity_stats,
            "metrics_summary_count": summary_count,
        }

    def write_report(self) -> dict[str, Any]:
        """Persist analytics report to JSON file and return payload."""
        report = self.build_report()
        write_json(self.report_path, report)
        return report
