"""Stage-6 edge event engine for cloud review upload."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable

import cv2

from src.common.config import ensure_output_dirs, load_config
from src.detector.schemas import FramePrediction


@dataclass(slots=True)
class EventDecision:
    """One event decision derived from a frame prediction."""

    event_id: str
    stream_id: str
    timestamp: float
    event_type: str
    reason: str
    detect_count: int
    max_confidence: float
    avg_confidence: float
    mode: str | None = None


class EventEngine:
    """
    Trigger cloud-review events from edge predictions.

    Network upload runs in background worker threads to keep edge main loop
    non-blocking even when cloud service is down.
    """

    def __init__(
        self,
        *,
        config_path: str = "configs/cloud/cloud_review.yaml",
        logger: logging.Logger | None = None,
        on_review_result: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = ensure_output_dirs(load_config(config_path))
        self.logger = logger or logging.getLogger(__name__)
        self.on_review_result = on_review_result

        cloud_cfg = self.config.get("cloud_service", {})
        event_cfg = self.config.get("event_engine", {})

        self.enabled = bool(event_cfg.get("enabled", True))
        host = str(cloud_cfg.get("host", "127.0.0.1")).strip()
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        port = int(cloud_cfg.get("port", 8081))
        default_base = f"http://{host}:{port}"

        self.review_api_base_url = str(event_cfg.get("review_api_base_url", default_base)).rstrip("/")
        self.request_timeout_sec = float(
            event_cfg.get("request_timeout_sec", cloud_cfg.get("request_timeout_sec", 1.5))
        )

        self.low_conf_threshold = float(event_cfg.get("low_conf_threshold", 0.45))
        self.dense_crowd_threshold = int(event_cfg.get("dense_crowd_threshold", 20))
        self.abnormal_activity_detection_threshold = int(
            event_cfg.get("abnormal_activity_detection_threshold", 14)
        )
        self.min_event_interval_sec = float(event_cfg.get("min_event_interval_sec", 1.0))
        self.clip_history_size = int(event_cfg.get("clip_history_size", 8))
        self.clip_max_frames = int(event_cfg.get("clip_max_frames", 8))
        self.jpeg_quality = int(event_cfg.get("jpeg_quality", 80))

        self.max_workers = int(event_cfg.get("max_workers", 2))
        self.max_pending_uploads = int(event_cfg.get("max_pending_uploads", 32))

        self._executor: ThreadPoolExecutor | None = None
        if self.enabled:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="event_upload")

        self._state_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._inflight: set[Future] = set()
        self._frame_history: dict[str, deque[tuple[float, Any]]] = {}
        self._last_event_ts: dict[str, dict[str, float]] = {}
        self._stats: dict[str, int] = {
            "events_triggered": 0,
            "upload_attempted": 0,
            "upload_success": 0,
            "upload_failed": 0,
            "upload_skipped": 0,
        }

    def on_prediction(self, prediction: FramePrediction, *, frame: Any | None = None) -> None:
        """Evaluate and asynchronously dispatch event uploads for one prediction."""
        if not self.enabled:
            return
        if not bool(prediction.cloud_review):
            return

        if frame is not None:
            self._remember_frame(prediction.stream_id, prediction.timestamp, frame)

        decision = self._build_decision(prediction)
        if decision is None:
            return

        if self._is_rate_limited(decision):
            self._incr("upload_skipped")
            return

        if self.pending_uploads() >= self.max_pending_uploads:
            self._incr("upload_skipped")
            self.logger.warning(
                "Skip event upload due to high pending queue. stream_id=%s pending=%s max_pending=%s",
                prediction.stream_id,
                self.pending_uploads(),
                self.max_pending_uploads,
            )
            return

        self._incr("events_triggered")
        self._incr("upload_attempted")

        assert self._executor is not None
        future = self._executor.submit(self._upload_decision, decision, prediction)
        with self._inflight_lock:
            self._inflight.add(future)
        future.add_done_callback(self._finalize_future)

    def pending_uploads(self) -> int:
        with self._inflight_lock:
            return len(self._inflight)

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            payload = dict(self._stats)
        payload["enabled"] = self.enabled
        payload["pending_uploads"] = self.pending_uploads()
        return payload

    def close(self, *, drain_timeout_sec: float = 2.0) -> None:
        """Stop background uploader and try to drain pending uploads quickly."""
        if self._executor is None:
            return

        deadline = time.time() + max(drain_timeout_sec, 0.1)
        while time.time() < deadline:
            if self.pending_uploads() == 0:
                break
            time.sleep(0.05)

        self._executor.shutdown(wait=False, cancel_futures=False)
        self._executor = None

    def _remember_frame(self, stream_id: str, timestamp: float, frame: Any) -> None:
        with self._state_lock:
            queue_ref = self._frame_history.setdefault(stream_id, deque(maxlen=self.clip_history_size))
            snapshot = frame.copy() if hasattr(frame, "copy") else frame
            queue_ref.append((float(timestamp), snapshot))

    def _build_decision(self, prediction: FramePrediction) -> EventDecision | None:
        detections = prediction.detections
        if not detections:
            return None

        confidences = [float(det.confidence) for det in detections]
        detect_count = len(confidences)
        max_conf = max(confidences) if confidences else 0.0
        avg_conf = (sum(confidences) / detect_count) if detect_count else 0.0

        event_type: str | None = None
        reason: str | None = None

        if detect_count >= self.abnormal_activity_detection_threshold:
            event_type = "abnormal_activity_suspected"
            reason = (
                f"detect_count={detect_count} >= abnormal_activity_detection_threshold="
                f"{self.abnormal_activity_detection_threshold}"
            )
        elif detect_count >= self.dense_crowd_threshold:
            event_type = "dense_crowd_frame"
            reason = f"detect_count={detect_count} >= dense_crowd_threshold={self.dense_crowd_threshold}"
        elif max_conf < self.low_conf_threshold:
            event_type = "low_confidence_detection"
            reason = f"max_confidence={max_conf:.3f} < low_conf_threshold={self.low_conf_threshold:.3f}"

        if event_type is None or reason is None:
            return None

        return EventDecision(
            event_id=str(uuid.uuid4()),
            stream_id=prediction.stream_id,
            timestamp=float(prediction.timestamp),
            event_type=event_type,
            reason=reason,
            detect_count=detect_count,
            max_confidence=max_conf,
            avg_confidence=avg_conf,
            mode=prediction.mode,
        )

    def _is_rate_limited(self, decision: EventDecision) -> bool:
        with self._state_lock:
            stream_map = self._last_event_ts.setdefault(decision.stream_id, {})
            last_ts = stream_map.get(decision.event_type)
            if last_ts is not None and (decision.timestamp - last_ts) < self.min_event_interval_sec:
                return True
            stream_map[decision.event_type] = decision.timestamp
            return False

    def _upload_decision(self, decision: EventDecision, prediction: FramePrediction) -> None:
        request_payload: dict[str, Any]
        endpoint: str

        if decision.event_type == "abnormal_activity_suspected":
            endpoint = "/review/clip"
            request_payload = self._build_clip_payload(decision, prediction)
        else:
            endpoint = "/review/frame"
            request_payload = self._build_frame_payload(decision, prediction)

        try:
            review_response = self._post_json(endpoint, request_payload)
        except Exception as exc:
            self._incr("upload_failed")
            self.logger.warning(
                "Cloud review upload failed. stream_id=%s event_type=%s err=%s",
                decision.stream_id,
                decision.event_type,
                exc,
            )
            return

        self._incr("upload_success")
        if self.on_review_result is not None:
            try:
                self.on_review_result(
                    {
                        "event": asdict(decision),
                        "request": request_payload,
                    },
                    review_response,
                )
            except Exception as exc:  # pragma: no cover - callback behavior is runtime-dependent
                self.logger.warning("Event review callback failed. event_id=%s err=%s", decision.event_id, exc)

    def _build_frame_payload(self, decision: EventDecision, prediction: FramePrediction) -> dict[str, Any]:
        latest_frame = self._latest_frame(decision.stream_id)
        image_b64 = self._encode_frame_to_base64(latest_frame) if latest_frame is not None else ""

        return {
            "stream_id": decision.stream_id,
            "timestamp": decision.timestamp,
            "image": image_b64,
            "edge_prediction": prediction.to_dict(),
            "event_type": decision.event_type,
        }

    def _build_clip_payload(self, decision: EventDecision, prediction: FramePrediction) -> dict[str, Any]:
        frames = self._collect_clip_frames(decision.stream_id)
        if frames:
            start_ts = float(frames[0][0])
            end_ts = float(frames[-1][0])
            encoded = [self._encode_frame_to_base64(item[1]) for item in frames]
        else:
            start_ts = decision.timestamp
            end_ts = decision.timestamp
            encoded = []

        return {
            "stream_id": decision.stream_id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "clip": encoded,
            "edge_prediction": prediction.to_dict(),
            "event_type": decision.event_type,
        }

    def _collect_clip_frames(self, stream_id: str) -> list[tuple[float, Any]]:
        with self._state_lock:
            history = list(self._frame_history.get(stream_id, []))
        if not history:
            return []
        selected = history[-self.clip_max_frames :]
        return selected

    def _latest_frame(self, stream_id: str) -> Any | None:
        with self._state_lock:
            history = self._frame_history.get(stream_id)
            if not history:
                return None
            return history[-1][1]

    def _encode_frame_to_base64(self, frame: Any) -> str:
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(self.jpeg_quality, 100))],
        )
        if not success:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.review_api_base_url}{endpoint}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request failed: {url} ({exc})") from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response from {url}") from exc

    def _incr(self, key: str) -> None:
        with self._state_lock:
            self._stats[key] = int(self._stats.get(key, 0)) + 1

    def _finalize_future(self, future: Future) -> None:
        with self._inflight_lock:
            self._inflight.discard(future)
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - upload handles most failures internally
            self.logger.warning("Unexpected event future failure: %s", exc)
