"""Sample bank storage for cloud review records."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


class SampleBank:
    """Persist reviewed samples and metadata into JSONL index."""

    def __init__(self, *, root_dir: str | Path, logger: logging.Logger | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.frames_dir = self.root_dir / "frames"
        self.clips_dir = self.root_dir / "clips"
        self.index_path = self.root_dir / "sample_bank.jsonl"

        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_fp = self.index_path.open("a", encoding="utf-8")

    def add_frame_review(
        self,
        *,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record one frame review result and save uploaded key frame."""
        stream_id = str(request_payload.get("stream_id", "unknown"))
        timestamp = float(request_payload.get("timestamp", time.time()))
        image_path = self._save_image_b64(
            b64_data=str(request_payload.get("image", "")),
            output_path=self.frames_dir / f"{stream_id}_{int(timestamp * 1000)}_{uuid.uuid4().hex[:8]}.jpg",
        )

        record = {
            "record_type": "frame_review",
            "stream_id": stream_id,
            "event_type": str(request_payload.get("event_type", "low_confidence_detection")),
            "timestamp": timestamp,
            "reviewed_at": time.time(),
            "frame_path": image_path,
            "clip_path": None,
            "edge_prediction": dict(request_payload.get("edge_prediction", {})),
            "cloud_prediction": dict(response_payload.get("refined_prediction", {})),
            "confidence": float(response_payload.get("confidence", 0.0)),
            "action": str(response_payload.get("action", "")),
        }
        self._append(record)
        return record

    def add_clip_review(
        self,
        *,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record one clip review result and save uploaded clip frames."""
        stream_id = str(request_payload.get("stream_id", "unknown"))
        start_ts = float(request_payload.get("start_ts", time.time()))
        end_ts = float(request_payload.get("end_ts", start_ts))
        clip_frames = request_payload.get("clip", [])
        clip_path = self._save_clip_frames(
            stream_id=stream_id,
            start_ts=start_ts,
            clip_frames=clip_frames if isinstance(clip_frames, list) else [],
        )

        record = {
            "record_type": "clip_review",
            "stream_id": stream_id,
            "event_type": str(response_payload.get("event_type", request_payload.get("event_type", ""))),
            "timestamp": start_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "reviewed_at": time.time(),
            "frame_path": None,
            "clip_path": clip_path,
            "edge_prediction": dict(request_payload.get("edge_prediction", {})),
            "cloud_prediction": {
                "event_type": str(response_payload.get("event_type", "")),
                "notes": str(response_payload.get("notes", "")),
            },
            "confidence": float(response_payload.get("confidence", 0.0)),
            "action": "clip_review",
        }
        self._append(record)
        return record

    def close(self) -> None:
        with self._lock:
            if self._index_fp and not self._index_fp.closed:
                self._index_fp.close()

    def _append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._index_fp.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            self._index_fp.flush()

    def _save_clip_frames(self, *, stream_id: str, start_ts: float, clip_frames: list[Any]) -> str | None:
        if not clip_frames:
            return None

        clip_dir = self.clips_dir / f"{stream_id}_{int(start_ts * 1000)}_{uuid.uuid4().hex[:8]}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for idx, frame_b64 in enumerate(clip_frames):
            output_path = clip_dir / f"frame_{idx:03d}.jpg"
            file_path = self._save_image_b64(b64_data=str(frame_b64), output_path=output_path)
            if file_path:
                saved += 1

        if saved == 0:
            return None
        return str(clip_dir)

    def _save_image_b64(self, *, b64_data: str, output_path: Path) -> str | None:
        if not b64_data:
            return None

        normalized = b64_data
        if b64_data.startswith("data:") and "," in b64_data:
            normalized = b64_data.split(",", 1)[1]

        try:
            raw = base64.b64decode(normalized, validate=False)
        except Exception:
            self.logger.warning("Invalid base64 image payload ignored. path=%s", output_path)
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw)
        return str(output_path)
