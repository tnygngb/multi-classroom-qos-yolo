"""Schema definitions for detector input/output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Detection:
    """A normalized single-object detection."""

    label: str
    confidence: float
    bbox: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
        }


@dataclass(slots=True)
class FramePrediction:
    """A normalized prediction payload for one frame."""

    stream_id: str
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    frame_index: int | None = None
    source: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    mode: str | None = None
    model_variant: str | None = None
    input_size: int | None = None
    cloud_review: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stream_id": self.stream_id,
            "timestamp": float(self.timestamp),
            "detections": [det.to_dict() for det in self.detections],
            "latency_ms": float(self.latency_ms),
        }
        if self.frame_index is not None:
            payload["frame_index"] = int(self.frame_index)
        if self.source is not None:
            payload["source"] = self.source
        if self.image_width is not None:
            payload["image_width"] = int(self.image_width)
        if self.image_height is not None:
            payload["image_height"] = int(self.image_height)
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.model_variant is not None:
            payload["model_variant"] = self.model_variant
        if self.input_size is not None:
            payload["input_size"] = int(self.input_size)
        if self.cloud_review is not None:
            payload["cloud_review"] = bool(self.cloud_review)
        return payload


@dataclass(slots=True)
class InferenceSummary:
    """Aggregated metrics after one inference run."""

    source: str
    source_type: str
    total_frames: int
    avg_latency_ms: float
    fps: float
    output_json: str
    output_visual: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "source_type": self.source_type,
            "total_frames": int(self.total_frames),
            "avg_latency_ms": float(self.avg_latency_ms),
            "fps": float(self.fps),
            "output_json": self.output_json,
        }
        if self.output_visual is not None:
            payload["output_visual"] = self.output_visual
        return payload
