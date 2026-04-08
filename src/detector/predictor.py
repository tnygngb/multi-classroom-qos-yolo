"""Detector inference entrypoint wrapper."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.detector.model_loader import ModelLoader
from src.detector.postprocess import postprocess_ultralytics_result
from src.detector.schemas import FramePrediction


class DetectorPredictor:
    """Wrap model inference and return normalized output schema."""

    def __init__(self, config: dict[str, Any], *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.loader = ModelLoader(config, logger=self.logger)
        self.model = self.loader.load()

        detector_cfg = self.config.get("detector", {})
        infer_cfg = detector_cfg.get("infer", {})

        self.conf_threshold = float(
            infer_cfg.get("conf_threshold", detector_cfg.get("confidence_threshold", 0.25))
        )
        self.iou_threshold = float(
            infer_cfg.get("iou_threshold", detector_cfg.get("iou_threshold", 0.45))
        )
        self.imgsz = int(infer_cfg.get("imgsz", 640))
        self.max_det = int(infer_cfg.get("max_det", 300))
        self.model_variant = str(detector_cfg.get("variant", "main"))
        self.class_names = self._resolve_class_names(detector_cfg)

    def _resolve_class_names(self, detector_cfg: dict[str, Any]) -> dict[int, str]:
        class_names_cfg = detector_cfg.get("class_names")
        if isinstance(class_names_cfg, dict):
            normalized: dict[int, str] = {}
            for key, value in class_names_cfg.items():
                try:
                    normalized[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        return self.loader.class_names_from_model()

    def predict_frame(
        self,
        frame: Any,
        *,
        stream_id: str = "single_stream",
        timestamp: float | None = None,
        frame_index: int | None = None,
        source: str | None = None,
        imgsz: int | None = None,
        mode: str | None = None,
        model_variant: str | None = None,
        cloud_review: bool | None = None,
    ) -> FramePrediction:
        """Infer on one frame and return normalized prediction."""
        if frame is None:
            raise ValueError("predict_frame received an empty frame.")

        ts = float(timestamp if timestamp is not None else time.time())
        effective_imgsz = int(imgsz) if imgsz is not None and int(imgsz) > 0 else self.imgsz
        started = time.perf_counter()

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=effective_imgsz,
                device=self.loader.device,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            self.logger.exception("Detector inference failed for stream=%s", stream_id)
            raise RuntimeError(f"Detector inference failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        result = results[0] if results else None
        detections = postprocess_ultralytics_result(
            result,
            conf_threshold=self.conf_threshold,
            class_names=self.class_names,
        )

        height = int(frame.shape[0]) if hasattr(frame, "shape") else None
        width = int(frame.shape[1]) if hasattr(frame, "shape") else None

        return FramePrediction(
            stream_id=stream_id,
            timestamp=ts,
            detections=detections,
            latency_ms=latency_ms,
            frame_index=frame_index,
            source=source,
            image_width=width,
            image_height=height,
            mode=mode,
            model_variant=(model_variant or self.model_variant),
            input_size=effective_imgsz,
            cloud_review=cloud_review,
        )
