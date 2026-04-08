"""Unified model loading for detector training and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.common.config import ConfigError, resolve_path


class ModelLoader:
    """Load and cache one detector model instance."""

    def __init__(self, config: dict[str, Any], *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._model = None
        self.device = self._resolve_device()
        self.model_spec = self._resolve_model_spec()

    def _resolve_device(self) -> str:
        detector_cfg = self.config.get("detector", {})
        requested = str(detector_cfg.get("device") or self.config.get("device", "cpu")).lower()
        if requested.startswith("cuda"):
            try:
                import torch
            except ImportError:
                self.logger.warning("Torch not available while checking CUDA. Falling back to cpu.")
                return "cpu"
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but unavailable. Falling back to cpu.")
                return "cpu"
        return requested

    def _resolve_model_spec(self) -> str:
        detector_cfg = self.config.get("detector", {})
        model_path = detector_cfg.get("model_path")
        model_name = detector_cfg.get("model_name", "yolov8n.pt")

        if model_path:
            base_dir = self.config.get("_meta", {}).get("project_root")
            resolved = resolve_path(str(model_path), base_dir=base_dir)
            if not Path(resolved).exists():
                raise ConfigError(
                    f"Configured detector model_path does not exist: {resolved}"
                )
            return str(resolved)

        model_name = str(model_name).strip()
        if not model_name:
            raise ConfigError("detector.model_name must not be empty when model_path is unset.")
        return model_name

    def load(self):  # type: ignore[no-untyped-def]
        """Load model once and reuse it across calls."""
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "ultralytics is not available. Install requirements.txt before running."
            ) from exc

        self.logger.info(
            "Loading detector model. model_spec=%s device=%s",
            self.model_spec,
            self.device,
        )
        self._model = YOLO(self.model_spec)
        return self._model

    def class_names_from_model(self) -> dict[int, str]:
        """Get class name mapping from loaded model metadata."""
        model = self.load()
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}
