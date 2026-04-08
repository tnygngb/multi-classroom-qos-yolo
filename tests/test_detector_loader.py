from __future__ import annotations

import pytest

from src.common.config import ConfigError, load_config
from src.detector.model_loader import ModelLoader


def test_model_loader_raises_for_missing_explicit_model_path() -> None:
    config = load_config("configs/detector/yolov8_head.yaml")
    config["detector"]["model_path"] = "weights/does_not_exist.pt"
    with pytest.raises(ConfigError):
        _ = ModelLoader(config)
