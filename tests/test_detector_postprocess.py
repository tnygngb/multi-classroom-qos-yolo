from __future__ import annotations

import numpy as np

from src.detector.postprocess import postprocess_ultralytics_result


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=float)
        self.conf = np.array([0.9, 0.1], dtype=float)
        self.cls = np.array([0, 0], dtype=float)


class FakeResult:
    def __init__(self) -> None:
        self.boxes = FakeBoxes()


def test_postprocess_filters_low_confidence() -> None:
    detections = postprocess_ultralytics_result(
        FakeResult(),
        conf_threshold=0.25,
        class_names={0: "head"},
    )
    assert len(detections) == 1
    assert detections[0].label == "head"
    assert detections[0].confidence == 0.9
    assert detections[0].bbox == [10.0, 20.0, 30.0, 40.0]
