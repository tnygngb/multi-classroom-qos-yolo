"""Post-processing utilities for detector outputs."""

from __future__ import annotations

from typing import Any, Mapping

from src.detector.schemas import Detection


def _normalize_class_names(class_names: Mapping[Any, Any] | None) -> dict[int, str]:
    if not class_names:
        return {}
    normalized: dict[int, str] = {}
    for key, value in class_names.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        normalized[class_id] = str(value)
    return normalized


def _resolve_label(class_id: int, class_names: Mapping[int, str]) -> str:
    return class_names.get(class_id, str(class_id))


def postprocess_ultralytics_result(
    result: Any,
    *,
    conf_threshold: float = 0.25,
    class_names: Mapping[Any, Any] | None = None,
) -> list[Detection]:
    """
    Convert one Ultralytics result object into normalized detections.

    The function intentionally accepts Any to avoid tight coupling with
    Ultralytics runtime types and keep unit testing simple.
    """
    if result is None:
        return []

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    if xyxy is None or conf is None or cls is None:
        return []

    names = _normalize_class_names(class_names)
    detections: list[Detection] = []

    total = len(xyxy)
    for idx in range(total):
        score = float(conf[idx].item() if hasattr(conf[idx], "item") else conf[idx])
        if score < conf_threshold:
            continue

        class_id = int(cls[idx].item() if hasattr(cls[idx], "item") else cls[idx])
        bbox_values = xyxy[idx].tolist() if hasattr(xyxy[idx], "tolist") else list(xyxy[idx])

        detections.append(
            Detection(
                label=_resolve_label(class_id, names),
                confidence=score,
                bbox=[float(v) for v in bbox_values],
            )
        )

    return detections
