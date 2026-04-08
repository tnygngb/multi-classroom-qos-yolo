"""Fairness metric utilities for multi-stream scheduling analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(slots=True)
class FairnessStats:
    """Fairness summary from per-stream throughput values."""

    count: int
    mean: float
    std: float
    coefficient_of_variation: float
    jain_index: float
    min_value: float
    max_value: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std": float(self.std),
            "coefficient_of_variation": float(self.coefficient_of_variation),
            "jain_index": float(self.jain_index),
            "min": float(self.min_value),
            "max": float(self.max_value),
        }


def jains_fairness_index(values: Iterable[float]) -> float:
    """Compute Jain's fairness index in [0,1]."""
    samples = [max(float(value), 0.0) for value in values]
    if not samples:
        return 0.0
    numerator = sum(samples) ** 2
    denominator = len(samples) * sum(value * value for value in samples)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def compute_fairness(values: Iterable[float]) -> FairnessStats:
    """Compute mean/std/CV/Jain index for stream-level throughput values."""
    samples = [float(value) for value in values]
    if not samples:
        return FairnessStats(
            count=0,
            mean=0.0,
            std=0.0,
            coefficient_of_variation=0.0,
            jain_index=0.0,
            min_value=0.0,
            max_value=0.0,
        )

    count = len(samples)
    mean = sum(samples) / count
    variance = sum((value - mean) ** 2 for value in samples) / count
    std = sqrt(max(variance, 0.0))
    cv = (std / mean) if mean > 1e-9 else 0.0

    return FairnessStats(
        count=count,
        mean=float(mean),
        std=float(std),
        coefficient_of_variation=float(cv),
        jain_index=jains_fairness_index(samples),
        min_value=float(min(samples)),
        max_value=float(max(samples)),
    )
