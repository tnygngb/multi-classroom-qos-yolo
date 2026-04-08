"""Batching utilities for stage-4 edge inference dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InferenceTask:
    """One sampled frame waiting for inference."""

    stream_id: str
    frame: Any
    timestamp: float
    frame_index: int
    source: str
    enqueue_monotonic: float = field(default_factory=time.perf_counter)


def build_batches(tasks: list[InferenceTask], batch_size: int) -> list[list[InferenceTask]]:
    """
    Build batches from tasks.

    Stage-4 default uses `batch_size=1` (single-frame inference), but this keeps
    batching entrypoints ready for later optimization.
    """
    if batch_size <= 1:
        return [[task] for task in tasks]

    batches: list[list[InferenceTask]] = []
    current: list[InferenceTask] = []
    for task in tasks:
        current.append(task)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches
