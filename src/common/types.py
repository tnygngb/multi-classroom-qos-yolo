"""Shared typed structures used across modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StreamFrameMeta:
    """Minimal frame metadata placeholder for later phases."""

    stream_id: str
    timestamp: float
