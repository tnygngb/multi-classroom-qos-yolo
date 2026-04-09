"""Schema definitions for local report query API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health status payload."""

    status: str = "ok"


class SummaryResponse(BaseModel):
    """Latest edge summary payload wrapper."""

    path: str
    summary: dict = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    """One normalized prediction row from JSONL report."""

    stream_id: str
    timestamp: float
    latency_ms: float
    detections: list = Field(default_factory=list)
    frame_index: int | None = None
    mode: str | None = None
    model_variant: str | None = None


class PredictionQueryResponse(BaseModel):
    """Recent prediction records query result."""

    source: str
    total_scanned: int
    returned: int
    records: list[PredictionRecord] = Field(default_factory=list)
