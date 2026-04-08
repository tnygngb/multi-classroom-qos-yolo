"""FastAPI application for stage-6 cloud review service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.cloud.analytics import CloudAnalytics
from src.cloud.review_service import CloudReviewService
from src.cloud.sample_bank import SampleBank
from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config


class FrameReviewRequest(BaseModel):
    stream_id: str
    timestamp: float
    image: str
    edge_prediction: dict[str, Any] = Field(default_factory=dict)
    event_type: str | None = None


class FrameReviewResponse(BaseModel):
    refined_prediction: dict[str, Any]
    confidence: float
    action: str


class ClipReviewRequest(BaseModel):
    stream_id: str
    start_ts: float
    end_ts: float
    clip: list[str] = Field(default_factory=list)
    edge_prediction: dict[str, Any] = Field(default_factory=dict)
    event_type: str | None = None


class ClipReviewResponse(BaseModel):
    event_type: str
    confidence: float
    notes: str


class MetricsSummaryResponse(BaseModel):
    ack: bool


def create_app(config_path: str = "configs/cloud/cloud_review.yaml") -> FastAPI:
    """Create cloud review FastAPI app from config path."""
    config = ensure_output_dirs(load_config(config_path))
    logger = configure_logger_from_config(config, logger_name="cloud_service")

    reports_dir = Path(config["paths"]["reports"])
    sample_cfg = config.get("sample_bank", {})
    analytics_cfg = config.get("analytics", {})

    sample_root_dir = sample_cfg.get("root_dir", reports_dir / "cloud_sample_bank")
    analytics_report_path = analytics_cfg.get("report_path", reports_dir / "cloud_analytics_summary.json")

    sample_bank = SampleBank(root_dir=sample_root_dir, logger=logger)
    analytics = CloudAnalytics(report_path=analytics_report_path, logger=logger)
    review_service = CloudReviewService(sample_bank=sample_bank, analytics=analytics, logger=logger)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        review_service.close()

    app = FastAPI(title="Multi-Classroom Cloud Review", version="0.1.0", lifespan=lifespan)
    app.state.logger = logger
    app.state.review_service = review_service
    app.state.config = config

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/review/frame", response_model=FrameReviewResponse)
    def review_frame(payload: FrameReviewRequest) -> FrameReviewResponse:
        try:
            result = review_service.review_frame(payload.model_dump())
            return FrameReviewResponse(**result)
        except Exception as exc:
            logger.exception("Frame review failed.")
            raise HTTPException(status_code=500, detail=f"frame review failed: {exc}") from exc

    @app.post("/review/clip", response_model=ClipReviewResponse)
    def review_clip(payload: ClipReviewRequest) -> ClipReviewResponse:
        try:
            result = review_service.review_clip(payload.model_dump())
            return ClipReviewResponse(**result)
        except Exception as exc:
            logger.exception("Clip review failed.")
            raise HTTPException(status_code=500, detail=f"clip review failed: {exc}") from exc

    @app.post("/metrics/summary", response_model=MetricsSummaryResponse)
    def metrics_summary(payload: dict[str, Any]) -> MetricsSummaryResponse:
        try:
            result = review_service.ingest_metrics_summary(payload)
            return MetricsSummaryResponse(**result)
        except Exception as exc:
            logger.exception("Metrics summary ingestion failed.")
            raise HTTPException(status_code=500, detail=f"metrics summary failed: {exc}") from exc

    logger.info("Cloud app initialized. config=%s", config["_meta"]["active_config_path"])
    return app


def get_app_logger(app: FastAPI) -> logging.Logger:
    """Return app logger with explicit type for call sites."""
    return app.state.logger
