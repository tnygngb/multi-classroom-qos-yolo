"""Local FastAPI router for querying experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    HealthResponse,
    PredictionQueryResponse,
    PredictionRecord,
    SummaryResponse,
)


class ReportApi:
    """Small helper around `outputs/reports` for testable API behavior."""

    def __init__(self, reports_dir: str | Path = "outputs/reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def latest_summary(self) -> SummaryResponse:
        summary_path = self._latest_file("*_summary.json")
        if summary_path is None:
            raise FileNotFoundError("No summary JSON found under reports directory.")

        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            payload = {}

        return SummaryResponse(path=str(summary_path), summary=payload)

    def latest_predictions(self, *, limit: int = 50, stream_id: str | None = None) -> PredictionQueryResponse:
        jsonl_path = self._latest_file("*.jsonl", exclude_suffixes=("_events.jsonl",))
        if jsonl_path is None:
            raise FileNotFoundError("No prediction JSONL found under reports directory.")

        max_rows = max(1, int(limit))
        rows: list[PredictionRecord] = []
        scanned = 0

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue

                scanned += 1
                if stream_id and str(payload.get("stream_id")) != stream_id:
                    continue

                row = PredictionRecord(
                    stream_id=str(payload.get("stream_id", "unknown")),
                    timestamp=float(payload.get("timestamp", 0.0)),
                    latency_ms=float(payload.get("latency_ms", 0.0)),
                    detections=list(payload.get("detections", [])) if isinstance(payload.get("detections"), list) else [],
                    frame_index=payload.get("frame_index"),
                    mode=(str(payload.get("mode")) if payload.get("mode") is not None else None),
                    model_variant=(
                        str(payload.get("model_variant")) if payload.get("model_variant") is not None else None
                    ),
                )
                rows.append(row)

        rows = rows[-max_rows:]
        return PredictionQueryResponse(
            source=str(jsonl_path),
            total_scanned=scanned,
            returned=len(rows),
            records=rows,
        )

    def _latest_file(
        self,
        pattern: str,
        *,
        exclude_suffixes: tuple[str, ...] = (),
    ) -> Path | None:
        candidates = sorted(
            self.reports_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            candidate_name = candidate.name
            if any(candidate_name.endswith(item) for item in exclude_suffixes):
                continue
            if candidate.is_file():
                return candidate
        return None


def create_api_app(*, reports_dir: str | Path = "outputs/reports") -> FastAPI:
    """Create report-query API app."""
    report_api = ReportApi(reports_dir=reports_dir)
    app = FastAPI(title="Multi-Classroom Local API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return report_api.health()

    @app.get("/reports/latest-summary", response_model=SummaryResponse)
    def latest_summary() -> SummaryResponse:
        try:
            return report_api.latest_summary()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/reports/latest-predictions", response_model=PredictionQueryResponse)
    def latest_predictions(
        limit: int = Query(default=50, ge=1, le=500),
        stream_id: str | None = Query(default=None),
    ) -> PredictionQueryResponse:
        try:
            return report_api.latest_predictions(limit=limit, stream_id=stream_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
