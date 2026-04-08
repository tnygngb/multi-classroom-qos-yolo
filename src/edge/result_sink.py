"""Persist stage-4 inference outputs to JSONL/CSV."""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.detector.schemas import FramePrediction


class ResultSink:
    """Thread-safe result sink writing JSONL and optional CSV."""

    def __init__(
        self,
        *,
        reports_dir: str | Path,
        run_name: str = "edge_run",
        write_csv: bool = True,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.write_csv = write_csv
        self._lock = threading.Lock()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.reports_dir / f"{self.run_name}_{ts}.jsonl"
        self.csv_path = self.reports_dir / f"{self.run_name}_{ts}.csv"
        self.summary_path = self.reports_dir / f"{self.run_name}_{ts}_summary.json"

        self._jsonl_fp = self.jsonl_path.open("w", encoding="utf-8")
        self._csv_fp = None
        self._csv_writer = None
        if self.write_csv:
            self._csv_fp = self.csv_path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_fp,
                fieldnames=[
                    "stream_id",
                    "timestamp",
                    "frame_index",
                    "latency_ms",
                    "detect_count",
                    "source",
                    "image_width",
                    "image_height",
                ],
            )
            self._csv_writer.writeheader()

    def write_prediction(self, prediction: FramePrediction) -> None:
        payload = prediction.to_dict()
        with self._lock:
            self._jsonl_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._jsonl_fp.flush()

            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    {
                        "stream_id": payload["stream_id"],
                        "timestamp": payload["timestamp"],
                        "frame_index": payload.get("frame_index"),
                        "latency_ms": payload["latency_ms"],
                        "detect_count": len(payload.get("detections", [])),
                        "source": payload.get("source"),
                        "image_width": payload.get("image_width"),
                        "image_height": payload.get("image_height"),
                    }
                )
                self._csv_fp.flush()  # type: ignore[union-attr]

    def write_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

    def close(self) -> None:
        with self._lock:
            if self._jsonl_fp and not self._jsonl_fp.closed:
                self._jsonl_fp.close()
            if self._csv_fp and not self._csv_fp.closed:
                self._csv_fp.close()
