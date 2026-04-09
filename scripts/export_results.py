"""Export benchmark results into unified JSON/CSV tables."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.common.config import ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config
from src.metrics.recorder import ExperimentRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export benchmark results into one summary file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base config YAML.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/reports",
        help="Directory to search benchmark JSON outputs.",
    )
    parser.add_argument(
        "--single-json",
        type=str,
        default=None,
        help="Optional explicit single-stream benchmark JSON path.",
    )
    parser.add_argument(
        "--multi-json",
        type=str,
        default=None,
        help="Optional explicit multi-stream benchmark JSON path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve file paths and log plan.",
    )
    return parser


def _latest_file_by_pattern(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _normalize_single_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []

    rows: list[dict[str, Any]] = []
    candidates = payload.get("records")
    if not isinstance(candidates, list):
        candidates = payload.get("results", [])

    if not isinstance(candidates, list):
        return rows

    for item in candidates:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _normalize_multi_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []

    records = payload.get("records", [])
    if not isinstance(records, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            continue

        metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), Mapping) else {}
        latency = metrics.get("latency_ms", {}) if isinstance(metrics.get("latency_ms"), Mapping) else {}
        fairness = metrics.get("fairness", {}) if isinstance(metrics.get("fairness"), Mapping) else {}
        gpu = item.get("gpu", {}) if isinstance(item.get("gpu"), Mapping) else {}

        row = {
            "strategy": item.get("strategy"),
            "stream_count": item.get("stream_count"),
            "status": item.get("status"),
            "total_fps": metrics.get("total_fps"),
            "avg_latency_ms": latency.get("mean_ms"),
            "p95_latency_ms": latency.get("p95_ms"),
            "effective_detection_score": metrics.get("effective_detection_score"),
            "drop_rate_avg": metrics.get("drop_rate_avg"),
            "buffer_avg_backlog": metrics.get("buffer_avg_backlog"),
            "fairness_jain_index": fairness.get("jain_index"),
            "gpu_utilization_pct": gpu.get("gpu_utilization_pct"),
            "gpu_memory_allocated_mb": gpu.get("gpu_memory_allocated_mb"),
            "summary_json": item.get("summary_json"),
            "results_jsonl": item.get("results_jsonl"),
        }
        rows.append(row)

    return rows


def _resolve_output_path(config: Mapping[str, Any], output_arg: str | None) -> Path:
    reports_dir = Path(str(config["paths"]["reports"]))
    reports_dir.mkdir(parents=True, exist_ok=True)

    if output_arg:
        return Path(resolve_path(output_arg, base_dir=config["_meta"]["project_root"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return reports_dir / f"experiment_summary_{timestamp}.json"


def main() -> int:
    args = build_parser().parse_args()
    config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(config, logger_name="export_results")

    input_dir = Path(resolve_path(args.input_dir, base_dir=config["_meta"]["project_root"]))
    single_path = (
        Path(resolve_path(args.single_json, base_dir=config["_meta"]["project_root"]))
        if args.single_json
        else _latest_file_by_pattern(input_dir, "single_stream_compare_*.json")
    )
    multi_path = (
        Path(resolve_path(args.multi_json, base_dir=config["_meta"]["project_root"]))
        if args.multi_json
        else _latest_file_by_pattern(input_dir, "multi_stream_benchmark_*.json")
    )

    logger.info("Resolved single benchmark JSON: %s", single_path)
    logger.info("Resolved multi benchmark JSON: %s", multi_path)

    if args.dry_run:
        logger.info("Dry-run finished. No files were written.")
        return 0

    single_payload = _read_json(single_path)
    multi_payload = _read_json(multi_path)

    single_rows = _normalize_single_rows(single_payload)
    multi_rows = _normalize_multi_rows(multi_payload)

    output_json = _resolve_output_path(config, args.output)
    output_csv = output_json.with_suffix(".csv")

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "inputs": {
            "single_json": str(single_path) if single_path else None,
            "multi_json": str(multi_path) if multi_path else None,
        },
        "single_stream": single_payload,
        "multi_stream": multi_payload,
        "single_table": single_rows,
        "multi_table": multi_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    recorder = ExperimentRecorder(experiment_name="export_results", output_dir=output_csv.parent)
    recorder.add_metadata("source_single", str(single_path) if single_path else None)
    recorder.add_metadata("source_multi", str(multi_path) if multi_path else None)
    for row in multi_rows:
        recorder.add_record(row)
    recorder.write_csv(output_path=output_csv)

    logger.info("Export finished. summary_json=%s summary_csv=%s", output_json, output_csv)
    logger.info("Row counts: single=%s multi=%s", len(single_rows), len(multi_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



