"""Multi-stream benchmark runner for stage-7 experiment metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.common.config import ConfigError, ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config
from src.edge.node import EdgeNode
from src.metrics.fairness import compute_fairness
from src.metrics.gpu_stats import collect_gpu_stats
from src.metrics.latency import compute_latency_stats
from src.metrics.recorder import ExperimentRecorder
from src.metrics.throughput import compute_throughput_from_counts


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass(slots=True)
class StrategyConfig:
    name: str
    detector_config: str
    scheduler_config: str
    enable_qos: bool
    sampling_fps: float
    enable_event_engine: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-7 multi-stream benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/multi_stream.yaml",
        help="Path to multi-stream experiment config YAML.",
    )
    parser.add_argument(
        "--streams-config",
        type=str,
        default="configs/streams/demo_4streams.yaml",
        help="Template streams config used to discover source files.",
    )
    parser.add_argument(
        "--detector-config",
        type=str,
        default="configs/detector/yolov8_head.yaml",
        help="Default detector config path.",
    )
    parser.add_argument(
        "--enhanced-detector-config",
        type=str,
        default="configs/detector/yolov12_p2_head.yaml",
        help="High static strategy detector config path.",
    )
    parser.add_argument(
        "--scheduler-config",
        type=str,
        default="configs/scheduler/qos_policy.yaml",
        help="Scheduler config path for QoS strategy.",
    )
    parser.add_argument(
        "--cloud-config",
        type=str,
        default="configs/cloud/cloud_review.yaml",
        help="Cloud config path (used only when event engine enabled).",
    )
    parser.add_argument(
        "--stream-counts",
        type=str,
        default=None,
        help="Comma-separated stream counts override, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Measurement duration per run.",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=None,
        help="Warmup seconds ignored in final metrics.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Detector worker count.",
    )
    parser.add_argument(
        "--status-interval-sec",
        type=float,
        default=1.0,
        help="Status log interval passed to edge node.",
    )
    parser.add_argument(
        "--dispatch-interval-sec",
        type=float,
        default=0.05,
        help="Dispatch loop interval passed to edge node.",
    )
    parser.add_argument(
        "--max-pending-tasks",
        type=int,
        default=256,
        help="Pending queue threshold in edge node.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size passed to edge node.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional custom output JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate planned benchmark matrix and output metadata.",
    )
    return parser


def _parse_stream_counts(value: str | None, fallback: list[int]) -> list[int]:
    if value is None:
        return fallback
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ConfigError("--stream-counts cannot be empty")
    parsed = []
    for item in items:
        try:
            number = int(item)
        except ValueError as exc:
            raise ConfigError(f"Invalid stream count: {item}") from exc
        if number <= 0:
            raise ConfigError("stream count must be > 0")
        parsed.append(number)
    return sorted(set(parsed))


def _discover_sources(template_streams_config: str, *, project_root: str) -> list[str]:
    config = ensure_output_dirs(load_config(template_streams_config))
    sources: list[str] = []

    for item in config.get("streams", []):
        if not isinstance(item, Mapping):
            continue
        source = str(item.get("source", "")).strip()
        if not source:
            continue
        resolved = resolve_path(source, base_dir=project_root)
        if resolved.exists():
            sources.append(str(resolved))

    demos_dir = resolve_path("data/demos", base_dir=project_root)
    if demos_dir.exists():
        for path in sorted(demos_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                sources.append(str(path.resolve()))

    deduped: list[str] = []
    seen = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        deduped.append(source)
    return deduped


def _build_stream_entries(sources: list[str], stream_count: int) -> list[dict[str, Any]]:
    if not sources:
        raise ConfigError("No available video sources found for multi-stream benchmark.")

    entries: list[dict[str, Any]] = []
    for index in range(stream_count):
        source = sources[index % len(sources)]
        entries.append(
            {
                "stream_id": f"room_{index + 1:03d}",
                "source": source,
            }
        )
    return entries


def _write_streams_config(
    *,
    path: Path,
    stream_entries: list[dict[str, Any]],
    sampling_fps: float,
    buffer_size: int,
) -> None:
    payload = {
        "extends": "configs/base.yaml",
        "streams": stream_entries,
        "sampling_fps": float(sampling_fps),
        "buffer_size": int(buffer_size),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _filter_measurement_window(predictions: list[dict[str, Any]], *, warmup_sec: float) -> list[dict[str, Any]]:
    if not predictions:
        return []

    timestamps = [float(item.get("timestamp", 0.0)) for item in predictions]
    if not timestamps:
        return predictions

    start_ts = min(timestamps) + max(float(warmup_sec), 0.0)
    selected = [item for item in predictions if float(item.get("timestamp", 0.0)) >= start_ts]
    return selected if selected else predictions


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 1e-9:
        return 0.0
    return float(numerator / denominator)


def _compute_backlog_metrics(summary: Mapping[str, Any], *, warmup_sec: float) -> tuple[float, float, dict[str, float]]:
    history = summary.get("status_history", [])
    backlog_samples: list[float] = []
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, Mapping):
                continue
            if float(item.get("elapsed_sec", 0.0)) < max(warmup_sec, 0.0):
                continue
            backlog_samples.append(float(item.get("buffer_size", 0.0)))

    avg_backlog = sum(backlog_samples) / len(backlog_samples) if backlog_samples else 0.0

    stream_state_final = summary.get("stream_state_final", {})
    per_stream_drop_rate: dict[str, float] = {}
    if isinstance(stream_state_final, Mapping):
        for stream_id, payload in stream_state_final.items():
            if not isinstance(payload, Mapping):
                continue
            drop_count = float(payload.get("drop_count", 0.0))
            read_count = float(payload.get("read_frame_count", 0.0))
            per_stream_drop_rate[str(stream_id)] = _safe_div(drop_count, max(read_count, 1.0))

    avg_drop_rate = (
        sum(per_stream_drop_rate.values()) / len(per_stream_drop_rate)
        if per_stream_drop_rate
        else 0.0
    )
    return (float(avg_backlog), float(avg_drop_rate), per_stream_drop_rate)


def _compute_run_metrics(
    *,
    predictions: list[dict[str, Any]],
    summary: Mapping[str, Any],
    expected_stream_ids: list[str],
    duration_sec: float,
    warmup_sec: float,
) -> dict[str, Any]:
    measured = _filter_measurement_window(predictions, warmup_sec=warmup_sec)

    latencies = [float(item.get("latency_ms", 0.0)) for item in measured]
    latency_stats = compute_latency_stats(latencies)

    per_stream_counts = {stream_id: 0 for stream_id in expected_stream_ids}
    per_stream_latencies: dict[str, list[float]] = {stream_id: [] for stream_id in expected_stream_ids}
    detect_counts: list[int] = []

    for item in measured:
        stream_id = str(item.get("stream_id", "unknown"))
        per_stream_counts[stream_id] = int(per_stream_counts.get(stream_id, 0)) + 1
        per_stream_latencies.setdefault(stream_id, []).append(float(item.get("latency_ms", 0.0)))

        detections = item.get("detections", [])
        detect_count = len(detections) if isinstance(detections, list) else 0
        detect_counts.append(int(detect_count))

    elapsed = max(float(duration_sec), 1e-6)
    throughput_stats = compute_throughput_from_counts(per_stream_counts, elapsed_sec=elapsed)

    per_stream_latency_summary: dict[str, dict[str, float | int]] = {}
    for stream_id, values in per_stream_latencies.items():
        per_stream_latency_summary[stream_id] = compute_latency_stats(values).to_dict()

    fairness_stats = compute_fairness(list(throughput_stats.per_stream_fps.values()))

    avg_backlog, avg_drop_rate, per_stream_drop_rate = _compute_backlog_metrics(
        summary,
        warmup_sec=warmup_sec,
    )

    non_empty_frames = sum(1 for count in detect_counts if count > 0)
    avg_detect_count = _safe_div(sum(detect_counts), max(len(detect_counts), 1))
    non_empty_rate = _safe_div(non_empty_frames, max(len(detect_counts), 1))
    effective_detection_score = float(avg_detect_count * non_empty_rate)

    return {
        "total_fps": throughput_stats.total_fps,
        "latency_ms": latency_stats.to_dict(),
        "per_stream_fps": throughput_stats.per_stream_fps,
        "per_stream_latency_ms": per_stream_latency_summary,
        "buffer_avg_backlog": avg_backlog,
        "drop_rate_avg": avg_drop_rate,
        "per_stream_drop_rate": per_stream_drop_rate,
        "fairness": fairness_stats.to_dict(),
        "avg_detect_count": float(avg_detect_count),
        "non_empty_frame_rate": float(non_empty_rate),
        "effective_detection_score": float(effective_detection_score),
    }


def _resolve_strategy_configs(config: Mapping[str, Any], args: argparse.Namespace) -> list[StrategyConfig]:
    experiment_cfg = config.get("experiment", {})
    strategies_cfg = experiment_cfg.get("strategies", {})

    defaults = {
        "static_high": {
            "detector_config": args.enhanced_detector_config,
            "scheduler_config": args.scheduler_config,
            "enable_qos": False,
            "sampling_fps": 8.0,
            "enable_event_engine": False,
        },
        "static_low": {
            "detector_config": args.detector_config,
            "scheduler_config": args.scheduler_config,
            "enable_qos": False,
            "sampling_fps": 3.0,
            "enable_event_engine": False,
        },
        "qos": {
            "detector_config": args.detector_config,
            "scheduler_config": args.scheduler_config,
            "enable_qos": True,
            "sampling_fps": 5.0,
            "enable_event_engine": False,
        },
    }

    resolved: list[StrategyConfig] = []
    for name in ("static_high", "static_low", "qos"):
        base = dict(defaults[name])
        override = strategies_cfg.get(name, {}) if isinstance(strategies_cfg, Mapping) else {}
        if isinstance(override, Mapping):
            base.update(dict(override))

        resolved.append(
            StrategyConfig(
                name=name,
                detector_config=str(base["detector_config"]),
                scheduler_config=str(base["scheduler_config"]),
                enable_qos=bool(base["enable_qos"]),
                sampling_fps=float(base["sampling_fps"]),
                enable_event_engine=bool(base.get("enable_event_engine", False)),
            )
        )
    return resolved


def _resolve_output_paths(base_config: Mapping[str, Any], output_json_arg: str | None) -> tuple[Path, Path]:
    reports_dir = Path(str(base_config["paths"]["reports"]))
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = (
        resolve_path(output_json_arg, base_dir=base_config["_meta"]["project_root"])
        if output_json_arg
        else reports_dir / f"multi_stream_benchmark_{timestamp}.json"
    )
    output_csv = output_json.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    return (Path(output_json), Path(output_csv))


def main() -> int:
    args = build_parser().parse_args()
    config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(config, logger_name="benchmark_multi")

    experiment_cfg = config.get("experiment", {})
    stream_counts = _parse_stream_counts(
        args.stream_counts,
        [int(item) for item in experiment_cfg.get("stream_counts", [1, 2, 4, 8])],
    )

    warmup_sec = float(args.warmup_sec if args.warmup_sec is not None else experiment_cfg.get("warmup_seconds", 5.0))
    duration_sec = float(
        args.duration_sec if args.duration_sec is not None else experiment_cfg.get("duration_seconds", 15.0)
    )
    workers = int(args.workers if args.workers is not None else experiment_cfg.get("workers", 1))
    buffer_size = int(experiment_cfg.get("buffer_size", 32))

    sources = _discover_sources(args.streams_config, project_root=config["_meta"]["project_root"])
    if not sources:
        raise ConfigError("No source videos available for benchmark_multi.")

    strategies = _resolve_strategy_configs(config, args)
    output_json, output_csv = _resolve_output_paths(config, args.output_json)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_cfg_dir = Path(config["paths"]["metrics"]) / "generated_stream_configs" / run_tag
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    recorder = ExperimentRecorder(experiment_name="multi_stream_benchmark", output_dir=output_json.parent)
    recorder.add_metadata("config_path", config["_meta"]["active_config_path"])
    recorder.add_metadata("stream_counts", stream_counts)
    recorder.add_metadata("warmup_sec", warmup_sec)
    recorder.add_metadata("duration_sec", duration_sec)
    recorder.add_metadata("workers", workers)
    recorder.add_metadata("strategies", [strategy.name for strategy in strategies])

    logger.info(
        "Benchmark matrix: stream_counts=%s strategies=%s warmup=%s duration=%s",
        stream_counts,
        [strategy.name for strategy in strategies],
        warmup_sec,
        duration_sec,
    )

    for stream_count in stream_counts:
        entries = _build_stream_entries(sources, stream_count)

        for strategy in strategies:
            stream_cfg_path = generated_cfg_dir / f"streams_{strategy.name}_{stream_count}.yaml"
            _write_streams_config(
                path=stream_cfg_path,
                stream_entries=entries,
                sampling_fps=strategy.sampling_fps,
                buffer_size=buffer_size,
            )

            run_meta = {
                "strategy": strategy.name,
                "stream_count": int(stream_count),
                "streams_config_path": str(stream_cfg_path),
                "detector_config_path": strategy.detector_config,
                "scheduler_config_path": strategy.scheduler_config,
                "cloud_config_path": args.cloud_config,
                "enable_qos": strategy.enable_qos,
                "enable_event_engine": strategy.enable_event_engine,
                "sampling_fps": strategy.sampling_fps,
                "warmup_sec": warmup_sec,
                "duration_sec": duration_sec,
                "workers": workers,
            }

            if args.dry_run:
                recorder.add_record({**run_meta, "status": "planned"})
                logger.info("Planned run: %s", run_meta)
                continue

            logger.info("Running benchmark: strategy=%s stream_count=%s", strategy.name, stream_count)
            node = EdgeNode(
                streams_config_path=str(stream_cfg_path),
                detector_config_path=strategy.detector_config,
                scheduler_config_path=strategy.scheduler_config,
                cloud_config_path=args.cloud_config,
                logger=logger,
                num_workers=workers,
                dispatch_interval_sec=args.dispatch_interval_sec,
                status_interval_sec=args.status_interval_sec,
                batch_size=args.batch_size,
                write_csv=True,
                max_pending_tasks=args.max_pending_tasks,
                enable_qos=strategy.enable_qos,
                enable_event_engine=strategy.enable_event_engine,
            )
            summary = node.run(duration_sec=warmup_sec + duration_sec)

            predictions = _load_jsonl(summary.get("results_jsonl", ""))
            run_metrics = _compute_run_metrics(
                predictions=predictions,
                summary=summary,
                expected_stream_ids=[item["stream_id"] for item in entries],
                duration_sec=duration_sec,
                warmup_sec=warmup_sec,
            )

            gpu_stats = collect_gpu_stats(str(config.get("device", "cuda")))
            record = {
                **run_meta,
                "status": "completed",
                "summary_json": summary.get("summary_json"),
                "results_jsonl": summary.get("results_jsonl"),
                "results_csv": summary.get("results_csv"),
                "event_engine_stats": summary.get("event_engine_stats"),
                "metrics": run_metrics,
                "gpu": gpu_stats,
            }
            recorder.add_record(record)

            logger.info(
                "Run done: strategy=%s stream_count=%s total_fps=%.3f p95=%.3f fairness=%.3f",
                strategy.name,
                stream_count,
                run_metrics["total_fps"],
                run_metrics["latency_ms"]["p95_ms"],
                run_metrics["fairness"]["jain_index"],
            )

    if args.dry_run:
        logger.info("Dry-run finished. Planned runs=%s", len(recorder.records))
        return 0

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(recorder.to_payload(), handle, ensure_ascii=False, indent=2)
    recorder.write_csv(output_path=output_csv)

    logger.info("Benchmark finished. JSON=%s CSV=%s", output_json, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
