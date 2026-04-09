"""Unified stage-8 experiment runner for paper-ready outputs."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.common.config import ConfigError, ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config
from src.detector.predictor import DetectorPredictor
from src.metrics.latency import compute_latency_stats


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


@dataclass(slots=True)
class StepResult:
    """Execution metadata for one stage-8 step."""

    name: str
    status: str
    started_at: float
    finished_at: float
    command: list[str] | None = None
    output_json: str | None = None
    output_csv: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": float(self.started_at),
            "finished_at": float(self.finished_at),
            "duration_sec": float(max(self.finished_at - self.started_at, 0.0)),
            "command": list(self.command) if self.command else None,
            "output_json": self.output_json,
            "output_csv": self.output_csv,
            "notes": self.notes,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-8 experiment bundle and generate paper assets.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/stage8_paper.yaml",
        help="Path to stage-8 experiment config YAML.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("all", "single", "multi", "cross", "ablation", "figures"),
        default="all",
        help="Subset of stage-8 workflow to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate plans/manifests, skip heavy inference runs.",
    )
    return parser


def _discover_demo_sources(project_root: str) -> list[str]:
    demos_dir = resolve_path("data/demos", base_dir=project_root)
    if not demos_dir.exists():
        return []

    sources: list[str] = []
    for path in sorted(demos_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            sources.append(str(path.resolve()))
    return sources


def _run_python_module(module: str, args: list[str], *, logger: logging.Logger, dry_run: bool) -> tuple[int, list[str]]:
    command = [sys.executable, "-m", module, *args]
    logger.info("Run command: %s", " ".join(command))
    if dry_run:
        return (0, command)

    completed = subprocess.run(command, check=False)
    return (int(completed.returncode), command)


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
    return file_path


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return file_path


def _normalize_single_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    records = payload.get("records")
    if not isinstance(records, list):
        records = payload.get("results", [])
    if not isinstance(records, list):
        return []
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _normalize_multi_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []

    records = payload.get("records", [])
    if not isinstance(records, list):
        return []

    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue

        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), Mapping) else {}
        latency = metrics.get("latency_ms", {}) if isinstance(metrics.get("latency_ms"), Mapping) else {}
        fairness = metrics.get("fairness", {}) if isinstance(metrics.get("fairness"), Mapping) else {}
        gpu = record.get("gpu", {}) if isinstance(record.get("gpu"), Mapping) else {}

        rows.append(
            {
                "strategy": record.get("strategy"),
                "stream_count": record.get("stream_count"),
                "status": record.get("status"),
                "total_fps": metrics.get("total_fps"),
                "avg_latency_ms": latency.get("mean_ms"),
                "p95_latency_ms": latency.get("p95_ms"),
                "drop_rate_avg": metrics.get("drop_rate_avg"),
                "gpu_utilization_pct": gpu.get("gpu_utilization_pct"),
                "fairness_jain_index": fairness.get("jain_index"),
                "effective_detection_score": metrics.get("effective_detection_score"),
                "summary_json": record.get("summary_json"),
                "results_jsonl": record.get("results_jsonl"),
                "enable_qos": record.get("enable_qos"),
                "enable_event_engine": record.get("enable_event_engine"),
            }
        )
    return rows


def run_single_experiment(
    *,
    config: Mapping[str, Any],
    output_json: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> StepResult:
    started = time.time()
    single_cfg = config.get("stage8", {}).get("single", {}) if isinstance(config.get("stage8"), Mapping) else {}

    cmd_args = [
        "--config",
        str(single_cfg.get("config", "configs/experiments/single_stream.yaml")),
        "--output-json",
        str(output_json),
    ]
    if single_cfg.get("source"):
        cmd_args.extend(["--source", str(single_cfg["source"])])
    if single_cfg.get("warmup_frames") is not None:
        cmd_args.extend(["--warmup-frames", str(int(single_cfg.get("warmup_frames", 20)))])
    if single_cfg.get("measure_frames") is not None:
        cmd_args.extend(["--measure-frames", str(int(single_cfg.get("measure_frames", 100)))])
    if bool(single_cfg.get("skip_val", True)):
        cmd_args.append("--skip-val")

    code, command = _run_python_module("scripts.benchmark_single", cmd_args, logger=logger, dry_run=dry_run)
    finished = time.time()
    status = "planned" if dry_run else ("completed" if code == 0 else "failed")
    return StepResult(
        name="single_stream_compare",
        status=status,
        started_at=started,
        finished_at=finished,
        command=command,
        output_json=str(output_json),
        output_csv=str(output_json.with_suffix(".csv")),
        notes=None if code == 0 else f"exit_code={code}",
    )


def run_multi_experiment(
    *,
    config: Mapping[str, Any],
    output_json: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> StepResult:
    started = time.time()
    stage8_cfg = config.get("stage8", {}) if isinstance(config.get("stage8"), Mapping) else {}
    multi_cfg = stage8_cfg.get("multi", {}) if isinstance(stage8_cfg.get("multi"), Mapping) else {}

    cmd_args = [
        "--config",
        str(multi_cfg.get("config", "configs/experiments/multi_stream.yaml")),
        "--output-json",
        str(output_json),
    ]
    if multi_cfg.get("stream_counts"):
        joined = ",".join(str(int(item)) for item in list(multi_cfg.get("stream_counts", [])))
        cmd_args.extend(["--stream-counts", joined])
    if multi_cfg.get("warmup_sec") is not None:
        cmd_args.extend(["--warmup-sec", str(float(multi_cfg.get("warmup_sec", 3.0)))])
    if multi_cfg.get("duration_sec") is not None:
        cmd_args.extend(["--duration-sec", str(float(multi_cfg.get("duration_sec", 12.0)))])
    if multi_cfg.get("workers") is not None:
        cmd_args.extend(["--workers", str(int(multi_cfg.get("workers", 1)))])

    code, command = _run_python_module("scripts.benchmark_multi", cmd_args, logger=logger, dry_run=dry_run)
    finished = time.time()
    status = "planned" if dry_run else ("completed" if code == 0 else "failed")
    return StepResult(
        name="multi_stream_compare",
        status=status,
        started_at=started,
        finished_at=finished,
        command=command,
        output_json=str(output_json),
        output_csv=str(output_json.with_suffix(".csv")),
        notes=None if code == 0 else f"exit_code={code}",
    )

def _room_id_from_source(path: str) -> str:
    return Path(path).stem


def _evaluate_cross_room_video(
    predictor: DetectorPredictor,
    *,
    source: str,
    room_id: str,
    max_frames: int,
    low_conf_threshold: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    frame_count = 0
    latencies: list[float] = []
    detection_counts: list[int] = []
    confidences: list[float] = []
    low_conf_detections = 0

    started = time.perf_counter()
    try:
        while frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            prediction = predictor.predict_frame(
                frame,
                stream_id=room_id,
                frame_index=frame_count,
                source=source,
            )
            frame_count += 1
            latencies.append(float(prediction.latency_ms))
            detection_counts.append(len(prediction.detections))

            for detection in prediction.detections:
                conf = float(detection.confidence)
                confidences.append(conf)
                if conf < low_conf_threshold:
                    low_conf_detections += 1
    finally:
        capture.release()

    elapsed = max(time.perf_counter() - started, 1e-6)
    latency_stats = compute_latency_stats(latencies)

    non_empty_frames = sum(1 for count in detection_counts if count > 0)
    total_detections = sum(detection_counts)
    avg_detect_count = total_detections / max(frame_count, 1)
    non_empty_rate = non_empty_frames / max(frame_count, 1)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    false_positive_proxy = low_conf_detections / max(total_detections, 1)
    false_negative_proxy = (frame_count - non_empty_frames) / max(frame_count, 1)

    return {
        "room_id": room_id,
        "source": source,
        "frames": frame_count,
        "fps": frame_count / elapsed,
        "avg_latency_ms": latency_stats.mean_ms,
        "p95_latency_ms": latency_stats.p95_ms,
        "avg_detect_count": avg_detect_count,
        "non_empty_frame_rate": non_empty_rate,
        "avg_confidence": avg_confidence,
        "cross_classroom_map50": None,
        "cross_classroom_map50_proxy": non_empty_rate * avg_confidence,
        "false_positive_proxy_rate": false_positive_proxy,
        "false_negative_proxy_rate": false_negative_proxy,
    }


def _build_cross_classroom_rows(
    *,
    detector_cfg_map: Mapping[str, str],
    sources: list[str],
    max_frames_per_room: int,
    low_conf_threshold: float,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for variant, detector_cfg in detector_cfg_map.items():
        cfg = ensure_output_dirs(load_config(detector_cfg))
        predictor = DetectorPredictor(cfg, logger=logger)

        room_metrics: dict[str, dict[str, Any]] = {}
        for source in sources:
            room_id = _room_id_from_source(source)
            metrics = _evaluate_cross_room_video(
                predictor,
                source=source,
                room_id=room_id,
                max_frames=max_frames_per_room,
                low_conf_threshold=low_conf_threshold,
            )
            room_metrics[room_id] = metrics

        for room_id, holdout in room_metrics.items():
            seen_scores = [
                float(item["cross_classroom_map50_proxy"])
                for key, item in room_metrics.items()
                if key != room_id
            ]
            seen_avg = sum(seen_scores) / len(seen_scores) if seen_scores else float(holdout["cross_classroom_map50_proxy"])
            holdout_score = float(holdout["cross_classroom_map50_proxy"])
            drop_rate = (seen_avg - holdout_score) / max(seen_avg, 1e-9)

            row = {
                "variant": variant,
                "holdout_room": room_id,
                "seen_avg_score": seen_avg,
                "holdout_score": holdout_score,
                "performance_drop_rate": max(drop_rate, -1.0),
                **holdout,
            }
            rows.append(row)

    return rows


def run_cross_classroom_experiment(
    *,
    config: Mapping[str, Any],
    output_json: Path,
    output_csv: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> StepResult:
    started = time.time()
    stage8_cfg = config.get("stage8", {}) if isinstance(config.get("stage8"), Mapping) else {}
    cross_cfg = stage8_cfg.get("cross_classroom", {}) if isinstance(stage8_cfg.get("cross_classroom"), Mapping) else {}

    detector_map = cross_cfg.get(
        "detector_configs",
        {
            "baseline": "configs/detector/yolov8_head.yaml",
            "enhanced": "configs/detector/yolov12_p2_head.yaml",
        },
    )
    if not isinstance(detector_map, Mapping) or not detector_map:
        raise ConfigError("stage8.cross_classroom.detector_configs must be a non-empty mapping")

    sources = cross_cfg.get("video_sources") if isinstance(cross_cfg.get("video_sources"), list) else None
    if not sources:
        sources = _discover_demo_sources(str(config["_meta"]["project_root"]))

    normalized_sources = [
        str(resolve_path(str(source), base_dir=config["_meta"]["project_root"]))
        for source in sources
    ]
    normalized_sources = [source for source in normalized_sources if Path(source).exists()]

    max_frames = int(cross_cfg.get("max_frames_per_room", 36))
    low_conf_threshold = float(cross_cfg.get("low_conf_threshold", 0.35))

    if dry_run:
        rows = [
            {
                "variant": key,
                "holdout_room": Path(source).stem,
                "status": "planned",
                "source": source,
            }
            for key in detector_map.keys()
            for source in normalized_sources
        ]
    else:
        rows = _build_cross_classroom_rows(
            detector_cfg_map={str(key): str(value) for key, value in detector_map.items()},
            sources=normalized_sources,
            max_frames_per_room=max_frames,
            low_conf_threshold=low_conf_threshold,
            logger=logger,
        )

    _write_json(output_json, {"rows": rows, "sources": normalized_sources, "max_frames_per_room": max_frames})
    _write_csv(output_csv, rows)

    finished = time.time()
    return StepResult(
        name="cross_classroom",
        status="planned" if dry_run else "completed",
        started_at=started,
        finished_at=finished,
        output_json=str(output_json),
        output_csv=str(output_csv),
    )


def _compute_p2_ablation_rows(single_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in single_rows if str(row.get("variant")) in {"baseline", "main"}), None)
    enhanced = next((row for row in single_rows if str(row.get("variant")) in {"enhanced", "p2", "light"}), None)
    if baseline is None or enhanced is None:
        return []

    base_fps = float(baseline.get("fps", 0.0) or 0.0)
    enh_fps = float(enhanced.get("fps", 0.0) or 0.0)
    base_map = baseline.get("mAP@0.5")
    enh_map = enhanced.get("mAP@0.5")

    return [
        {
            "ablation": "p2_branch",
            "setting": "without_p2",
            "variant": baseline.get("variant"),
            "fps": base_fps,
            "map50": base_map,
            "delta_fps_vs_without": 0.0,
            "delta_map50_vs_without": 0.0,
        },
        {
            "ablation": "p2_branch",
            "setting": "with_p2",
            "variant": enhanced.get("variant"),
            "fps": enh_fps,
            "map50": enh_map,
            "delta_fps_vs_without": enh_fps - base_fps,
            "delta_map50_vs_without": (
                (float(enh_map) - float(base_map))
                if enh_map is not None and base_map is not None
                else None
            ),
        },
    ]


def _compute_scheduler_ablation_rows(multi_rows: list[dict[str, Any]], target_stream_counts: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream_count in target_stream_counts:
        group = [row for row in multi_rows if int(row.get("stream_count") or 0) == int(stream_count)]
        static_row = next((row for row in group if str(row.get("strategy")) == "static_low"), None)
        qos_row = next((row for row in group if str(row.get("strategy")) == "qos"), None)
        if static_row is None or qos_row is None:
            continue

        static_fps = float(static_row.get("total_fps", 0.0) or 0.0)
        qos_fps = float(qos_row.get("total_fps", 0.0) or 0.0)
        static_p95 = float(static_row.get("p95_latency_ms", 0.0) or 0.0)
        qos_p95 = float(qos_row.get("p95_latency_ms", 0.0) or 0.0)

        rows.append(
            {
                "ablation": "scheduler",
                "stream_count": int(stream_count),
                "setting": "without_scheduler",
                "strategy": "static_low",
                "total_fps": static_fps,
                "p95_latency_ms": static_p95,
                "delta_fps_vs_without": 0.0,
                "delta_p95_vs_without": 0.0,
            }
        )
        rows.append(
            {
                "ablation": "scheduler",
                "stream_count": int(stream_count),
                "setting": "with_scheduler",
                "strategy": "qos",
                "total_fps": qos_fps,
                "p95_latency_ms": qos_p95,
                "delta_fps_vs_without": qos_fps - static_fps,
                "delta_p95_vs_without": qos_p95 - static_p95,
            }
        )
    return rows


def _compute_cloud_ablation_rows(*, dry_run: bool) -> list[dict[str, Any]]:
    if dry_run:
        return [
            {"ablation": "cloud_review", "setting": "disabled", "status": "planned"},
            {"ablation": "cloud_review", "setting": "enabled", "status": "planned"},
        ]

    return [
        {
            "ablation": "cloud_review",
            "setting": "disabled",
            "status": "derived",
            "notes": "Event engine disabled (reference).",
        },
        {
            "ablation": "cloud_review",
            "setting": "enabled",
            "status": "derived",
            "notes": "Event engine enabled; network failures are non-blocking by design.",
        },
    ]


def run_ablation_experiment(
    *,
    config: Mapping[str, Any],
    single_rows: list[dict[str, Any]],
    multi_rows: list[dict[str, Any]],
    output_json: Path,
    output_csv: Path,
    dry_run: bool,
) -> StepResult:
    started = time.time()
    stage8_cfg = config.get("stage8", {}) if isinstance(config.get("stage8"), Mapping) else {}
    ablation_cfg = stage8_cfg.get("ablation", {}) if isinstance(stage8_cfg.get("ablation"), Mapping) else {}

    target_stream_counts = ablation_cfg.get("target_stream_counts", [1, 2, 4, 8])
    if not isinstance(target_stream_counts, list) or not target_stream_counts:
        target_stream_counts = [1, 2, 4, 8]
    target_stream_counts = [int(item) for item in target_stream_counts]

    rows: list[dict[str, Any]] = []
    rows.extend(_compute_p2_ablation_rows(single_rows))
    rows.extend(_compute_scheduler_ablation_rows(multi_rows, target_stream_counts))
    rows.extend(_compute_cloud_ablation_rows(dry_run=dry_run))

    _write_json(output_json, {"rows": rows, "target_stream_counts": target_stream_counts})
    _write_csv(output_csv, rows)

    finished = time.time()
    return StepResult(
        name="ablation",
        status="planned" if dry_run else "completed",
        started_at=started,
        finished_at=finished,
        output_json=str(output_json),
        output_csv=str(output_csv),
    )

def _load_jsonl_records(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []
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


def _collect_qos_mode_distribution(multi_payload: Mapping[str, Any] | None) -> dict[int, dict[str, float]]:
    if multi_payload is None:
        return {}

    records = multi_payload.get("records", [])
    if not isinstance(records, list):
        return {}

    output: dict[int, dict[str, float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if str(record.get("strategy")) != "qos":
            continue

        stream_count = int(record.get("stream_count", 0) or 0)
        jsonl_path = record.get("results_jsonl")
        frames = _load_jsonl_records(str(jsonl_path) if jsonl_path else None)

        mode_counts = {"ECO": 0, "NORMAL": 0, "ALERT": 0}
        for item in frames:
            mode = str(item.get("mode", "NORMAL")).upper()
            mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1

        total = max(sum(mode_counts.values()), 1)
        output[stream_count] = {
            key: float(value / total) for key, value in sorted(mode_counts.items())
        }
    return output


def _plot_mode_allocation(distribution: Mapping[int, Mapping[str, float]], *, output_path: Path) -> None:
    if not distribution:
        return

    stream_counts = sorted(distribution.keys())
    eco = [float(distribution[count].get("ECO", 0.0)) for count in stream_counts]
    normal = [float(distribution[count].get("NORMAL", 0.0)) for count in stream_counts]
    alert = [float(distribution[count].get("ALERT", 0.0)) for count in stream_counts]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.bar(stream_counts, eco, label="ECO", color="#2a9d8f")
    ax.bar(stream_counts, normal, bottom=eco, label="NORMAL", color="#457b9d")
    stacked = [eco_val + normal_val for eco_val, normal_val in zip(eco, normal)]
    ax.bar(stream_counts, alert, bottom=stacked, label="ALERT", color="#e63946")

    ax.set_title("QoS Mode Allocation by Stream Count")
    ax.set_xlabel("Concurrent Streams")
    ax.set_ylabel("Mode Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _draw_simple_diagram(*, title: str, blocks: list[tuple[str, float, float]], edges: list[tuple[int, int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    for text, x, y in blocks:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f1f5f9", "edgecolor": "#334155"},
        )

    for src, dst in edges:
        x1, y1 = blocks[src][1], blocks[src][2]
        x2, y2 = blocks[dst][1], blocks[dst][2]
        arrow = FancyArrowPatch((x1 + 0.7, y1), (x2 - 0.7, y2), arrowstyle="->", mutation_scale=12, lw=1.6)
        ax.add_patch(arrow)

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _render_detection_sample(
    *,
    source: str,
    detector_configs: Mapping[str, str],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source for detection sample: {source}")

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from source: {source}")

    rendered_list: list[np.ndarray] = []

    for variant, cfg_path in detector_configs.items():
        cfg = ensure_output_dirs(load_config(cfg_path))
        predictor = DetectorPredictor(cfg, logger=logger)
        prediction = predictor.predict_frame(frame.copy(), stream_id="sample", frame_index=0, source=source)

        canvas = frame.copy()
        for det in prediction.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 90), 2)
            cv2.putText(
                canvas,
                f"{det.label}:{det.confidence:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (40, 220, 90),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"{variant}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rendered_list.append(canvas)

    stacked = cv2.hconcat(rendered_list) if len(rendered_list) > 1 else rendered_list[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), stacked)


def generate_stage8_figures(
    *,
    config: Mapping[str, Any],
    summary_json: Path,
    multi_payload: Mapping[str, Any] | None,
    figures_dir: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> StepResult:
    started = time.time()

    cmd_args = [
        "--input",
        str(summary_json),
        "--output-dir",
        str(figures_dir),
        "--title-prefix",
        "Stage-8 Paper",
    ]
    code, command = _run_python_module("scripts.visualize_results", cmd_args, logger=logger, dry_run=dry_run)

    mode_allocation_png = figures_dir / "stage8_qos_mode_allocation.png"
    arch_png = figures_dir / "stage8_system_architecture.png"
    detector_png = figures_dir / "stage8_detector_structure.png"
    scheduler_png = figures_dir / "stage8_scheduler_flow.png"
    sample_png = figures_dir / "stage8_detection_sample.png"

    if not dry_run:
        distribution = _collect_qos_mode_distribution(multi_payload)
        _plot_mode_allocation(distribution, output_path=mode_allocation_png)

        _draw_simple_diagram(
            title="System Architecture",
            blocks=[
                ("Video Streams", 1.0, 2.0),
                ("Edge Reader+Buffer", 3.0, 2.0),
                ("Shared Worker Pool", 5.0, 2.0),
                ("QoS Scheduler", 7.0, 2.0),
                ("Cloud Review", 9.0, 2.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
            path=arch_png,
        )
        _draw_simple_diagram(
            title="Detector Structure",
            blocks=[
                ("Input", 1.0, 2.0),
                ("Backbone", 3.0, 2.0),
                ("Neck(FPN)", 5.0, 2.0),
                ("Head(Baseline)", 7.0, 2.0),
                ("P2 Branch(Enhanced)", 9.0, 2.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (2, 4)],
            path=detector_png,
        )
        _draw_simple_diagram(
            title="Scheduler Flow",
            blocks=[
                ("Collect Metrics", 1.0, 2.0),
                ("Score Stream", 3.0, 2.0),
                ("Mode Decision", 5.0, 2.0),
                ("Apply Policy", 7.0, 2.0),
                ("Feedback", 9.0, 2.0),
            ],
            edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
            path=scheduler_png,
        )

        stage8_cfg = config.get("stage8", {}) if isinstance(config.get("stage8"), Mapping) else {}
        cross_cfg = stage8_cfg.get("cross_classroom", {}) if isinstance(stage8_cfg.get("cross_classroom"), Mapping) else {}
        sources = cross_cfg.get("video_sources") if isinstance(cross_cfg.get("video_sources"), list) else None
        if not sources:
            sources = _discover_demo_sources(str(config["_meta"]["project_root"]))

        if sources:
            detector_map = cross_cfg.get(
                "detector_configs",
                {
                    "baseline": "configs/detector/yolov8_head.yaml",
                    "enhanced": "configs/detector/yolov12_p2_head.yaml",
                },
            )
            if isinstance(detector_map, Mapping):
                resolved_map = {str(key): str(value) for key, value in detector_map.items()}
                source = str(resolve_path(str(sources[0]), base_dir=config["_meta"]["project_root"]))
                _render_detection_sample(
                    source=source,
                    detector_configs=resolved_map,
                    output_path=sample_png,
                    logger=logger,
                )

    finished = time.time()
    return StepResult(
        name="figures",
        status="planned" if dry_run else ("completed" if code == 0 else "failed"),
        started_at=started,
        finished_at=finished,
        command=command,
        notes=None if code == 0 else f"exit_code={code}",
    )


def run_export_summary(
    *,
    single_json: Path,
    multi_json: Path,
    output_json: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> StepResult:
    started = time.time()

    cmd_args = [
        "--config",
        "configs/base.yaml",
        "--single-json",
        str(single_json),
        "--multi-json",
        str(multi_json),
        "--output",
        str(output_json),
    ]
    code, command = _run_python_module("scripts.export_results", cmd_args, logger=logger, dry_run=dry_run)

    finished = time.time()
    return StepResult(
        name="export_summary",
        status="planned" if dry_run else ("completed" if code == 0 else "failed"),
        started_at=started,
        finished_at=finished,
        command=command,
        output_json=str(output_json),
        output_csv=str(output_json.with_suffix(".csv")),
        notes=None if code == 0 else f"exit_code={code}",
    )

def run_stage8(config_path: str, *, mode: str, output_dir_arg: str | None, dry_run: bool) -> dict[str, Any]:
    config = ensure_output_dirs(load_config(config_path))
    logger = configure_logger_from_config(config, logger_name="run_experiments_stage8")

    stage8_cfg = config.get("stage8", {}) if isinstance(config.get("stage8"), Mapping) else {}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    default_output_dir = Path(config["paths"]["reports"]) / f"stage8_{run_id}"
    output_dir = (
        Path(resolve_path(output_dir_arg, base_dir=config["_meta"]["project_root"]))
        if output_dir_arg
        else default_output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = output_dir / "tables"
    manifests_dir = output_dir / "manifests"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": 8,
        "run_id": run_id,
        "mode": mode,
        "dry_run": dry_run,
        "config_path": config["_meta"]["active_config_path"],
        "output_dir": str(output_dir),
        "steps": [],
    }

    single_json = tables_dir / "single_stream_compare.json"
    multi_json = tables_dir / "multi_stream_compare.json"
    cross_json = tables_dir / "cross_classroom.json"
    ablation_json = tables_dir / "ablation.json"
    summary_json = tables_dir / "experiment_summary.json"

    step_results: list[StepResult] = []

    run_single = mode in {"all", "single"}
    run_multi = mode in {"all", "multi"}
    run_cross = mode in {"all", "cross"}
    run_ablation = mode in {"all", "ablation"}
    run_figures = mode in {"all", "figures"}

    if run_single and bool(stage8_cfg.get("single", {}).get("enabled", True)):
        step = run_single_experiment(config=config, output_json=single_json, logger=logger, dry_run=dry_run)
        step_results.append(step)
    if run_multi and bool(stage8_cfg.get("multi", {}).get("enabled", True)):
        step = run_multi_experiment(config=config, output_json=multi_json, logger=logger, dry_run=dry_run)
        step_results.append(step)

    single_payload = _read_json(single_json)
    multi_payload = _read_json(multi_json)
    single_rows = _normalize_single_rows(single_payload)
    multi_rows = _normalize_multi_rows(multi_payload)

    if run_cross and bool(stage8_cfg.get("cross_classroom", {}).get("enabled", True)):
        step = run_cross_classroom_experiment(
            config=config,
            output_json=cross_json,
            output_csv=cross_json.with_suffix(".csv"),
            logger=logger,
            dry_run=dry_run,
        )
        step_results.append(step)

    if run_ablation and bool(stage8_cfg.get("ablation", {}).get("enabled", True)):
        step = run_ablation_experiment(
            config=config,
            single_rows=single_rows,
            multi_rows=multi_rows,
            output_json=ablation_json,
            output_csv=ablation_json.with_suffix(".csv"),
            dry_run=dry_run,
        )
        step_results.append(step)

    if run_single or run_multi or mode == "all":
        step = run_export_summary(
            single_json=single_json,
            multi_json=multi_json,
            output_json=summary_json,
            logger=logger,
            dry_run=dry_run,
        )
        step_results.append(step)

    summary_payload = _read_json(summary_json)
    if summary_payload is None and dry_run:
        summary_payload = {
            "generated_at": datetime.now().isoformat(),
            "single_table": single_rows,
            "multi_table": multi_rows,
        }
        _write_json(summary_json, summary_payload)

    if run_figures:
        step = generate_stage8_figures(
            config=config,
            summary_json=summary_json,
            multi_payload=multi_payload,
            figures_dir=figures_dir,
            logger=logger,
            dry_run=dry_run,
        )
        step_results.append(step)

    manifest["steps"] = [step.to_dict() for step in step_results]
    manifest["outputs"] = {
        "single_json": str(single_json),
        "multi_json": str(multi_json),
        "cross_json": str(cross_json),
        "ablation_json": str(ablation_json),
        "summary_json": str(summary_json),
        "figures_dir": str(figures_dir),
    }

    manifest_path = manifests_dir / "stage8_manifest.json"
    _write_json(manifest_path, manifest)

    logger.info("Stage-8 completed. manifest=%s", manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run_stage8(
        args.config,
        mode=args.mode,
        output_dir_arg=args.output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



