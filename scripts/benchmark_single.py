"""Single-stream baseline/enhanced comparison benchmark."""

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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config import ConfigError, ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config
from src.detector.predictor import DetectorPredictor
from src.metrics.latency import compute_latency_stats
from src.metrics.recorder import ExperimentRecorder
from src.metrics.throughput import ThroughputTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and enhanced single-stream detector performance.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/single_stream.yaml",
        help="Experiment config file for default compare settings.",
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default=None,
        help="Override baseline detector config path.",
    )
    parser.add_argument(
        "--enhanced-config",
        type=str,
        default=None,
        help="Override enhanced detector config path.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional benchmark source (image/video). If omitted, uses synthetic frames.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=None,
        help="Warmup frame count before FPS measurement.",
    )
    parser.add_argument(
        "--measure-frames",
        type=int,
        default=None,
        help="Measured frame count for FPS/latency.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip mAP validation even if dataset exists.",
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
        help="Validate configuration only.",
    )
    return parser


def _load_detector_cfg(path: str) -> dict[str, Any]:
    return ensure_output_dirs(load_config(path))


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_map_metrics(val_result: Any) -> tuple[float | None, float | None]:
    if val_result is None:
        return (None, None)

    box = getattr(val_result, "box", None)
    if box is None:
        return (None, None)

    map50 = _to_float(getattr(box, "map50", None))
    map5095 = _to_float(getattr(box, "map", None))
    return (map50, map5095)


def _count_params(model: Any) -> int | None:
    module = getattr(model, "model", None)
    if module is None or not hasattr(module, "parameters"):
        return None
    try:
        return int(sum(parameter.numel() for parameter in module.parameters()))
    except Exception:
        return None


def _estimate_flops(model: Any, imgsz: int) -> float | None:
    module = getattr(model, "model", None)
    if module is None:
        return None

    try:
        import torch
        from thop import profile
    except Exception:
        return None

    try:
        module = module.to("cpu").eval()
        dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
        flops, _params = profile(module, inputs=(dummy,), verbose=False)
        return float(flops)
    except Exception:
        return None


def _resolve_data_path(config: dict[str, Any]) -> Path | None:
    data_rel = config.get("detector", {}).get("train", {}).get("data")
    if not data_rel:
        return None
    path = resolve_path(str(data_rel), base_dir=config["_meta"]["project_root"])
    return path if path.exists() else None


def _run_map_validation(
    model: Any,
    config: dict[str, Any],
    logger,
    *,
    device: str,
) -> tuple[float | None, float | None]:
    data_path = _resolve_data_path(config)
    if data_path is None:
        logger.warning("Validation skipped: dataset config not found.")
        return (None, None)

    detector_cfg = config.get("detector", {})
    train_cfg = detector_cfg.get("train", {})
    infer_cfg = detector_cfg.get("infer", {})

    val_kwargs = {
        "data": str(data_path),
        "imgsz": int(train_cfg.get("imgsz", infer_cfg.get("imgsz", 640))),
        "batch": int(train_cfg.get("batch", 8)),
        "device": device,
        "split": "val",
        "plots": False,
        "save_json": False,
        "verbose": False,
        "project": str(Path(config["paths"]["runs"])),
        "name": f"val_{detector_cfg.get('variant', 'detector')}",
        "exist_ok": True,
    }

    try:
        result = model.val(**val_kwargs)
    except Exception as exc:
        logger.warning("Validation failed and will be skipped: %s", exc)
        return (None, None)

    return _extract_map_metrics(result)


def _load_source_frames(source_path: str | None, measure_frames: int, imgsz: int) -> list[np.ndarray]:
    if source_path is None:
        frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        return [frame.copy() for _ in range(measure_frames)]

    resolved = Path(source_path)
    if not resolved.exists():
        raise ConfigError(f"Benchmark source does not exist: {resolved}")

    if resolved.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(resolved))
        if frame is None:
            raise RuntimeError(f"Failed to read image source: {resolved}")
        return [frame.copy() for _ in range(measure_frames)]

    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {resolved}")

    frames: list[np.ndarray] = []
    try:
        while len(frames) < measure_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames available from source: {resolved}")

    while len(frames) < measure_frames:
        frames.append(frames[-1].copy())

    return frames


def _benchmark_fps(
    predictor: DetectorPredictor,
    *,
    frames: list[np.ndarray],
    warmup_frames: int,
    stream_id: str,
) -> dict[str, float]:
    warmup_list = frames[: max(1, min(warmup_frames, len(frames)))]
    for frame in warmup_list:
        _ = predictor.predict_frame(frame, stream_id=stream_id)

    latencies_ms: list[float] = []
    detect_counts: list[int] = []
    throughput = ThroughputTracker()
    started = time.perf_counter()
    for idx, frame in enumerate(frames):
        prediction = predictor.predict_frame(
            frame,
            stream_id=stream_id,
            frame_index=idx,
        )
        latencies_ms.append(prediction.latency_ms)
        detect_counts.append(len(prediction.detections))
        throughput.mark(stream_id)
    elapsed = time.perf_counter() - started

    latency_stats = compute_latency_stats(latencies_ms)
    fps = throughput.snapshot(now_ts=throughput.start_ts + elapsed).total_fps
    non_empty = sum(1 for count in detect_counts if count > 0)
    avg_detect_count = (sum(detect_counts) / len(detect_counts)) if detect_counts else 0.0
    non_empty_rate = (non_empty / len(detect_counts)) if detect_counts else 0.0

    return {
        "avg_latency_ms": float(latency_stats.mean_ms),
        "p95_latency_ms": float(latency_stats.p95_ms),
        "fps": float(fps),
        "avg_detect_count": float(avg_detect_count),
        "non_empty_frame_rate": float(non_empty_rate),
    }


def _benchmark_variant(
    detector_config_path: str,
    *,
    source: str | None,
    warmup_frames: int,
    measure_frames: int,
    skip_val: bool,
    logger,
) -> dict[str, Any]:
    config = _load_detector_cfg(detector_config_path)
    variant = config.get("detector", {}).get("variant", "unknown")
    imgsz = int(config.get("detector", {}).get("infer", {}).get("imgsz", 640))

    predictor = DetectorPredictor(config, logger=logger)
    model = predictor.model

    params = _count_params(model)
    flops = _estimate_flops(model, imgsz=imgsz)

    map50 = None
    map5095 = None
    if not skip_val:
        map50, map5095 = _run_map_validation(
            model,
            config,
            logger,
            device=predictor.loader.device,
        )

    source_resolved: str | None = None
    if source is not None:
        source_resolved = str(resolve_path(source, base_dir=config["_meta"]["project_root"]))
    frames = _load_source_frames(source_resolved, measure_frames, imgsz)
    runtime_metrics = _benchmark_fps(
        predictor,
        frames=frames,
        warmup_frames=warmup_frames,
        stream_id=f"{variant}_benchmark",
    )

    return {
        "variant": variant,
        "config_path": str(config["_meta"]["active_config_path"]),
        "model_spec": config.get("detector", {}).get("model_path")
        or config.get("detector", {}).get("model_name"),
        "imgsz": imgsz,
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "params": params,
        "flops": flops,
        "avg_latency_ms": runtime_metrics["avg_latency_ms"],
        "p95_latency_ms": runtime_metrics["p95_latency_ms"],
        "fps": runtime_metrics["fps"],
        "avg_detect_count": runtime_metrics["avg_detect_count"],
        "non_empty_frame_rate": runtime_metrics["non_empty_frame_rate"],
    }


def _output_paths(
    base_config: dict[str, Any],
    output_json_arg: str | None,
) -> tuple[Path, Path]:
    reports_dir = Path(base_config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = (
        resolve_path(output_json_arg, base_dir=base_config["_meta"]["project_root"])
        if output_json_arg
        else reports_dir / f"single_stream_compare_{timestamp}.json"
    )
    output_csv = output_json.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    return (output_json, output_csv)


def _write_outputs(results: list[dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    recorder = ExperimentRecorder(experiment_name="single_stream_compare", output_dir=output_json.parent)
    recorder.add_metadata("output_json", str(output_json))
    recorder.add_metadata("output_csv", str(output_csv))
    for row in results:
        recorder.add_record(row)

    payload = recorder.to_payload()
    with output_json.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for row in results for key in row.keys()})
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _resolve_compare_settings(
    experiment_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str, str | None, int, int]:
    exp = experiment_cfg.get("experiment", {})
    baseline_cfg = args.baseline_config or exp.get("baseline_config") or "configs/detector/yolov8_head.yaml"
    enhanced_cfg = args.enhanced_config or exp.get("enhanced_config") or "configs/detector/yolov12_p2_head.yaml"
    source = args.source if args.source is not None else exp.get("source")
    warmup_frames = int(args.warmup_frames if args.warmup_frames is not None else exp.get("warmup_frames", 20))
    measure_frames = int(
        args.measure_frames if args.measure_frames is not None else exp.get("measure_frames", 100)
    )
    return (str(baseline_cfg), str(enhanced_cfg), source, warmup_frames, measure_frames)


def main() -> int:
    args = build_parser().parse_args()
    experiment_cfg = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(experiment_cfg, logger_name="benchmark_single")
    logger.info("Loaded benchmark config: %s", experiment_cfg["_meta"]["active_config_path"])

    baseline_cfg, enhanced_cfg, source, warmup_frames, measure_frames = _resolve_compare_settings(
        experiment_cfg,
        args,
    )
    logger.info(
        "Compare settings: baseline=%s enhanced=%s source=%s warmup=%s measure=%s",
        baseline_cfg,
        enhanced_cfg,
        source or "synthetic",
        warmup_frames,
        measure_frames,
    )

    if args.dry_run:
        logger.info("Dry-run finished. Comparison not executed.")
        return 0

    baseline_result = _benchmark_variant(
        baseline_cfg,
        source=source,
        warmup_frames=warmup_frames,
        measure_frames=measure_frames,
        skip_val=args.skip_val,
        logger=logger,
    )
    enhanced_result = _benchmark_variant(
        enhanced_cfg,
        source=source,
        warmup_frames=warmup_frames,
        measure_frames=measure_frames,
        skip_val=args.skip_val,
        logger=logger,
    )
    results = [baseline_result, enhanced_result]

    output_json, output_csv = _output_paths(experiment_cfg, args.output_json)
    _write_outputs(results, output_json, output_csv)

    logger.info("Comparison finished. JSON=%s CSV=%s", output_json, output_csv)
    for row in results:
        logger.info(
            "variant=%s map50=%s map5095=%s params=%s flops=%s fps=%.3f latency_ms=%.3f",
            row["variant"],
            row["mAP@0.5"],
            row["mAP@0.5:0.95"],
            row["params"],
            row["flops"],
            row["fps"],
            row["avg_latency_ms"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



