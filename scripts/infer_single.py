"""Single-stream detector inference script."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.common.config import ConfigError, ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config
from src.detector.model_loader import ModelLoader
from src.detector.predictor import DetectorPredictor
from src.detector.schemas import FramePrediction, InferenceSummary


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VARIANT_TO_CONFIG = {
    "baseline": "configs/detector/yolov8_head.yaml",
    "enhanced": "configs/detector/yolov12_p2_head.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-stream detector inference.")
    parser.add_argument(
        "--variant",
        type=str,
        choices=("baseline", "enhanced"),
        default="baseline",
        help="Detector variant preset used when --config is not provided.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to detector config YAML. Overrides --variant.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image, video, camera index (e.g. 0), RTSP URL.",
    )
    parser.add_argument(
        "--stream-id",
        type=str,
        default="room_101",
        help="Logical stream identifier included in JSON output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process for video/camera.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional custom output JSON path.",
    )
    parser.add_argument(
        "--output-visual",
        type=str,
        default=None,
        help="Optional custom output visualization path.",
    )
    parser.add_argument(
        "--no-save-visual",
        action="store_true",
        help="Disable saving rendered image/video with boxes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup and model loading without inference.",
    )
    return parser


def _resolve_config_path(args: argparse.Namespace) -> str:
    return args.config or VARIANT_TO_CONFIG[args.variant]


def _is_camera_source(source: str) -> bool:
    return source.isdigit()


def _is_rtsp_source(source: str) -> bool:
    lowered = source.lower()
    return lowered.startswith("rtsp://") or lowered.startswith("http://") or lowered.startswith(
        "https://"
    )


def _resolve_source_path(config: dict, source: str) -> Path:
    path = resolve_path(source, base_dir=config["_meta"]["project_root"])
    if not path.exists():
        raise ConfigError(f"Source file does not exist: {path}")
    return path


def _source_kind(config: dict, source: str) -> tuple[str, str | int]:
    if _is_camera_source(source):
        return ("camera", int(source))
    if _is_rtsp_source(source):
        return ("stream", source)

    source_path = _resolve_source_path(config, source)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return ("image", str(source_path))
    return ("video", str(source_path))


def _render_prediction(frame, prediction: FramePrediction):  # type: ignore[no-untyped-def]
    for detection in prediction.detections:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        label_text = f"{detection.label}:{detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(
            frame,
            label_text,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def _default_outputs(
    config: dict,
    source: str,
    source_kind: str,
    output_json_arg: str | None,
    output_visual_arg: str | None,
    save_visual: bool,
) -> tuple[Path, Path | None]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if source_kind in {"image", "video"}:
        stem = Path(source).stem
    elif source_kind == "camera":
        stem = f"camera_{source}"
    else:
        stem = "stream_input"

    report_dir = Path(config["paths"]["reports"])
    video_dir = Path(config["paths"]["videos"])

    output_json = (
        resolve_path(output_json_arg, base_dir=config["_meta"]["project_root"])
        if output_json_arg
        else (report_dir / f"infer_{stem}_{ts}.json")
    )

    output_visual: Path | None = None
    if save_visual:
        if output_visual_arg:
            output_visual = resolve_path(output_visual_arg, base_dir=config["_meta"]["project_root"])
        else:
            suffix = ".jpg" if source_kind == "image" else ".mp4"
            output_visual = video_dir / f"infer_{stem}_{ts}{suffix}"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_visual is not None:
        output_visual.parent.mkdir(parents=True, exist_ok=True)

    return output_json, output_visual


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _run_image_inference(
    predictor: DetectorPredictor,
    *,
    source_path: str,
    stream_id: str,
    output_json: Path,
    output_visual: Path | None,
) -> InferenceSummary:
    image = cv2.imread(source_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {source_path}")

    prediction = predictor.predict_frame(
        image,
        stream_id=stream_id,
        timestamp=time.time(),
        frame_index=0,
        source=source_path,
    )

    if output_visual is not None:
        rendered = _render_prediction(image.copy(), prediction)
        cv2.imwrite(str(output_visual), rendered)

    summary = InferenceSummary(
        source=source_path,
        source_type="image",
        total_frames=1,
        avg_latency_ms=prediction.latency_ms,
        fps=1000.0 / prediction.latency_ms if prediction.latency_ms > 0 else 0.0,
        output_json=str(output_json),
        output_visual=str(output_visual) if output_visual else None,
    )
    payload = {"summary": summary.to_dict(), "predictions": [prediction.to_dict()]}
    _write_json(output_json, payload)
    return summary


def _run_video_inference(
    predictor: DetectorPredictor,
    *,
    source_value: str | int,
    source_label: str,
    source_kind: str,
    stream_id: str,
    output_json: Path,
    output_visual: Path | None,
    max_frames: int | None,
) -> InferenceSummary:
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {source_label}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_input = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_input = fps_input if fps_input > 0 else 25.0

    writer = None
    if output_visual is not None and frame_width > 0 and frame_height > 0:
        writer = cv2.VideoWriter(
            str(output_visual),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_input,
            (frame_width, frame_height),
        )

    predictions: list[dict] = []
    latency_list: list[float] = []
    frame_index = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            prediction = predictor.predict_frame(
                frame,
                stream_id=stream_id,
                timestamp=time.time(),
                frame_index=frame_index,
                source=source_label,
            )
            predictions.append(prediction.to_dict())
            latency_list.append(prediction.latency_ms)

            if writer is not None:
                writer.write(_render_prediction(frame.copy(), prediction))

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    total_frames = frame_index
    avg_latency_ms = sum(latency_list) / len(latency_list) if latency_list else 0.0
    fps = (total_frames / elapsed) if elapsed > 0 else 0.0

    summary = InferenceSummary(
        source=source_label,
        source_type=source_kind,
        total_frames=total_frames,
        avg_latency_ms=avg_latency_ms,
        fps=fps,
        output_json=str(output_json),
        output_visual=str(output_visual) if output_visual else None,
    )
    payload = {"summary": summary.to_dict(), "predictions": predictions}
    _write_json(output_json, payload)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    config_path = _resolve_config_path(args)
    config = ensure_output_dirs(load_config(config_path))
    logger = configure_logger_from_config(config, logger_name="infer_single")

    logger.info(
        "Loaded config: %s (variant=%s)",
        config["_meta"]["active_config_path"],
        config.get("detector", {}).get("variant", args.variant),
    )
    kind, source_value = _source_kind(config, args.source)
    source_label = str(source_value)

    output_json, output_visual = _default_outputs(
        config,
        source=str(args.source),
        source_kind=kind,
        output_json_arg=args.output_json,
        output_visual_arg=args.output_visual,
        save_visual=(not args.no_save_visual),
    )
    logger.info("Source=%s kind=%s", source_label, kind)
    logger.info("Output JSON=%s", output_json)
    if output_visual is not None:
        logger.info("Output visual=%s", output_visual)

    if args.dry_run:
        _ = ModelLoader(config, logger=logger)
        logger.info("Dry-run finished. Inference not executed.")
        return 0

    predictor = DetectorPredictor(config, logger=logger)

    if kind == "image":
        summary = _run_image_inference(
            predictor,
            source_path=str(source_value),
            stream_id=args.stream_id,
            output_json=output_json,
            output_visual=output_visual,
        )
    else:
        max_frames = args.max_frames
        if kind == "camera" and max_frames is None:
            max_frames = 300
            logger.warning("Camera source without --max-frames, defaulting to 300 frames.")

        summary = _run_video_inference(
            predictor,
            source_value=source_value,
            source_label=source_label,
            source_kind=kind,
            stream_id=args.stream_id,
            output_json=output_json,
            output_visual=output_visual,
            max_frames=max_frames,
        )

    logger.info(
        "Inference finished. frames=%s avg_latency_ms=%.3f fps=%.3f",
        summary.total_frames,
        summary.avg_latency_ms,
        summary.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
