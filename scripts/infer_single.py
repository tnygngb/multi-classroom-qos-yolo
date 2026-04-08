"""Single-stream inference entrypoint (stage 0 placeholder)."""

from __future__ import annotations

import argparse

from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-stream inference (placeholder).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detector/yolov8_head.yaml",
        help="Path to detector config YAML.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/demos/room_101.mp4",
        help="Input image/video/camera source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate config and logger, then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(config, logger_name="infer_single")

    logger.info("Loaded config: %s", config["_meta"]["active_config_path"])
    logger.info("Input source: %s", args.source)
    if args.dry_run:
        logger.info("Dry-run finished. Inference logic is not implemented in phase 0.")
        return 0

    logger.warning("Phase 0 placeholder only. Inference logic will be added in phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
