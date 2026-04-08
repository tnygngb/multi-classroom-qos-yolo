"""Single-stream benchmark entrypoint (stage 0 placeholder)."""

from __future__ import annotations

import argparse

from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-stream benchmark (placeholder).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/single_stream.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--stream-id",
        type=str,
        default="room_101",
        help="Stream identifier to benchmark.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames planned for benchmark run.",
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
    logger = configure_logger_from_config(config, logger_name="benchmark_single")

    logger.info("Loaded config: %s", config["_meta"]["active_config_path"])
    logger.info("Benchmark target: stream_id=%s, num_frames=%s", args.stream_id, args.num_frames)
    if args.dry_run:
        logger.info("Dry-run finished. Benchmark logic is not implemented in phase 0.")
        return 0

    logger.warning("Phase 0 placeholder only. Benchmark logic will be added in phase 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
