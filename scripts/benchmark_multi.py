"""Multi-stream benchmark entrypoint (stage 0 placeholder)."""

from __future__ import annotations

import argparse

from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-stream benchmark (placeholder).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/multi_stream.yaml",
        help="Path to multi-stream experiment config YAML.",
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
    logger = configure_logger_from_config(config, logger_name="benchmark_multi")

    logger.info("Loaded config: %s", config["_meta"]["active_config_path"])
    if args.dry_run:
        logger.info("Dry-run finished. Benchmark logic is not implemented in phase 0.")
        return 0

    logger.warning("Phase 0 placeholder only. Multi-stream benchmark will be added in phase 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
