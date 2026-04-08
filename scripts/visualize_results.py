"""Results visualization entrypoint (stage 0 placeholder)."""

from __future__ import annotations

import argparse

from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize experiment results (placeholder).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base config YAML.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/metrics",
        help="Input metrics directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports",
        help="Directory for generated charts.",
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
    logger = configure_logger_from_config(config, logger_name="visualize_results")

    logger.info("Loaded config: %s", config["_meta"]["active_config_path"])
    logger.info("Visualize input: %s | output_dir: %s", args.input, args.output_dir)
    if args.dry_run:
        logger.info("Dry-run finished. Visualization logic is not implemented in phase 0.")
        return 0

    logger.warning("Phase 0 placeholder only. Visualization logic will be added in phase 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
