"""Cloud review service entrypoint (stage 6)."""

from __future__ import annotations

import argparse

import uvicorn

from src.cloud.app import create_app
from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cloud review service (stage 6).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cloud/cloud_review.yaml",
        help="Path to cloud service config YAML.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override host from config.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate config and app creation, then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(config, logger_name="run_cloud_service")

    cloud_cfg = config.get("cloud_service", {})
    host = str(args.host or cloud_cfg.get("host", "0.0.0.0"))
    port = int(args.port if args.port is not None else cloud_cfg.get("port", 8081))

    logger.info("Loaded config: %s", config["_meta"]["active_config_path"])
    logger.info("Cloud service bind: %s:%s", host, port)

    app = create_app(args.config)
    if args.dry_run:
        logger.info("Dry-run finished. Cloud app initialized successfully.")
        return 0

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=str(config.get("log_level", "INFO")).lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
