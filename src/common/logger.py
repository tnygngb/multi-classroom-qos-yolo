"""Unified logging utilities used by all modules."""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LOGGER_NAME = "multi_classroom_qos_yolo"


def _to_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = str(level).upper()
    value = logging.getLevelName(normalized)
    if isinstance(value, int):
        return value
    raise ValueError(f"Unsupported log level: {level}")


def setup_logger(
    logger_name: str = DEFAULT_LOGGER_NAME,
    *,
    level: str | int = "INFO",
    log_dir: str | Path = "outputs/logs",
    log_filename: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(_to_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_to_level(level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_dir_path = Path(log_dir).expanduser()
    if not log_dir_path.is_absolute():
        log_dir_path = (Path.cwd() / log_dir_path).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    filename = log_filename or f"{logger_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = RotatingFileHandler(
        filename=log_dir_path / filename,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(_to_level(level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized. log_file=%s", file_handler.baseFilename)
    return logger


def configure_logger_from_config(
    config: Mapping[str, Any],
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    log_filename: str | None = None,
) -> logging.Logger:
    """Build logger from config dictionary."""
    level = config.get("log_level", "INFO")
    paths = config.get("paths", {})
    output_dir = Path(str(config.get("output_dir", "outputs")))
    log_dir = paths.get("logs", output_dir / "logs")
    return setup_logger(
        logger_name=logger_name,
        level=level,
        log_dir=log_dir,
        log_filename=log_filename,
    )


def get_logger(logger_name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Fetch an existing logger by name."""
    return logging.getLogger(logger_name)
