"""Single-stream detector training entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.config import ConfigError, ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config
from src.detector.model_loader import ModelLoader


VARIANT_TO_CONFIG = {
    "baseline": "configs/detector/yolov8_head.yaml",
    "enhanced": "configs/detector/yolov12_p2_head.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train detector with Ultralytics.")
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
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML file (overrides config).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Override training batch size.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name under outputs/runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate training setup without starting training.",
    )
    return parser


def _resolve_config_path(args: argparse.Namespace) -> str:
    return args.config or VARIANT_TO_CONFIG[args.variant]


def _require_data_path(config: dict[str, Any], data_override: str | None) -> Path:
    detector_cfg = config.get("detector", {})
    train_cfg = detector_cfg.get("train", {})
    data_value = data_override or train_cfg.get("data")
    if not data_value:
        raise ConfigError("Missing dataset config. Set detector.train.data or pass --data.")

    path = resolve_path(str(data_value), base_dir=config["_meta"]["project_root"])
    if not path.exists():
        raise ConfigError(f"Dataset config path does not exist: {path}")
    return path


def _build_train_kwargs(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    detector_cfg = config.get("detector", {})
    train_cfg = detector_cfg.get("train", {})
    runs_dir = Path(config["paths"]["runs"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.name or train_cfg.get(
        "run_name",
        f"{detector_cfg.get('variant', 'baseline')}_{datetime.now():%Y%m%d_%H%M%S}",
    )

    return {
        "data": str(_require_data_path(config, args.data)),
        "epochs": int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 100)),
        "imgsz": int(args.imgsz if args.imgsz is not None else train_cfg.get("imgsz", 640)),
        "batch": int(args.batch if args.batch is not None else train_cfg.get("batch", 16)),
        "workers": int(train_cfg.get("workers", 4)),
        "patience": int(train_cfg.get("patience", 50)),
        "project": str(runs_dir),
        "name": str(run_name),
        "device": str(detector_cfg.get("device", config.get("device", "cpu"))),
        "exist_ok": bool(train_cfg.get("exist_ok", True)),
        "pretrained": bool(train_cfg.get("pretrained", True)),
    }


def main() -> int:
    args = build_parser().parse_args()
    config_path = _resolve_config_path(args)
    config = ensure_output_dirs(load_config(config_path))
    logger = configure_logger_from_config(config, logger_name="train_detector")

    logger.info(
        "Loaded config: %s (variant=%s)",
        config["_meta"]["active_config_path"],
        config.get("detector", {}).get("variant", args.variant),
    )
    train_kwargs = _build_train_kwargs(config, args)
    logger.info(
        "Training setup ready. model=%s data=%s device=%s project=%s name=%s",
        config.get("detector", {}).get("model_path") or config.get("detector", {}).get("model_name"),
        train_kwargs["data"],
        train_kwargs["device"],
        train_kwargs["project"],
        train_kwargs["name"],
    )

    loader = ModelLoader(config, logger=logger)

    if args.dry_run:
        logger.info("Dry-run finished. Training not started.")
        return 0

    model = loader.load()
    try:
        logger.info("Training started.")
        _ = model.train(**train_kwargs)
    except Exception as exc:
        logger.exception("Training failed.")
        raise RuntimeError(f"Training failed: {exc}") from exc

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    logger.info("Training finished. save_dir=%s", save_dir if save_dir else "unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
