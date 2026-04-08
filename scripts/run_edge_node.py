"""Stage-5 multi-stream edge inference entrypoint with QoS scheduler."""

from __future__ import annotations

import argparse
import json

from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config
from src.edge.node import EdgeNode, install_signal_handlers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-5 edge node with heuristic QoS scheduling.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/streams/demo_4streams.yaml",
        help="Path to streams config YAML.",
    )
    parser.add_argument(
        "--detector-config",
        type=str,
        default="configs/detector/yolov8_head.yaml",
        help="Path to detector config YAML.",
    )
    parser.add_argument(
        "--scheduler-config",
        type=str,
        default="configs/scheduler/qos_policy.yaml",
        help="Path to scheduler policy YAML.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=15.0,
        help="Run time in seconds. Use <=0 for infinite run.",
    )
    parser.add_argument(
        "--status-interval-sec",
        type=float,
        default=2.0,
        help="Seconds between status log outputs.",
    )
    parser.add_argument(
        "--dispatch-interval-sec",
        type=float,
        default=0.05,
        help="Main loop dispatch interval in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of shared detector workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Logical batch size for task grouping. Stage-5 default is 1.",
    )
    parser.add_argument(
        "--max-pending-tasks",
        type=int,
        default=256,
        help="Backpressure threshold to stop submitting new tasks when queue is high.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV output; keep JSONL only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup only, do not start readers/workers.",
    )
    parser.add_argument(
        "--disable-qos",
        action="store_true",
        help="Disable dynamic QoS scheduler and keep static profile.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    streams_config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(streams_config, logger_name="run_edge_node")

    logger.info("Loaded streams config: %s", streams_config["_meta"]["active_config_path"])
    logger.info("Detector config: %s", args.detector_config)
    logger.info("Scheduler config: %s", args.scheduler_config)

    node = EdgeNode(
        streams_config_path=args.config,
        detector_config_path=args.detector_config,
        scheduler_config_path=args.scheduler_config,
        logger=logger,
        num_workers=args.workers,
        dispatch_interval_sec=args.dispatch_interval_sec,
        status_interval_sec=args.status_interval_sec,
        batch_size=args.batch_size,
        write_csv=(not args.no_csv),
        max_pending_tasks=args.max_pending_tasks,
        enable_qos=(not args.disable_qos),
    )

    if args.dry_run:
        logger.info("Dry-run finished. EdgeNode initialized successfully.")
        return 0

    install_signal_handlers(node)
    summary = node.run(duration_sec=args.duration_sec)
    logger.info("Final summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
