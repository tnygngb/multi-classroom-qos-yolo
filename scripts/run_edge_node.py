"""Stage 3 multi-stream reader and buffer runtime entrypoint."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from typing import Any

from src.common.config import ensure_output_dirs, load_config
from src.common.logger import configure_logger_from_config
from src.streams.source_manager import SourceManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-3 multi-stream edge reader service.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/streams/demo_4streams.yaml",
        help="Path to multi-stream config YAML.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=15.0,
        help="Runtime seconds for demo. Use <=0 for infinite run.",
    )
    parser.add_argument(
        "--status-interval-sec",
        type=float,
        default=2.0,
        help="Seconds between status log outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and stream construction only.",
    )
    return parser


def _log_state_snapshot(logger, snapshot: list[dict[str, Any]], sampled_count: int) -> None:
    logger.info("tick sampled_frames=%s active_streams=%s", sampled_count, len(snapshot))
    for item in snapshot:
        logger.info(
            "stream_id=%s online=%s buffer=%s drop=%s read_fps=%.2f sample_fps=%.2f last_ts=%s error=%s",
            item["stream_id"],
            item["online"],
            item["buffer_size"],
            item["drop_count"],
            float(item.get("read_fps", 0.0)),
            float(item.get("current_sampling_fps", 0.0)),
            item.get("last_frame_ts"),
            item.get("error_message"),
        )


def main() -> int:
    args = build_parser().parse_args()
    config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(config, logger_name="run_edge_node")

    logger.info("Loaded config: %s", config["_meta"]["active_config_path"])
    manager = SourceManager(config, logger=logger)
    logger.info("Configured streams=%s", manager.stream_ids)

    if args.dry_run:
        logger.info("Dry-run finished. Readers were constructed but not started.")
        return 0

    stop_event = threading.Event()

    def _request_stop(_sig, _frame) -> None:  # type: ignore[no-untyped-def]
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    manager.start_all()
    started = time.time()
    next_status_ts = started

    try:
        while not stop_event.is_set():
            now = time.time()
            if args.duration_sec > 0 and (now - started) >= args.duration_sec:
                break

            sampled = manager.poll_sampled_frames()
            if now >= next_status_ts:
                snapshot = manager.get_state_snapshot()
                _log_state_snapshot(logger, snapshot, sampled_count=len(sampled))
                next_status_ts = now + max(args.status_interval_sec, 0.2)

            time.sleep(0.05)
    finally:
        manager.stop_all()
        snapshot = manager.get_state_snapshot()
        _log_state_snapshot(logger, snapshot, sampled_count=0)
        logger.info("Edge node stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
