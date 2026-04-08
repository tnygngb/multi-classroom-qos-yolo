"""Stage-4 edge node orchestration with shared worker pool."""

from __future__ import annotations

import logging
import signal
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.common.config import ensure_output_dirs, load_config
from src.edge.batching import InferenceTask, build_batches
from src.edge.result_sink import ResultSink
from src.edge.worker_pool import DetectorWorkerPool, WorkerResult
from src.streams.source_manager import SourceManager


@dataclass(slots=True)
class EdgeNodeStats:
    """Runtime stats for stage-4 fixed policy pipeline."""

    start_ts: float
    total_infer_count: int = 0
    total_latency_ms: float = 0.0
    total_queue_wait_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        elapsed = max(time.time() - self.start_ts, 1e-6)
        avg_latency = self.total_latency_ms / self.total_infer_count if self.total_infer_count else 0.0
        avg_queue_wait = self.total_queue_wait_ms / self.total_infer_count if self.total_infer_count else 0.0
        return {
            "elapsed_sec": elapsed,
            "total_infer_count": float(self.total_infer_count),
            "total_throughput_fps": float(self.total_infer_count / elapsed),
            "avg_latency_ms": float(avg_latency),
            "avg_queue_wait_ms": float(avg_queue_wait),
        }


class EdgeNode:
    """Fixed-policy multi-stream inference node for stage 4."""

    def __init__(
        self,
        *,
        streams_config_path: str,
        detector_config_path: str,
        logger: logging.Logger | None = None,
        num_workers: int = 1,
        dispatch_interval_sec: float = 0.05,
        status_interval_sec: float = 2.0,
        batch_size: int = 1,
        write_csv: bool = True,
        max_pending_tasks: int = 256,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.streams_config = ensure_output_dirs(load_config(streams_config_path))
        self.detector_config = ensure_output_dirs(load_config(detector_config_path))
        self.dispatch_interval_sec = float(dispatch_interval_sec)
        self.status_interval_sec = float(status_interval_sec)
        self.batch_size = int(batch_size)
        self.max_pending_tasks = int(max_pending_tasks)

        self.source_manager = SourceManager(self.streams_config, logger=self.logger)
        self.worker_pool = DetectorWorkerPool(
            self.detector_config,
            num_workers=num_workers,
            logger=self.logger,
        )
        self.result_sink = ResultSink(
            reports_dir=self.streams_config["paths"]["reports"],
            run_name=f"edge_stage4_{datetime.now():%Y%m%d_%H%M%S}",
            write_csv=write_csv,
        )

        self._pending_futures: list[Future] = []
        self._stop_event = threading.Event()
        self._stats = EdgeNodeStats(start_ts=time.time())
        self._per_stream_count: dict[str, int] = {stream_id: 0 for stream_id in self.source_manager.stream_ids}
        self._per_stream_latency: dict[str, float] = {stream_id: 0.0 for stream_id in self.source_manager.stream_ids}

    def request_stop(self) -> None:
        self._stop_event.set()

    def _submit_sampled_tasks(self) -> None:
        sampled = self.source_manager.poll_sampled_frames()
        tasks = [
            InferenceTask(
                stream_id=stream_id,
                frame=buffered.frame,
                timestamp=buffered.timestamp,
                frame_index=buffered.frame_index,
                source=self.source_manager.state_store.get(stream_id).source,
            )
            for stream_id, buffered in sampled.items()
        ]
        if not tasks:
            return

        if self.worker_pool.pending_tasks() >= self.max_pending_tasks:
            self.logger.warning(
                "Skipping task submission because pending queue is high. pending=%s max=%s",
                self.worker_pool.pending_tasks(),
                self.max_pending_tasks,
            )
            return

        for batch in build_batches(tasks, self.batch_size):
            for task in batch:
                try:
                    future = self.worker_pool.submit(task)
                    self._pending_futures.append(future)
                except RuntimeError as exc:
                    self.logger.warning("Task dropped due to full worker queue. err=%s", exc)

    def _collect_results(self) -> None:
        if not self._pending_futures:
            return

        still_pending: list[Future] = []
        for future in self._pending_futures:
            if not future.done():
                still_pending.append(future)
                continue

            try:
                worker_result = future.result()
            except Exception as exc:  # pragma: no cover - runtime dependent
                self.logger.warning("Inference future failed: %s", exc)
                continue

            self._handle_worker_result(worker_result)

        self._pending_futures = still_pending

    def _handle_worker_result(self, worker_result: WorkerResult) -> None:
        prediction = worker_result.prediction
        stream_id = prediction.stream_id
        latency_ms = float(prediction.latency_ms)
        detect_count = len(prediction.detections)

        self.result_sink.write_prediction(prediction)
        self.source_manager.state_store.update(
            stream_id,
            last_infer_latency=latency_ms,
            last_detect_count=detect_count,
            buffer_size=self.source_manager.get_buffer(stream_id).size(),
            drop_count=self.source_manager.get_buffer(stream_id).drop_count,
        )

        self._stats.total_infer_count += 1
        self._stats.total_latency_ms += latency_ms
        self._stats.total_queue_wait_ms += worker_result.queue_wait_ms

        self._per_stream_count[stream_id] = self._per_stream_count.get(stream_id, 0) + 1
        self._per_stream_latency[stream_id] = self._per_stream_latency.get(stream_id, 0.0) + latency_ms

    def _log_status(self) -> None:
        base = self._stats.to_dict()
        self.logger.info(
            "status total_fps=%.3f infer_count=%s avg_latency_ms=%.3f avg_queue_wait_ms=%.3f pending=%s",
            base["total_throughput_fps"],
            int(base["total_infer_count"]),
            base["avg_latency_ms"],
            base["avg_queue_wait_ms"],
            self.worker_pool.pending_tasks(),
        )

        elapsed = max(base["elapsed_sec"], 1e-6)
        for state in self.source_manager.get_state_snapshot():
            stream_id = state["stream_id"]
            stream_count = self._per_stream_count.get(stream_id, 0)
            stream_latency_sum = self._per_stream_latency.get(stream_id, 0.0)
            stream_fps = stream_count / elapsed
            stream_avg_latency = stream_latency_sum / stream_count if stream_count else 0.0
            self.logger.info(
                "stream_id=%s online=%s stream_fps=%.3f avg_latency_ms=%.3f buffer=%s drop=%s read_fps=%.2f",
                stream_id,
                state["online"],
                stream_fps,
                stream_avg_latency,
                state["buffer_size"],
                state["drop_count"],
                float(state.get("read_fps", 0.0)),
            )

    def run(self, *, duration_sec: float) -> dict[str, Any]:
        self.logger.info(
            "EdgeNode started. streams_config=%s detector_config=%s",
            self.streams_config["_meta"]["active_config_path"],
            self.detector_config["_meta"]["active_config_path"],
        )

        self.worker_pool.start()
        self.source_manager.start_all()

        started = time.time()
        next_status = started
        try:
            while not self._stop_event.is_set():
                now = time.time()
                if duration_sec > 0 and (now - started) >= duration_sec:
                    break

                self._submit_sampled_tasks()
                self._collect_results()

                if now >= next_status:
                    self._log_status()
                    next_status = now + max(self.status_interval_sec, 0.2)

                time.sleep(max(self.dispatch_interval_sec, 0.01))
        finally:
            self._stop_event.set()
            self.source_manager.stop_all()

            # Drain any finished results.
            drain_deadline = time.time() + 3.0
            while self._pending_futures and time.time() < drain_deadline:
                self._collect_results()
                time.sleep(0.05)

            self.worker_pool.stop()
            self.result_sink.close()

        summary = self._build_summary()
        self.result_sink.write_summary(summary)
        self.logger.info("EdgeNode stopped. summary=%s", summary)
        return summary

    def _build_summary(self) -> dict[str, Any]:
        base = self._stats.to_dict()
        elapsed = max(base["elapsed_sec"], 1e-6)
        per_stream = {}
        for stream_id in self.source_manager.stream_ids:
            count = self._per_stream_count.get(stream_id, 0)
            latency_sum = self._per_stream_latency.get(stream_id, 0.0)
            per_stream[stream_id] = {
                "infer_count": count,
                "throughput_fps": count / elapsed,
                "avg_latency_ms": (latency_sum / count) if count else 0.0,
            }

        return {
            "stats": base,
            "per_stream": per_stream,
            "results_jsonl": str(self.result_sink.jsonl_path),
            "results_csv": str(self.result_sink.csv_path) if self.result_sink.write_csv else None,
            "summary_json": str(self.result_sink.summary_path),
        }


def install_signal_handlers(node: EdgeNode) -> None:
    """Install SIGINT/SIGTERM handlers to stop node gracefully."""

    def _handler(_sig, _frame) -> None:  # type: ignore[no-untyped-def]
        node.request_stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
