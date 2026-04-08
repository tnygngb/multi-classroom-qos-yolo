"""Shared detector worker pool for multi-stream inference."""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from src.common.config import deep_merge_dict
from src.detector.predictor import DetectorPredictor
from src.detector.schemas import FramePrediction
from src.edge.batching import InferenceTask


@dataclass(slots=True)
class WorkerResult:
    """Inference result metadata from one worker."""

    prediction: FramePrediction
    worker_id: int
    queue_wait_ms: float
    mode: str
    model_variant: str
    input_size: int | None


class DetectorWorkerPool:
    """
    Shared inference worker pool.

    It avoids one-model-per-stream by reusing a small pool of workers across
    all streams.
    """

    def __init__(
        self,
        detector_config: dict[str, Any],
        *,
        num_workers: int = 1,
        max_queue_size: int = 128,
        variant_overrides: dict[str, dict[str, Any]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")

        self.detector_config = detector_config
        self.num_workers = int(num_workers)
        self.max_queue_size = int(max_queue_size)
        self.variant_overrides = variant_overrides or {}
        self.logger = logger or logging.getLogger(__name__)

        self._rr_index = 0
        self._lock = threading.Lock()
        self._started = False
        self._queues: list[queue.Queue[tuple[InferenceTask, Future] | None]] = []
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._queues = [queue.Queue(maxsize=self.max_queue_size) for _ in range(self.num_workers)]
        self._threads = []

        for worker_id in range(self.num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"detector_worker_{worker_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

        self._started = True
        self.logger.info(
            "Worker pool started. num_workers=%s max_queue_size=%s",
            self.num_workers,
            self.max_queue_size,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._stop_event.set()
        for queue_ref in self._queues:
            queue_ref.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._started = False
        self.logger.info("Worker pool stopped.")

    def pending_tasks(self) -> int:
        return sum(queue_ref.qsize() for queue_ref in self._queues)

    def submit(self, task: InferenceTask) -> Future:
        if not self._started:
            raise RuntimeError("Worker pool is not started.")

        future: Future = Future()
        with self._lock:
            worker_id = self._rr_index
            self._rr_index = (self._rr_index + 1) % self.num_workers

        try:
            self._queues[worker_id].put_nowait((task, future))
        except queue.Full as exc:
            future.set_exception(RuntimeError(f"Worker queue is full for worker_id={worker_id}"))
            self.logger.warning("Queue full. worker_id=%s pending=%s", worker_id, self.pending_tasks())
            raise RuntimeError("Worker queue full.") from exc

        return future

    def _worker_loop(self, worker_id: int) -> None:
        queue_ref = self._queues[worker_id]
        predictor_cache: dict[str, DetectorPredictor] = {}
        self.logger.info("Worker initialized. worker_id=%s", worker_id)

        while not self._stop_event.is_set():
            item = queue_ref.get()
            if item is None:
                break

            task, future = item
            if future.cancelled():
                continue

            started = time.perf_counter()
            queue_wait_ms = (started - task.enqueue_monotonic) * 1000.0
            try:
                model_variant = str(task.model_variant or "main")
                predictor = self._get_predictor_for_variant(
                    worker_id=worker_id,
                    model_variant=model_variant,
                    cache=predictor_cache,
                )
                prediction = predictor.predict_frame(
                    task.frame,
                    stream_id=task.stream_id,
                    timestamp=task.timestamp,
                    frame_index=task.frame_index,
                    source=task.source,
                    imgsz=task.input_size,
                    mode=task.mode,
                    model_variant=model_variant,
                    cloud_review=task.cloud_review,
                )
                future.set_result(
                    WorkerResult(
                        prediction=prediction,
                        worker_id=worker_id,
                        queue_wait_ms=queue_wait_ms,
                        mode=task.mode,
                        model_variant=model_variant,
                        input_size=task.input_size,
                    )
                )
            except Exception as exc:  # pragma: no cover - runtime-dependent
                future.set_exception(exc)
                self.logger.warning("Inference failed. worker_id=%s stream_id=%s err=%s", worker_id, task.stream_id, exc)

    def _get_predictor_for_variant(
        self,
        *,
        worker_id: int,
        model_variant: str,
        cache: dict[str, DetectorPredictor],
    ) -> DetectorPredictor:
        predictor = cache.get(model_variant)
        if predictor is not None:
            return predictor

        variant_config = self._build_variant_config(model_variant)
        predictor = DetectorPredictor(variant_config, logger=self.logger)
        cache[model_variant] = predictor
        self.logger.info("Worker=%s loaded predictor variant=%s", worker_id, model_variant)
        return predictor

    def _build_variant_config(self, model_variant: str) -> dict[str, Any]:
        base = deep_merge_dict(self.detector_config, {"detector": {"variant": model_variant}})
        override = self.variant_overrides.get(model_variant)
        if isinstance(override, dict) and override:
            base = deep_merge_dict(base, {"detector": override})
        return base
