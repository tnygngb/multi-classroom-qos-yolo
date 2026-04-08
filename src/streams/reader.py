"""Video stream reader with background capture thread."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from src.common.config import resolve_path
from src.streams.frame_buffer import FrameBuffer
from src.streams.stream_state import StreamStateStore


def parse_stream_source(source: str, *, base_dir: str | Path | None = None) -> tuple[str, str | int]:
    """Parse source into ('camera'|'stream'|'file', value)."""
    source = str(source).strip()
    lowered = source.lower()

    if source.isdigit():
        return ("camera", int(source))
    if lowered.startswith("rtsp://") or lowered.startswith("http://") or lowered.startswith("https://"):
        return ("stream", source)

    path = resolve_path(source, base_dir=base_dir)
    return ("file", str(path))


class VideoStreamReader:
    """Continuously read frames from one source into its FrameBuffer."""

    def __init__(
        self,
        *,
        stream_id: str,
        source: str,
        frame_buffer: FrameBuffer,
        state_store: StreamStateStore,
        project_root: str | Path,
        logger: logging.Logger | None = None,
        reconnect_interval_sec: float = 1.0,
    ) -> None:
        self.stream_id = stream_id
        self.raw_source = source
        self.source_kind, self.source_value = parse_stream_source(source, base_dir=project_root)
        self.frame_buffer = frame_buffer
        self.state_store = state_store
        self.logger = logger or logging.getLogger(__name__)
        self.reconnect_interval_sec = float(reconnect_interval_sec)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._frame_index = 0
        self._read_count = 0
        self._fps_window: deque[float] = deque()
        self._frame_interval_sec = 0.0
        self._last_frame_monotonic: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"reader_{self.stream_id}",
            daemon=True,
        )
        self._thread.start()
        self.logger.info(
            "Reader started. stream_id=%s source=%s kind=%s",
            self.stream_id,
            self.source_value,
            self.source_kind,
        )

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._release_capture()
        self.state_store.update(self.stream_id, online=False)
        self.logger.info("Reader stopped. stream_id=%s", self.stream_id)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _open_capture(self) -> Any:
        if self.source_kind == "file":
            source_path = Path(str(self.source_value))
            if not source_path.exists():
                raise FileNotFoundError(f"Stream file source not found: {source_path}")
            return cv2.VideoCapture(str(source_path))
        return cv2.VideoCapture(self.source_value)

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _update_read_fps(self, timestamp: float) -> float:
        self._fps_window.append(timestamp)
        while self._fps_window and (timestamp - self._fps_window[0]) > 1.0:
            self._fps_window.popleft()
        return float(len(self._fps_window))

    def _mark_offline(self, error_message: str | None = None) -> None:
        self.state_store.update(
            self.stream_id,
            online=False,
            error_message=error_message,
            buffer_size=self.frame_buffer.size(),
            drop_count=self.frame_buffer.drop_count,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._capture is None or not self._capture.isOpened():
                    self._capture = self._open_capture()
                    if not self._capture.isOpened():
                        raise RuntimeError(f"Failed to open stream source: {self.source_value}")

                    if self.source_kind == "file":
                        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
                        self._frame_interval_sec = 1.0 / fps if fps > 0 else 1.0 / 25.0
                    else:
                        self._frame_interval_sec = 0.0
                    self._last_frame_monotonic = None

                    self.state_store.update(
                        self.stream_id,
                        online=True,
                        error_message=None,
                    )

                if (
                    self.source_kind == "file"
                    and self._frame_interval_sec > 0
                    and self._last_frame_monotonic is not None
                ):
                    elapsed = time.perf_counter() - self._last_frame_monotonic
                    if elapsed < self._frame_interval_sec:
                        time.sleep(self._frame_interval_sec - elapsed)

                ok, frame = self._capture.read()
                if not ok or frame is None:
                    if self.source_kind == "file":
                        # Loop local file for stable long-running demo.
                        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._last_frame_monotonic = None
                        continue

                    self._release_capture()
                    self._mark_offline("read_failed")
                    time.sleep(self.reconnect_interval_sec)
                    continue

                self._last_frame_monotonic = time.perf_counter()
                ts = time.time()
                accepted = self.frame_buffer.push(frame, ts, self._frame_index)
                if accepted:
                    self._frame_index += 1
                    self._read_count += 1

                read_fps = self._update_read_fps(ts)
                self.state_store.update(
                    self.stream_id,
                    online=True,
                    error_message=None,
                    buffer_size=self.frame_buffer.size(),
                    drop_count=self.frame_buffer.drop_count,
                    last_frame_ts=ts,
                    read_fps=read_fps,
                    read_frame_count=self._read_count,
                )
            except Exception as exc:
                self.logger.warning(
                    "Reader error. stream_id=%s error=%s",
                    self.stream_id,
                    exc,
                )
                self._release_capture()
                self._mark_offline(str(exc))
                time.sleep(self.reconnect_interval_sec)

        self._release_capture()
