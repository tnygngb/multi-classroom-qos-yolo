"""Per-stream fixed-rate sampler."""

from __future__ import annotations

import threading


class FixedRateSampler:
    """Decide whether a stream frame should be sampled at fixed FPS."""

    def __init__(self, default_sampling_fps: float) -> None:
        if default_sampling_fps <= 0:
            raise ValueError("default_sampling_fps must be > 0")
        self._default_sampling_fps = float(default_sampling_fps)
        self._fps_by_stream: dict[str, float] = {}
        self._last_sample_ts: dict[str, float] = {}
        self._lock = threading.Lock()

    def set_stream_fps(self, stream_id: str, fps: float) -> None:
        if fps <= 0:
            raise ValueError("sampling fps must be > 0")
        with self._lock:
            self._fps_by_stream[stream_id] = float(fps)

    def get_stream_fps(self, stream_id: str) -> float:
        with self._lock:
            return self._fps_by_stream.get(stream_id, self._default_sampling_fps)

    def should_sample(self, stream_id: str, timestamp: float) -> bool:
        """Return True when this frame timestamp crosses the stream sampling interval."""
        with self._lock:
            fps = self._fps_by_stream.get(stream_id, self._default_sampling_fps)
            interval = 1.0 / fps
            last = self._last_sample_ts.get(stream_id)

            if last is None:
                self._last_sample_ts[stream_id] = float(timestamp)
                return True

            if (timestamp - last) >= interval:
                self._last_sample_ts[stream_id] = float(timestamp)
                return True
            return False

    def reset_stream(self, stream_id: str) -> None:
        with self._lock:
            self._last_sample_ts.pop(stream_id, None)
