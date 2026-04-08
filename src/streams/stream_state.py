"""Stream runtime state tracking utilities."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StreamState:
    """Runtime state for one stream."""

    stream_id: str
    source: str
    online: bool = False
    buffer_size: int = 0
    drop_count: int = 0
    last_frame_ts: float | None = None
    current_sampling_fps: float = 0.0
    last_infer_latency: float | None = None
    last_detect_count: int = 0
    read_fps: float = 0.0
    read_frame_count: int = 0
    error_message: str | None = None


class StreamStateStore:
    """Thread-safe registry for all stream states."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, StreamState] = {}

    def register(self, stream_id: str, source: str, sampling_fps: float) -> StreamState:
        with self._lock:
            state = StreamState(
                stream_id=stream_id,
                source=source,
                current_sampling_fps=float(sampling_fps),
            )
            self._states[stream_id] = state
            return state

    def exists(self, stream_id: str) -> bool:
        with self._lock:
            return stream_id in self._states

    def update(self, stream_id: str, **fields: Any) -> None:
        with self._lock:
            state = self._states.get(stream_id)
            if state is None:
                raise KeyError(f"Unknown stream_id: {stream_id}")
            for key, value in fields.items():
                if not hasattr(state, key):
                    raise AttributeError(f"StreamState has no field '{key}'")
                setattr(state, key, value)

    def get(self, stream_id: str) -> StreamState:
        with self._lock:
            state = self._states.get(stream_id)
            if state is None:
                raise KeyError(f"Unknown stream_id: {stream_id}")
            return StreamState(**asdict(state))

    def all(self) -> dict[str, StreamState]:
        with self._lock:
            return {stream_id: StreamState(**asdict(state)) for stream_id, state in self._states.items()}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(state) for state in self._states.values()]
