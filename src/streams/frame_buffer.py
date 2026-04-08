"""Thread-safe fixed-size frame buffer for one stream."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BufferedFrame:
    """Frame payload stored in FrameBuffer."""

    frame: Any
    timestamp: float
    frame_index: int


class FrameBuffer:
    """Fixed-size stream buffer with drop accounting."""

    def __init__(self, capacity: int, *, drop_policy: str = "drop_oldest") -> None:
        if capacity <= 0:
            raise ValueError("FrameBuffer capacity must be > 0")
        if drop_policy not in {"drop_oldest", "drop_newest"}:
            raise ValueError("drop_policy must be one of: drop_oldest, drop_newest")

        self.capacity = int(capacity)
        self.drop_policy = drop_policy
        self._buffer: deque[BufferedFrame] = deque()
        self._lock = threading.Lock()
        self._drop_count = 0
        self._push_count = 0

    def push(self, frame: Any, timestamp: float, frame_index: int) -> bool:
        """
        Push a frame into the buffer.

        Returns True if frame is accepted, False if dropped by policy.
        """
        item = BufferedFrame(frame=frame, timestamp=float(timestamp), frame_index=int(frame_index))
        with self._lock:
            self._push_count += 1
            if len(self._buffer) >= self.capacity:
                if self.drop_policy == "drop_oldest":
                    self._buffer.popleft()
                    self._drop_count += 1
                else:
                    self._drop_count += 1
                    return False
            self._buffer.append(item)
            return True

    def get_latest(self) -> BufferedFrame | None:
        """Return latest frame in O(1)."""
        with self._lock:
            if not self._buffer:
                return None
            latest = self._buffer[-1]
            return BufferedFrame(
                frame=latest.frame,
                timestamp=latest.timestamp,
                frame_index=latest.frame_index,
            )

    def get_batch(self, n: int) -> list[BufferedFrame]:
        """Return up to n latest frames ordered from older to newer."""
        if n <= 0:
            return []
        with self._lock:
            selected = list(self._buffer)[-n:]
            return [
                BufferedFrame(frame=item.frame, timestamp=item.timestamp, frame_index=item.frame_index)
                for item in selected
            ]

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def drop_count(self) -> int:
        with self._lock:
            return self._drop_count

    @property
    def push_count(self) -> int:
        with self._lock:
            return self._push_count

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
