"""Manager for lifecycle of multi-stream readers."""

from __future__ import annotations

import logging
from typing import Any

from src.streams.frame_buffer import BufferedFrame, FrameBuffer
from src.streams.reader import VideoStreamReader
from src.streams.sampler import FixedRateSampler
from src.streams.stream_state import StreamStateStore


class SourceManager:
    """Create and control readers based on stream config."""

    def __init__(self, config: dict[str, Any], *, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.project_root = config["_meta"]["project_root"]
        self.buffer_size = int(config.get("buffer_size", 32))
        self.sampling_fps = float(config.get("sampling_fps", 5))
        self.drop_policy = str(config.get("drop_policy", "drop_oldest"))

        self.state_store = StreamStateStore()
        self.sampler = FixedRateSampler(self.sampling_fps)

        self._buffers: dict[str, FrameBuffer] = {}
        self._readers: dict[str, VideoStreamReader] = {}
        self._last_sampled_frame_index: dict[str, int] = {}

        self._build_from_config()

    def _build_from_config(self) -> None:
        streams = self.config.get("streams", [])
        if not isinstance(streams, list) or not streams:
            raise ValueError("streams config must be a non-empty list.")

        for item in streams:
            stream_id = str(item.get("stream_id", "")).strip()
            source = str(item.get("source", "")).strip()
            if not stream_id:
                raise ValueError("Each stream entry must include non-empty stream_id.")
            if not source:
                raise ValueError(f"Stream '{stream_id}' missing source.")
            if stream_id in self._readers:
                raise ValueError(f"Duplicate stream_id found: {stream_id}")

            stream_fps = float(item.get("sampling_fps", self.sampling_fps))
            stream_buffer_size = int(item.get("buffer_size", self.buffer_size))

            frame_buffer = FrameBuffer(stream_buffer_size, drop_policy=self.drop_policy)
            self.state_store.register(stream_id, source, stream_fps)
            self.sampler.set_stream_fps(stream_id, stream_fps)

            reader = VideoStreamReader(
                stream_id=stream_id,
                source=source,
                frame_buffer=frame_buffer,
                state_store=self.state_store,
                project_root=self.project_root,
                logger=self.logger,
            )

            self._buffers[stream_id] = frame_buffer
            self._readers[stream_id] = reader
            self._last_sampled_frame_index[stream_id] = -1

    @property
    def stream_ids(self) -> list[str]:
        return list(self._readers.keys())

    def start_all(self) -> None:
        for stream_id, reader in self._readers.items():
            self.logger.info("Starting stream reader: %s", stream_id)
            reader.start()

    def stop_all(self) -> None:
        for stream_id, reader in self._readers.items():
            self.logger.info("Stopping stream reader: %s", stream_id)
            reader.stop()

    def get_buffer(self, stream_id: str) -> FrameBuffer:
        return self._buffers[stream_id]

    def get_state_snapshot(self) -> list[dict[str, Any]]:
        snapshots = self.state_store.snapshot()
        for item in snapshots:
            stream_id = item["stream_id"]
            item["buffer_size"] = self._buffers[stream_id].size()
            item["drop_count"] = self._buffers[stream_id].drop_count
        return snapshots

    def poll_sampled_frames(self) -> dict[str, BufferedFrame]:
        """
        Poll latest frames according to per-stream fixed-rate sampling.

        Returns stream_id -> BufferedFrame map for frames selected this cycle.
        """
        sampled: dict[str, BufferedFrame] = {}
        for stream_id in self.stream_ids:
            latest = self._buffers[stream_id].get_latest()
            if latest is None:
                continue

            last_sampled_idx = self._last_sampled_frame_index.get(stream_id, -1)
            if latest.frame_index == last_sampled_idx:
                continue

            if self.sampler.should_sample(stream_id, latest.timestamp):
                sampled[stream_id] = latest
                self._last_sampled_frame_index[stream_id] = latest.frame_index

        return sampled
