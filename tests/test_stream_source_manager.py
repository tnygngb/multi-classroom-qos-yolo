from __future__ import annotations

from src.common.config import ensure_output_dirs, load_config
from src.streams.source_manager import SourceManager


def test_source_manager_builds_stream_components() -> None:
    config = ensure_output_dirs(load_config("configs/streams/demo_4streams.yaml"), create=False)
    manager = SourceManager(config)

    assert len(manager.stream_ids) == 4
    snapshot = manager.get_state_snapshot()
    assert len(snapshot) == 4
    assert {item["stream_id"] for item in snapshot} == {
        "room_101",
        "room_102",
        "room_103",
        "room_104",
    }


def test_source_manager_updates_sampling_fps() -> None:
    config = ensure_output_dirs(load_config("configs/streams/demo_4streams.yaml"), create=False)
    manager = SourceManager(config)

    manager.set_stream_sampling_fps("room_101", 7.5)
    assert manager.sampler.get_stream_fps("room_101") == 7.5

    snapshot = {item["stream_id"]: item for item in manager.get_state_snapshot()}
    assert snapshot["room_101"]["current_sampling_fps"] == 7.5
