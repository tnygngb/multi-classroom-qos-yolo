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
