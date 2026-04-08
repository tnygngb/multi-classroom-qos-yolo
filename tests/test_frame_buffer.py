from __future__ import annotations

from src.streams.frame_buffer import FrameBuffer


def test_frame_buffer_drop_oldest_policy() -> None:
    buffer = FrameBuffer(capacity=2, drop_policy="drop_oldest")
    assert buffer.push("f1", 1.0, 1) is True
    assert buffer.push("f2", 2.0, 2) is True
    assert buffer.push("f3", 3.0, 3) is True

    assert buffer.size() == 2
    assert buffer.drop_count == 1
    batch = buffer.get_batch(10)
    assert [item.frame_index for item in batch] == [2, 3]


def test_frame_buffer_drop_newest_policy() -> None:
    buffer = FrameBuffer(capacity=1, drop_policy="drop_newest")
    assert buffer.push("f1", 1.0, 1) is True
    assert buffer.push("f2", 2.0, 2) is False

    latest = buffer.get_latest()
    assert latest is not None
    assert latest.frame_index == 1
    assert buffer.drop_count == 1
