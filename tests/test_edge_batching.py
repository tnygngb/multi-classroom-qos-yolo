from __future__ import annotations

from src.edge.batching import InferenceTask, build_batches


def _task(idx: int) -> InferenceTask:
    return InferenceTask(
        stream_id="room_101",
        frame=f"f{idx}",
        timestamp=float(idx),
        frame_index=idx,
        source="demo",
    )


def test_build_batches_default_single_frame() -> None:
    tasks = [_task(0), _task(1), _task(2)]
    batches = build_batches(tasks, batch_size=1)
    assert len(batches) == 3
    assert all(len(batch) == 1 for batch in batches)


def test_build_batches_multi_size() -> None:
    tasks = [_task(0), _task(1), _task(2), _task(3), _task(4)]
    batches = build_batches(tasks, batch_size=2)
    assert [len(batch) for batch in batches] == [2, 2, 1]
