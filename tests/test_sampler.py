from __future__ import annotations

from src.streams.sampler import FixedRateSampler


def test_sampler_fixed_rate_decision() -> None:
    sampler = FixedRateSampler(default_sampling_fps=2.0)  # interval 0.5s
    stream_id = "room_101"

    assert sampler.should_sample(stream_id, 0.0) is True
    assert sampler.should_sample(stream_id, 0.2) is False
    assert sampler.should_sample(stream_id, 0.5) is True
    assert sampler.should_sample(stream_id, 0.7) is False
    assert sampler.should_sample(stream_id, 1.01) is True


def test_sampler_per_stream_override() -> None:
    sampler = FixedRateSampler(default_sampling_fps=2.0)
    sampler.set_stream_fps("room_102", 4.0)  # interval 0.25s
    assert sampler.get_stream_fps("room_102") == 4.0

    assert sampler.should_sample("room_102", 0.0) is True
    assert sampler.should_sample("room_102", 0.2) is False
    assert sampler.should_sample("room_102", 0.25) is True
