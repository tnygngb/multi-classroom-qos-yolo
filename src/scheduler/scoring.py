"""Heuristic QoS scoring for multi-stream scheduling."""

from __future__ import annotations

from dataclasses import dataclass


def clamp01(value: float) -> float:
    """Clamp a value into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class ScoreWeights:
    """Weighted coefficients for score components."""

    density: float
    activity: float
    anomaly: float
    queue_pressure: float

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ScoreWeights":
        raw = cls(
            density=float(data.get("density", 0.35)),
            activity=float(data.get("activity", 0.20)),
            anomaly=float(data.get("anomaly", 0.25)),
            queue_pressure=float(data.get("queue_pressure", 0.20)),
        )
        return normalize_weights(raw)


@dataclass(slots=True)
class ScoreTuning:
    """Normalization baselines for heuristic scoring."""

    density_reference: float = 25.0
    activity_reference_fps: float = 12.0
    anomaly_detection_reference: float = 12.0
    anomaly_latency_ms: float = 150.0
    queue_buffer_reference: float = 1.0
    queue_drop_rate_reference: float = 0.20
    latency_target_ms: float = 80.0

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> "ScoreTuning":
        if not isinstance(data, dict):
            return cls()
        return cls(
            density_reference=float(data.get("density_reference", 25.0)),
            activity_reference_fps=float(data.get("activity_reference_fps", 12.0)),
            anomaly_detection_reference=float(data.get("anomaly_detection_reference", 12.0)),
            anomaly_latency_ms=float(data.get("anomaly_latency_ms", 150.0)),
            queue_buffer_reference=float(data.get("queue_buffer_reference", 1.0)),
            queue_drop_rate_reference=float(data.get("queue_drop_rate_reference", 0.20)),
            latency_target_ms=float(data.get("latency_target_ms", 80.0)),
        )


@dataclass(slots=True)
class StreamScoreInput:
    """Scheduler input signals for one stream."""

    stream_id: str
    people_estimate: int
    detect_count: int
    avg_latency_ms: float
    drop_rate: float
    buffer_occupancy: float
    activity_fps: float
    anomaly_hint: float = 0.0


@dataclass(slots=True)
class StreamScoreBreakdown:
    """Detailed score output for explainable scheduling."""

    stream_id: str
    total_score: float
    density_score: float
    activity_score: float
    anomaly_score: float
    queue_pressure_score: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "stream_id": self.stream_id,
            "total_score": float(self.total_score),
            "density_score": float(self.density_score),
            "activity_score": float(self.activity_score),
            "anomaly_score": float(self.anomaly_score),
            "queue_pressure_score": float(self.queue_pressure_score),
        }


def normalize_weights(weights: ScoreWeights) -> ScoreWeights:
    """Normalize all weights to sum to 1.0."""
    total = (
        float(weights.density)
        + float(weights.activity)
        + float(weights.anomaly)
        + float(weights.queue_pressure)
    )
    if total <= 0:
        return ScoreWeights(density=0.35, activity=0.20, anomaly=0.25, queue_pressure=0.20)
    return ScoreWeights(
        density=weights.density / total,
        activity=weights.activity / total,
        anomaly=weights.anomaly / total,
        queue_pressure=weights.queue_pressure / total,
    )


def score_stream(
    score_input: StreamScoreInput,
    *,
    weights: ScoreWeights,
    tuning: ScoreTuning,
) -> StreamScoreBreakdown:
    """Compute heuristic QoS score and component breakdown for one stream."""
    density_value = max(int(score_input.people_estimate), int(score_input.detect_count))
    density_score = clamp01(density_value / max(tuning.density_reference, 1e-6))

    activity_from_fps = clamp01(score_input.activity_fps / max(tuning.activity_reference_fps, 1e-6))
    activity_from_detect = clamp01(score_input.detect_count / max(tuning.anomaly_detection_reference, 1e-6))
    activity_score = clamp01(0.6 * activity_from_fps + 0.4 * activity_from_detect)

    anomaly_from_detect = clamp01(score_input.detect_count / max(tuning.anomaly_detection_reference, 1e-6))
    anomaly_from_latency = clamp01(score_input.avg_latency_ms / max(tuning.anomaly_latency_ms, 1e-6))
    anomaly_score = clamp01(max(anomaly_from_detect, anomaly_from_latency, score_input.anomaly_hint))

    buffer_pressure = clamp01(score_input.buffer_occupancy / max(tuning.queue_buffer_reference, 1e-6))
    drop_pressure = clamp01(score_input.drop_rate / max(tuning.queue_drop_rate_reference, 1e-6))
    latency_pressure = clamp01(score_input.avg_latency_ms / max(tuning.latency_target_ms, 1e-6))
    queue_pressure_score = clamp01(0.55 * buffer_pressure + 0.30 * drop_pressure + 0.15 * latency_pressure)

    total_score = clamp01(
        weights.density * density_score
        + weights.activity * activity_score
        + weights.anomaly * anomaly_score
        + weights.queue_pressure * queue_pressure_score
    )

    return StreamScoreBreakdown(
        stream_id=score_input.stream_id,
        total_score=total_score,
        density_score=density_score,
        activity_score=activity_score,
        anomaly_score=anomaly_score,
        queue_pressure_score=queue_pressure_score,
    )
