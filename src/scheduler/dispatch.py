"""Decision dispatch helpers for QoS scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.scheduler.policies import ModePolicy, SchedulerPolicy
from src.scheduler.scoring import StreamScoreBreakdown, StreamScoreInput


@dataclass(slots=True)
class DispatchDecision:
    """One stream scheduling decision for current scheduler tick."""

    stream_id: str
    mode: str
    previous_mode: str
    changed: bool
    score: float
    input_state: dict[str, Any]
    score_breakdown: dict[str, Any]
    action: ModePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "mode": self.mode,
            "previous_mode": self.previous_mode,
            "changed": self.changed,
            "score": float(self.score),
            "input_state": self.input_state,
            "score_breakdown": self.score_breakdown,
            "action": self.action.to_dict(),
        }


def build_input_state_payload(score_input: StreamScoreInput) -> dict[str, Any]:
    """Convert score input dataclass to plain dict for logging/audit."""
    return {
        "stream_id": score_input.stream_id,
        "people_estimate": int(score_input.people_estimate),
        "detect_count": int(score_input.detect_count),
        "avg_latency_ms": float(score_input.avg_latency_ms),
        "drop_rate": float(score_input.drop_rate),
        "buffer_occupancy": float(score_input.buffer_occupancy),
        "activity_fps": float(score_input.activity_fps),
        "anomaly_hint": float(score_input.anomaly_hint),
    }


def make_dispatch_decision(
    *,
    policy: SchedulerPolicy,
    score_input: StreamScoreInput,
    score_breakdown: StreamScoreBreakdown,
    previous_mode: str,
) -> DispatchDecision:
    """Map score output into concrete runtime action."""
    target_mode = policy.mode_for_score(score_breakdown.total_score)
    action = policy.action_for_mode(target_mode)

    return DispatchDecision(
        stream_id=score_input.stream_id,
        mode=target_mode,
        previous_mode=previous_mode,
        changed=(target_mode != previous_mode),
        score=float(score_breakdown.total_score),
        input_state=build_input_state_payload(score_input),
        score_breakdown=score_breakdown.to_dict(),
        action=action,
    )
