"""Policy schema and loading for heuristic QoS scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.config import load_config
from src.scheduler.scoring import ScoreTuning, ScoreWeights


@dataclass(slots=True)
class ModePolicy:
    """Action profile bound to one QoS mode."""

    name: str
    input_size: int
    sampling_fps: float
    model_variant: str
    cloud_review: bool

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ModePolicy":
        return cls(
            name=str(name),
            input_size=int(data.get("input_size", 640)),
            sampling_fps=float(data.get("sampling_fps", 5.0)),
            model_variant=str(data.get("model_variant", "main")),
            cloud_review=bool(data.get("cloud_review", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_size": int(self.input_size),
            "sampling_fps": float(self.sampling_fps),
            "model_variant": self.model_variant,
            "cloud_review": bool(self.cloud_review),
        }


@dataclass(slots=True)
class SchedulerThresholds:
    """Mode decision thresholds."""

    eco_max: float
    normal_max: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchedulerThresholds":
        eco_max = float(data.get("eco_max", 0.35))
        normal_max = float(data.get("normal_max", 0.70))
        if not (0.0 <= eco_max <= normal_max <= 1.0):
            raise ValueError("Invalid thresholds: expect 0 <= eco_max <= normal_max <= 1")
        return cls(eco_max=eco_max, normal_max=normal_max)


@dataclass(slots=True)
class SchedulerPolicy:
    """Parsed scheduler policy config used by runtime scheduler."""

    tick_sec: float
    weights: ScoreWeights
    tuning: ScoreTuning
    thresholds: SchedulerThresholds
    modes: dict[str, ModePolicy]
    default_mode: str = "NORMAL"
    score_window_size: int = 6
    detector_overrides: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        required_modes = {"ECO", "NORMAL", "ALERT"}
        missing = required_modes - set(self.modes.keys())
        if missing:
            raise ValueError(f"Missing required modes: {sorted(missing)}")
        if self.default_mode not in self.modes:
            raise ValueError(f"default_mode '{self.default_mode}' is not defined in modes")

    def mode_for_score(self, score: float) -> str:
        value = float(score)
        if value <= self.thresholds.eco_max:
            return "ECO"
        if value <= self.thresholds.normal_max:
            return "NORMAL"
        return "ALERT"

    def action_for_mode(self, mode: str) -> ModePolicy:
        if mode not in self.modes:
            raise KeyError(f"Unknown mode: {mode}")
        return self.modes[mode]


def parse_scheduler_policy(config: dict[str, Any]) -> SchedulerPolicy:
    """Build SchedulerPolicy from merged config dictionary."""
    modes_raw = config.get("modes", {})
    if not isinstance(modes_raw, dict):
        raise ValueError("scheduler config 'modes' must be a mapping")

    modes = {name: ModePolicy.from_dict(name, data) for name, data in modes_raw.items()}
    detector_overrides_raw = config.get("detector_overrides", {})
    detector_overrides = detector_overrides_raw if isinstance(detector_overrides_raw, dict) else {}

    return SchedulerPolicy(
        tick_sec=float(config.get("scheduler_tick_sec", 2.0)),
        weights=ScoreWeights.from_dict(config.get("weights", {})),
        tuning=ScoreTuning.from_dict(config.get("scoring", {})),
        thresholds=SchedulerThresholds.from_dict(config.get("thresholds", {})),
        modes=modes,
        default_mode=str(config.get("default_mode", "NORMAL")).upper(),
        score_window_size=int(config.get("score_window_size", 6)),
        detector_overrides=detector_overrides,
    )


def load_scheduler_policy(config_path: str) -> SchedulerPolicy:
    """Load and parse scheduler policy from YAML path."""
    merged = load_config(config_path)
    return parse_scheduler_policy(merged)
