"""Runtime heuristic QoS scheduler."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any

from src.scheduler.dispatch import DispatchDecision, make_dispatch_decision
from src.scheduler.policies import ModePolicy, SchedulerPolicy, load_scheduler_policy
from src.scheduler.scoring import StreamScoreInput, clamp01, score_stream


class QoSScheduler:
    """Periodic scheduler that turns stream state into mode actions."""

    def __init__(
        self,
        *,
        policy_config_path: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.policy: SchedulerPolicy = load_scheduler_policy(policy_config_path)

        self._last_tick_monotonic = 0.0
        self._latency_windows: dict[str, deque[float]] = {}
        self._prev_counters: dict[str, tuple[int, int]] = {}
        self._current_mode: dict[str, str] = {}

        if self.policy.tick_sec <= 0:
            raise ValueError("scheduler tick must be > 0")

    @property
    def tick_sec(self) -> float:
        return float(self.policy.tick_sec)

    @property
    def detector_overrides(self) -> dict[str, dict[str, Any]]:
        return self.policy.detector_overrides or {}

    def default_action(self) -> ModePolicy:
        return self.policy.action_for_mode(self.policy.default_mode)

    def get_current_mode(self, stream_id: str) -> str:
        return self._current_mode.get(stream_id, self.policy.default_mode)

    def should_run(self, now_monotonic: float | None = None) -> bool:
        now_value = float(now_monotonic if now_monotonic is not None else time.monotonic())
        if self._last_tick_monotonic <= 0:
            return True
        return (now_value - self._last_tick_monotonic) >= self.policy.tick_sec

    def evaluate(
        self,
        *,
        stream_states: list[dict[str, Any]],
        buffer_capacities: dict[str, int],
        force: bool = False,
    ) -> list[DispatchDecision]:
        now_mono = time.monotonic()
        if not force and not self.should_run(now_mono):
            return []

        self._last_tick_monotonic = now_mono
        decisions: list[DispatchDecision] = []

        for state in stream_states:
            stream_id = str(state.get("stream_id", "")).strip()
            if not stream_id:
                continue

            score_input = self._build_score_input(
                stream_id=stream_id,
                state=state,
                buffer_capacity=buffer_capacities.get(stream_id, 1),
            )
            breakdown = score_stream(
                score_input,
                weights=self.policy.weights,
                tuning=self.policy.tuning,
            )

            prev_mode = self.get_current_mode(stream_id)
            decision = make_dispatch_decision(
                policy=self.policy,
                score_input=score_input,
                score_breakdown=breakdown,
                previous_mode=prev_mode,
            )
            decisions.append(decision)
            self._current_mode[stream_id] = decision.mode
            self._log_decision(decision)

        return decisions

    def _build_score_input(
        self,
        *,
        stream_id: str,
        state: dict[str, Any],
        buffer_capacity: int,
    ) -> StreamScoreInput:
        detect_count = int(state.get("last_detect_count") or 0)
        people_estimate = detect_count

        latency_window = self._latency_windows.setdefault(
            stream_id,
            deque(maxlen=max(1, int(self.policy.score_window_size))),
        )
        latest_latency = state.get("last_infer_latency")
        if isinstance(latest_latency, (int, float)) and float(latest_latency) >= 0:
            latency_window.append(float(latest_latency))

        avg_latency_ms = float(sum(latency_window) / len(latency_window)) if latency_window else 0.0

        drop_count = int(state.get("drop_count") or 0)
        read_count = int(state.get("read_frame_count") or 0)
        drop_rate = self._compute_drop_rate(stream_id, drop_count=drop_count, read_count=read_count)

        cap = max(int(buffer_capacity), 1)
        buffer_size = int(state.get("buffer_size") or 0)
        buffer_occupancy = clamp01(buffer_size / cap)

        activity_fps = float(state.get("read_fps") or 0.0)

        anomaly_hint = 0.0
        if not bool(state.get("online", True)):
            anomaly_hint = max(anomaly_hint, 0.3)
        if drop_rate >= self.policy.tuning.queue_drop_rate_reference:
            anomaly_hint = max(anomaly_hint, 0.5)
        if avg_latency_ms >= self.policy.tuning.anomaly_latency_ms:
            anomaly_hint = max(anomaly_hint, 0.6)
        if detect_count >= self.policy.tuning.anomaly_detection_reference:
            anomaly_hint = max(anomaly_hint, 0.7)

        return StreamScoreInput(
            stream_id=stream_id,
            people_estimate=people_estimate,
            detect_count=detect_count,
            avg_latency_ms=avg_latency_ms,
            drop_rate=drop_rate,
            buffer_occupancy=buffer_occupancy,
            activity_fps=activity_fps,
            anomaly_hint=anomaly_hint,
        )

    def _compute_drop_rate(self, stream_id: str, *, drop_count: int, read_count: int) -> float:
        prev = self._prev_counters.get(stream_id)
        if prev is None:
            self._prev_counters[stream_id] = (drop_count, read_count)
            return 0.0

        prev_drop, prev_read = prev
        delta_drop = max(int(drop_count) - int(prev_drop), 0)
        delta_read = max(int(read_count) - int(prev_read), 0)
        self._prev_counters[stream_id] = (drop_count, read_count)

        denom = delta_drop + delta_read
        if denom <= 0:
            return 0.0
        return clamp01(delta_drop / float(denom))

    def _log_decision(self, decision: DispatchDecision) -> None:
        self.logger.info("qos_decision %s", json.dumps(decision.to_dict(), ensure_ascii=False))
