from __future__ import annotations

from src.edge.node import EdgeNode
from src.scheduler.policies import load_scheduler_policy
from src.scheduler.qos_scheduler import QoSScheduler
from src.scheduler.scoring import ScoreTuning, ScoreWeights, StreamScoreInput, score_stream


def test_scheduler_policy_loads_modes_and_thresholds() -> None:
    policy = load_scheduler_policy("configs/scheduler/qos_policy.yaml")

    assert set(policy.modes.keys()) == {"ECO", "NORMAL", "ALERT"}
    assert policy.mode_for_score(0.10) == "ECO"
    assert policy.mode_for_score(0.50) == "NORMAL"
    assert policy.mode_for_score(0.90) == "ALERT"


def test_score_stream_increases_under_pressure() -> None:
    weights = ScoreWeights.from_dict(
        {
            "density": 0.35,
            "activity": 0.20,
            "anomaly": 0.25,
            "queue_pressure": 0.20,
        }
    )
    tuning = ScoreTuning()

    low_input = StreamScoreInput(
        stream_id="room_101",
        people_estimate=1,
        detect_count=1,
        avg_latency_ms=15,
        drop_rate=0.0,
        buffer_occupancy=0.1,
        activity_fps=3,
    )
    high_input = StreamScoreInput(
        stream_id="room_101",
        people_estimate=30,
        detect_count=30,
        avg_latency_ms=250,
        drop_rate=0.5,
        buffer_occupancy=1.0,
        activity_fps=15,
        anomaly_hint=0.8,
    )

    low_score = score_stream(low_input, weights=weights, tuning=tuning)
    high_score = score_stream(high_input, weights=weights, tuning=tuning)

    assert high_score.total_score > low_score.total_score
    assert high_score.queue_pressure_score >= low_score.queue_pressure_score


def test_qos_scheduler_can_switch_to_alert_mode() -> None:
    scheduler = QoSScheduler(policy_config_path="configs/scheduler/qos_policy.yaml")

    low_state = [
        {
            "stream_id": "room_101",
            "online": True,
            "last_detect_count": 0,
            "last_infer_latency": 20.0,
            "drop_count": 0,
            "read_frame_count": 100,
            "buffer_size": 1,
            "read_fps": 4.0,
        }
    ]
    first = scheduler.evaluate(
        stream_states=low_state,
        buffer_capacities={"room_101": 32},
        force=True,
    )
    assert len(first) == 1

    high_state = [
        {
            "stream_id": "room_101",
            "online": True,
            "last_detect_count": 40,
            "last_infer_latency": 260.0,
            "drop_count": 30,
            "read_frame_count": 120,
            "buffer_size": 32,
            "read_fps": 14.0,
        }
    ]
    second = scheduler.evaluate(
        stream_states=high_state,
        buffer_capacities={"room_101": 32},
        force=True,
    )

    assert second[0].mode == "ALERT"
    assert second[0].action.model_variant == "enhanced"
    assert second[0].action.cloud_review is True


def test_edge_node_initializes_qos_runtime_profile() -> None:
    node = EdgeNode(
        streams_config_path="configs/streams/demo_4streams.yaml",
        detector_config_path="configs/detector/yolov8_head.yaml",
        scheduler_config_path="configs/scheduler/qos_policy.yaml",
        enable_qos=True,
    )

    snapshot = node.source_manager.get_state_snapshot()
    assert len(snapshot) == 4
    assert all(item["current_mode"] == "NORMAL" for item in snapshot)
    assert all(item["current_model_variant"] == "main" for item in snapshot)
    assert all(item["current_input_size"] == 960 for item in snapshot)
