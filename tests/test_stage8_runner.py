from __future__ import annotations

import json
import uuid
from pathlib import Path

from scripts import run_experiments


def _temp_root(tag: str) -> Path:
    root = Path("outputs") / "test_tmp" / f"{tag}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_normalize_rows_and_ablation_logic() -> None:
    single_payload = {
        "records": [
            {"variant": "baseline", "fps": 20.0, "mAP@0.5": 0.60},
            {"variant": "enhanced", "fps": 16.0, "mAP@0.5": 0.66},
        ]
    }
    single_rows = run_experiments._normalize_single_rows(single_payload)
    assert len(single_rows) == 2

    multi_rows = [
        {"strategy": "static_low", "stream_count": 4, "total_fps": 12.0, "p95_latency_ms": 60.0},
        {"strategy": "qos", "stream_count": 4, "total_fps": 13.5, "p95_latency_ms": 57.0},
    ]

    p2 = run_experiments._compute_p2_ablation_rows(single_rows)
    assert len(p2) == 2
    assert p2[1]["delta_fps_vs_without"] == -4.0

    scheduler = run_experiments._compute_scheduler_ablation_rows(multi_rows, [4])
    assert len(scheduler) == 2
    assert scheduler[1]["delta_fps_vs_without"] == 1.5


def test_collect_qos_mode_distribution() -> None:
    tmp_root = _temp_root("stage8_modes")
    jsonl_path = tmp_root / "qos.jsonl"
    lines = [
        {"mode": "ECO"},
        {"mode": "ECO"},
        {"mode": "NORMAL"},
        {"mode": "ALERT"},
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row) + "\n")

    payload = {
        "records": [
            {
                "strategy": "qos",
                "stream_count": 4,
                "results_jsonl": str(jsonl_path),
            }
        ]
    }

    dist = run_experiments._collect_qos_mode_distribution(payload)
    assert 4 in dist
    assert abs(dist[4]["ECO"] - 0.5) < 1e-6
    assert abs(dist[4]["NORMAL"] - 0.25) < 1e-6
    assert abs(dist[4]["ALERT"] - 0.25) < 1e-6


def test_run_stage8_dry_run_generates_manifest() -> None:
    root = _temp_root("stage8")
    result = run_experiments.run_stage8(
        "configs/experiments/stage8_paper.yaml",
        mode="all",
        output_dir_arg=str(root),
        dry_run=True,
    )

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 8
    assert payload["dry_run"] is True
    assert isinstance(payload["steps"], list)
    assert len(payload["steps"]) >= 3
