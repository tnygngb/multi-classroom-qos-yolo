from __future__ import annotations

from pathlib import Path

from src.common.config import ensure_output_dirs, load_config


def test_load_config_from_base_file() -> None:
    config = load_config("configs/base.yaml")
    assert config["project_name"] == "multi-classroom-qos-yolo"
    assert "output_dir" in config
    assert "_meta" in config


def test_ensure_output_dirs_creates_required_paths(tmp_path: Path) -> None:
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "paths": {},
    }
    updated = ensure_output_dirs(config, create=True)
    assert Path(updated["output_dir"]).exists()
    assert Path(updated["paths"]["logs"]).exists()
