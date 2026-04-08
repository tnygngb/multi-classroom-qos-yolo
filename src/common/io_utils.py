"""I/O helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for a file path if needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON payload to file."""
    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    """Read JSON payload from file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
