"""Experiment recorder for reproducible JSON/CSV benchmark outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class ExperimentRecorder:
    """Collect experiment rows and write JSON/CSV artifacts."""

    experiment_name: str
    output_dir: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_record(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[str(key)] = value

    def to_payload(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "generated_at": datetime.now().isoformat(),
            "metadata": dict(self.metadata),
            "records": list(self.records),
        }

    def write_json(self, *, output_path: str | Path | None = None, indent: int = 2) -> Path:
        path = Path(output_path) if output_path is not None else self.output_dir / self.default_filename("json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_payload(), handle, ensure_ascii=False, indent=indent)
        return path

    def write_csv(self, *, output_path: str | Path | None = None) -> Path:
        path = Path(output_path) if output_path is not None else self.output_dir / self.default_filename("csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        flat_rows = [flatten_dict(record) for record in self.records]
        fieldnames = sorted({key for row in flat_rows for key in row.keys()})

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)
        return path

    def default_filename(self, suffix: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}.{suffix}"


def flatten_dict(payload: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict into dotted-key dict for CSV export."""
    output: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(flatten_dict(value, prefix=full_key))
        elif isinstance(value, list):
            output[full_key] = json.dumps(value, ensure_ascii=False)
        else:
            output[full_key] = value
    return output
