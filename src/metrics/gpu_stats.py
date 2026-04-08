"""GPU statistics collectors for benchmark reports."""

from __future__ import annotations

import subprocess
import time
from typing import Any


def _parse_device_index(device: str | None) -> int:
    if device is None:
        return 0
    normalized = str(device).strip().lower()
    if normalized.startswith("cuda:"):
        try:
            return int(normalized.split(":", 1)[1])
        except (TypeError, ValueError):
            return 0
    return 0


def _query_nvidia_smi_utilization(index: int) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    text = result.stdout.strip().splitlines()
    if not text:
        return None

    try:
        return float(text[0].strip())
    except ValueError:
        return None


def collect_gpu_stats(device: str | None = None) -> dict[str, Any]:
    """Collect lightweight GPU memory/utilization snapshot."""
    payload: dict[str, Any] = {
        "sampled_at": float(time.time()),
        "device": device or "auto",
        "cuda_available": False,
        "gpu_index": None,
        "gpu_name": None,
        "gpu_memory_allocated_mb": 0.0,
        "gpu_memory_reserved_mb": 0.0,
        "gpu_memory_total_mb": 0.0,
        "gpu_utilization_pct": None,
    }

    try:
        import torch
    except Exception:
        return payload

    if not torch.cuda.is_available():
        return payload

    index = _parse_device_index(device)
    try:
        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            return payload
        index = max(0, min(index, device_count - 1))

        props = torch.cuda.get_device_properties(index)
        allocated_mb = torch.cuda.memory_allocated(index) / (1024.0 * 1024.0)
        reserved_mb = torch.cuda.memory_reserved(index) / (1024.0 * 1024.0)
        total_mb = float(props.total_memory) / (1024.0 * 1024.0)

        payload.update(
            {
                "cuda_available": True,
                "gpu_index": index,
                "gpu_name": str(props.name),
                "gpu_memory_allocated_mb": float(allocated_mb),
                "gpu_memory_reserved_mb": float(reserved_mb),
                "gpu_memory_total_mb": float(total_mb),
            }
        )

        util = _query_nvidia_smi_utilization(index)
        if util is not None:
            payload["gpu_utilization_pct"] = float(util)
    except Exception:
        return payload

    return payload
