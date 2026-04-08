"""Time-related helper functions."""

from __future__ import annotations

from datetime import datetime, timezone


def now_ts() -> float:
    """Return current UNIX timestamp in seconds."""
    return datetime.now(tz=timezone.utc).timestamp()


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()
