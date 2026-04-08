"""Configuration loading utilities for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_OUTPUT_SUBDIRS = ("logs", "runs", "metrics", "videos", "reports")


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


def project_root() -> Path:
    """Return repository root based on the current file location."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to base_dir (or project root if omitted)."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path(base_dir).expanduser() if base_dir is not None else project_root()
    return (root / path).resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise ConfigError(f"Config file not found: {yaml_path}")
    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML format: {yaml_path}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {yaml_path}")
    return data


def deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base and return a new dictionary."""
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge_dict(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load_config(
    config_path: str | Path | None = None,
    *,
    base_config_path: str | Path = DEFAULT_BASE_CONFIG,
) -> dict[str, Any]:
    """Load base config and optionally deep-merge an override config."""
    root = project_root()
    base_path = resolve_path(base_config_path, base_dir=root)
    base_config = read_yaml(base_path)

    active_path = base_path
    if config_path is not None:
        override_path = resolve_path(config_path, base_dir=root)
        active_path = override_path
        if override_path == base_path:
            merged = dict(base_config)
        else:
            override_config = read_yaml(override_path)
            merged = deep_merge_dict(base_config, override_config)
    else:
        merged = dict(base_config)

    merged.setdefault("_meta", {})
    merged["_meta"]["project_root"] = str(root)
    merged["_meta"]["base_config_path"] = str(base_path)
    merged["_meta"]["active_config_path"] = str(active_path)
    return merged


def ensure_output_dirs(config: dict[str, Any], *, create: bool = True) -> dict[str, Any]:
    """
    Normalize and optionally create output directories from config.

    It updates:
    - config["output_dir"] to absolute path
    - config["paths"][<subdir>] to absolute path
    """
    root = Path(config.get("_meta", {}).get("project_root", project_root()))
    output_dir = resolve_path(config.get("output_dir", "outputs"), base_dir=root)
    config["output_dir"] = str(output_dir)

    paths = config.setdefault("paths", {})
    for subdir in DEFAULT_OUTPUT_SUBDIRS:
        configured = paths.get(subdir, output_dir / subdir)
        resolved = resolve_path(configured, base_dir=root)
        paths[subdir] = str(resolved)
        if create:
            resolved.mkdir(parents=True, exist_ok=True)

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return config


def get_required(config: Mapping[str, Any], key: str) -> Any:
    """Get a required key from config or raise ConfigError."""
    if key not in config:
        raise ConfigError(f"Missing required config key: {key}")
    return config[key]
