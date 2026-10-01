"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with a stable, human-readable formatting."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    return path_obj
