from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置，并将相对路径解析为相对 config 文件的绝对路径。"""

    path = Path(config_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["_config_path"] = str(path)
    config["_config_root"] = str(path.parent)

    for section_name in ("paths", "demo"):
        section = config.get(section_name)
        if isinstance(section, dict):
            config[section_name] = _resolve_path_like_values(section, path.parent)

    return config


def _resolve_path_like_values(section: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in section.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_path_like_values(value, base_dir)
            continue
        if isinstance(value, str) and _looks_like_path(key, value):
            resolved[key] = str((base_dir / value).resolve()) if not Path(value).is_absolute() else value
        else:
            resolved[key] = value
    return resolved


def _looks_like_path(key: str, value: str) -> bool:
    key_lower = key.lower()
    return (
        key_lower.endswith("_path")
        or key_lower.endswith("_root")
        or key_lower.endswith("_file")
        or "/" in value
        or value.startswith(".")
    )
