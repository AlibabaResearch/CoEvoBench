"""Configuration loading with the documented precedence chain.

defaults < ~/.ceb/ceb.toml (user) < ./ceb.toml (project) < CEB_* env < CLI flags
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from coevobench.config.schema import Config, default_user_root

_PROJECT_FILE = "ceb.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("CEB_") or key == "CEB_":
            continue
        path = key.removeprefix("CEB_").lower().split("__")
        node: dict[str, Any] = overrides
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return overrides


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _coerce(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce(v) for v in node]
    if isinstance(node, str):
        if node.lower() in ("true", "false"):
            return node.lower() == "true"
        if node.lstrip("-").isdigit():
            return int(node)
        if node.replace(".", "", 1).lstrip("-").isdigit():
            return float(node)
    return node


def load_config(project_dir: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load configuration following the precedence chain."""
    cwd = project_dir or Path.cwd()
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, _load_toml(default_user_root() / _PROJECT_FILE))
    merged = _deep_merge(merged, _load_toml(cwd / _PROJECT_FILE))
    merged = _deep_merge(merged, _coerce(_env_overrides()))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config.model_validate(merged)
