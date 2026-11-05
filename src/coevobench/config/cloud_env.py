"""Load operator credentials as data, never execute a shell configuration file."""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, raw = line.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid env assignment on line {number}")
        parts = shlex.split(raw, comments=True)
        if len(parts) > 1:
            raise ValueError(f"quote env value on line {number}")
        values[key] = parts[0] if parts else ""
    return values


def load_env(path: Path) -> None:
    for key, value in read_env(path).items():
        # Blank template credentials must not shadow Modal's active local profile.
        if key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") and not value:
            continue
        os.environ.setdefault(key, value)


def update_env(path: Path, updates: dict[str, str]) -> None:
    values = read_env(path)
    values.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        for key, value in values.items():
            stream.write(f"{key}={shlex.quote(value)}\n")
    try:
        temp.chmod(0o600)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
