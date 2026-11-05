"""Pinned operator-side AppWorld installation and data locations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
UPSTREAM_COMMIT = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
DATA_VERSION = "0.2.0"
SOURCE = PROJECT / "vendor/appworld"


def python_path() -> Path:
    return Path(os.environ.get("CEB_APPWORLD_PYTHON", PROJECT / ".venv-appworld/bin/python"))


def data_root() -> Path:
    return Path(os.environ.get("CEB_APPWORLD_ROOT", PROJECT / ".local/appworld")).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source() -> dict:
    if not python_path().is_file():
        raise ValueError("AppWorld environment missing; run scripts/setup-appworld.py")
    head = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError("AppWorld upstream commit differs from the pinned protocol version")
    dirty = subprocess.check_output(
        ["git", "-C", str(SOURCE), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if dirty:
        raise ValueError("AppWorld tracked upstream files have local changes")
    manifest = PROJECT / "src/coevobench/bench/appworld/upstream_manifest.json"
    if not manifest.is_file():
        raise ValueError("AppWorld runtime manifest is missing")
    for name, expected in json.loads(manifest.read_text())["files"].items():
        path = SOURCE / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"AppWorld runtime file differs: {name}")
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "data_version": DATA_VERSION,
        "runtime_manifest_sha256": sha256(manifest),
        "implementation": "reconstructed",
        "historical_equivalence": "unverified",
    }
