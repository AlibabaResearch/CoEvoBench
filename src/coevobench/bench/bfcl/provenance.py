"""Reconstruction provenance, distinct from the missing historical manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
VENDOR_ROOT = (
    Path(__file__).resolve().parents[4] / "vendor/gorilla/berkeley-function-call-leaderboard"
)
MANIFEST = Path(__file__).with_name("upstream_manifest.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream(root: Path = VENDOR_ROOT) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("unexpected BFCL upstream revision")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"BFCL frozen source/data changed: {name}")
    actual = {str(p.relative_to(root)) for p in (root / "bfcl_eval").rglob("*.py")}
    expected_py = {name for name in manifest["files"] if name.endswith(".py")}
    if actual != expected_py:
        raise ValueError("BFCL source file inventory changed")
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "manifest_sha256": digest(MANIFEST),
        "implementation": "reconstructed",
        "historical_equivalence": "unverified",
    }
