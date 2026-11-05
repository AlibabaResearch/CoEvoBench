"""Anonymous task-id management for Selection and Test splits (paper
Appendix E.2: selection task contents and trajectories are withheld).

Stable mapping between real task ids and anonymous ids
(A- + first 8 hex of SHA-256(salt:bench:real_id)). Deterministic, so
operator and eval code agree without sharing state beyond the salt.
Workspace pools carry anonymous ids only; the real-id mapping is
operator-side.

SWE/TB generate mappings from the frozen splits; tau/appworld/bfcl
ship frozen mappings next to their split definitions (protocol
assets).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SALT = "coevobench-anon-2026"


def make_anon_id(real_id: str, bench: str) -> str:
    digest = hashlib.sha256(f"{SALT}:{bench}:{real_id}".encode()).hexdigest()
    return f"A-{digest[:8]}"


def _mapping_path(bench: str) -> Path:
    return Path(__file__).resolve().parent / bench / "anon_mapping.json"


def load_mapping(bench: str) -> dict[str, str] | None:
    """Shipped frozen mapping (real_id -> anon_id), when the bench has one."""
    path = _mapping_path(bench)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and payload:
        first = next(iter(payload.values()))
        if isinstance(first, dict):
            # anon-keyed entries: {A-xxx: {real_id, ...}} -> {real: anon}
            return {entry["real_id"]: anon for anon, entry in payload.items()}
    return payload


def generate_mapping(bench: str, real_ids: list[str]) -> dict[str, str]:
    """Deterministic mapping for benches without a shipped one (swe, tb)."""
    return {real: make_anon_id(real, bench) for real in real_ids}


def anon_to_real(anon_ids: list[str], mapping: dict[str, str]) -> list[str]:
    reverse = {anon: real for real, anon in mapping.items()}
    missing = [a for a in anon_ids if a not in reverse]
    if missing:
        raise ValueError(f"unknown anonymous ids: {missing[:5]}")
    return [reverse[a] for a in anon_ids]


def pool_file_contents(bench: str, mapping: dict[str, str]) -> str:
    """Workspace-safe listing: anonymous ids only, one per line."""
    return "\n".join(sorted(mapping.values())) + "\n"


__all__ = [
    "SALT",
    "anon_to_real",
    "generate_mapping",
    "load_mapping",
    "make_anon_id",
    "pool_file_contents",
]
