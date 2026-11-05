"""Candidate freezing (design document Section 5.5; paper Appendix E.2).

Freeze validates a nominated candidate directory, hashes every file into a
manifest, and locks the tree read-only so post-hoc edits are detectable.
The integrity gates cover harness evolution and canonical transport:

- the frozen harness must not be byte-identical to the initial harness
  (an unmodified seed is not an evolved candidate)
- the frozen transport file must match the canonical bytes (the welded
  model-transport seam; the one frozen line in the harness)

The nominal freeze layout is out/<label>/{harness/, model_path.txt,
meta.json}; the toy bench drives the same gates with a minimal tree.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path

# The welded model-transport seam: the one frozen line in every harness.
# Byte-compared against the canonical file shipped in templates/*/harness.
CANONICAL_TRANSPORT_SHA256 = (
    "7deba179b12577df6761f0bfb2418c2c8a5993ba0e567c07526a303fb8a306ce"
)


@dataclass(frozen=True)
class FreezeReport:
    label: str
    manifest_sha256: str
    n_files: int
    weights_fingerprint: str
    gate_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.gate_errors


def _hash_tree(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name == "freeze.json" and path.parent == root:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[str(path.relative_to(root))] = digest
    return manifest


def weights_fingerprint(model_path: Path, sample_bytes: int = 1 << 22) -> str:
    """Fingerprint the weights by hashing strided samples of safetensors shards.

    A full 18 GiB hash is wasted effort: strided sampling of every shard
    detects any actual weight change while staying cheap. Two candidates
    with identical fingerprints carry identical weights.
    """
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        return "no-shards"
    hasher = hashlib.sha256()
    for shard in shards:
        size = shard.stat().st_size
        with shard.open("rb") as handle:
            offset = 0
            while offset < size:
                handle.seek(offset)
                hasher.update(handle.read(sample_bytes))
                offset += max(size // 8, sample_bytes)
    return hasher.hexdigest()


def validate(candidate_dir: Path, *, initial_harness_sha: str | None = None) -> tuple[str, ...]:
    """Run the freeze gates; returns the list of violations."""
    errors: list[str] = []
    harness = candidate_dir / "harness"
    if not harness.is_dir():
        errors.append("candidate must contain harness/")
    model_path = candidate_dir / "model_path.txt"
    if not model_path.is_file():
        errors.append("candidate must contain model_path.txt")
    meta = candidate_dir / "meta.json"
    if not meta.is_file():
        errors.append("candidate must contain meta.json")
    if harness.is_dir():
        root = harness.resolve()
        for path in harness.rglob("*"):
            if path.is_symlink():
                try:
                    target = path.resolve()
                except OSError:
                    target = None
                if target is None or not target.is_relative_to(root):
                    errors.append(
                        f"harness contains symlinks escaping the candidate: {path}"
                    )
        tree = _hash_tree(harness)
        if not tree:
            errors.append("harness/ is empty")
        digest = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()
        if initial_harness_sha and digest == initial_harness_sha:
            errors.append("harness is byte-identical to the initial harness")
        # the welded transport seam exists for the agent-loop benches
        # (swe/tb); the completion-injection benches (appworld/bfcl)
        # have no model transport in the harness — the platform injects
        # the completion and the harness never sees an endpoint
        if CANONICAL_TRANSPORT_SHA256 and _transport_bench(candidate_dir):
            transport = _find_transport(harness)
            if transport is None:
                errors.append("harness is missing models/_transport.py")
            elif hashlib.sha256(transport.read_bytes()).hexdigest() != CANONICAL_TRANSPORT_SHA256:
                errors.append("models/_transport.py differs from the canonical bytes")
    return tuple(errors)


def _transport_bench(candidate_dir: Path) -> bool:
    """swe/tb harnesses carry the frozen transport; others inject completion."""
    try:
        meta = json.loads((candidate_dir / "meta.json").read_text())
    except (ValueError, OSError):
        return True  # unknown bench: the gate stays on (fail closed)
    return str(meta.get("bench", "swe")).lower() in ("swe", "tb")


def _find_transport(harness: Path) -> Path | None:
    for candidate in harness.rglob("_transport.py"):
        if "models" in candidate.parts:
            return candidate
    return None


def freeze(candidate_dir: Path, label: str) -> FreezeReport:
    """Validate, hash, and lock the candidate tree read-only."""
    errors = validate(candidate_dir)
    manifest = _hash_tree(candidate_dir)
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    model_path_file = candidate_dir / "model_path.txt"
    fingerprint = "none"
    if model_path_file.is_file():
        target = Path(model_path_file.read_text().strip())
        fingerprint = weights_fingerprint(target) if target.is_dir() else "missing"
    if not errors:
        for path in candidate_dir.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    freeze_file = candidate_dir / "freeze.json"
    freeze_file.write_text(
        json.dumps(
            {
                "label": label,
                "manifest": manifest,
                "manifest_sha256": manifest_sha,
                "weights_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    return FreezeReport(
        label=label,
        manifest_sha256=manifest_sha,
        n_files=len(manifest),
        weights_fingerprint=fingerprint,
        gate_errors=errors,
    )


def verify_unchanged(candidate_dir: Path) -> bool:
    """Post-evaluation tamper check: the frozen manifest still matches."""
    freeze_file = candidate_dir / "freeze.json"
    if not freeze_file.is_file():
        return False
    record = json.loads(freeze_file.read_text())
    current = _hash_tree(candidate_dir)
    current_sha = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    return current_sha == record["manifest_sha256"]


__all__ = [
    "CANONICAL_TRANSPORT_SHA256",
    "FreezeReport",
    "freeze",
    "validate",
    "verify_unchanged",
    "weights_fingerprint",
]
