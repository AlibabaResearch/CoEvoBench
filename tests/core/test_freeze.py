"""Freeze gates and tamper detection."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.core.grading.freeze import freeze, validate, verify_unchanged, weights_fingerprint


def _canonical_transport() -> bytes:
    """Copy the real transport bytes from the shipped template."""
    for bench in ("swe", "tb"):
        for path in (Path(__file__).parents[2] / "templates" / bench / "harness").rglob(
            "_transport.py"
        ):
            return path.read_bytes()
    raise AssertionError("canonical transport template missing")


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "out" / "cand1"
    (root / "harness" / "models").mkdir(parents=True)
    (root / "harness" / "models" / "_transport.py").write_bytes(_canonical_transport())
    (root / "harness" / "agent.py").write_text("def run(): pass\n")
    (root / "model_path.txt").write_text("/nonexistent/weights\n")
    (root / "meta.json").write_text(json.dumps({"label": "cand1"}))
    return root


def test_freeze_ok_and_locks_tree(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    report = freeze(root, "cand1")
    assert report.ok, report.gate_errors
    assert report.n_files == 4  # freeze.json itself is excluded from the manifest
    assert (root / "freeze.json").is_file()


def test_freeze_rejects_unmodified_seed(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = _hash_dir(root / "harness")
    import hashlib

    seed_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    errors = validate(root, initial_harness_sha=seed_sha)
    assert any("byte-identical" in e for e in errors)


def test_freeze_rejects_missing_pieces(tmp_path: Path) -> None:
    root = tmp_path / "out" / "broken"
    root.mkdir(parents=True)
    errors = validate(root)
    assert len(errors) == 3


def test_tamper_detection(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    freeze(root, "cand1")
    assert verify_unchanged(root)
    # freeze.json itself is part of the tree; appending a new file changes
    # the manifest and must be caught
    (root / "smuggled.txt").write_text("late edit")
    assert not verify_unchanged(root)


def test_weights_fingerprint_detects_change(tmp_path: Path) -> None:
    import os

    weights = tmp_path / "weights"
    weights.mkdir()
    shard = weights / "model-00001-of-00001.safetensors"
    shard.write_bytes(os.urandom(1 << 20))
    fp1 = weights_fingerprint(weights)
    # flip one byte in the middle of the shard
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))
    fp2 = weights_fingerprint(weights)
    assert fp1 != fp2
    assert weights_fingerprint(tmp_path) == "no-shards"


def _hash_dir(root: Path) -> dict[str, str]:
    import hashlib

    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_tampered_transport_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    transport = root / "harness" / "models" / "_transport.py"
    transport.write_bytes(_canonical_transport() + b"\n# tampered\n")
    errors = validate(root)
    assert any("canonical" in e for e in errors)


def test_missing_transport_is_rejected(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "harness" / "models" / "_transport.py").unlink()
    errors = validate(root)
    assert any("_transport.py" in e for e in errors)
