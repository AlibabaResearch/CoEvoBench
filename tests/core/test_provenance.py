"""Weights provenance: the four gates."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.core.grading.provenance import (
    check_derivation,
    check_identity,
    check_location,
    check_recipe,
    verify,
)

CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "head_dim": 128,
}


def _checkpoint(root: Path, payload: bytes = b"\x01" * 2048) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(CONFIG))
    (root / "model.safetensors").write_bytes(payload)
    return root


def _submission(tmp_path: Path, weights: Path, with_recipe: bool = True) -> Path:
    submission = tmp_path / "out" / "cand"
    submission.mkdir(parents=True)
    (submission / "model_path.txt").write_text(str(weights) + "\n")
    (submission / "meta.json").write_text("{}")
    if with_recipe:
        train = submission / "train"
        train.mkdir()
        (train / "run.sh").write_text("#!/bin/sh\npython train.py\n")
        (train / "DATA_PROVENANCE.md").write_text("sources: public SWE trajectories\n")
    return submission


def test_identity_matches_and_detects_swap(tmp_path: Path) -> None:
    base = _checkpoint(tmp_path / "base")
    twin = _checkpoint(tmp_path / "twin")
    assert check_identity(twin, base)[0]

    swapped = _checkpoint(tmp_path / "swapped")
    config = dict(CONFIG, hidden_size=8192)
    (swapped / "config.json").write_text(json.dumps(config))
    ok, detail = check_identity(swapped, base)
    assert not ok and "hidden_size" in detail


def test_identity_understands_multimodal_nesting(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": CONFIG})
    )
    base = _checkpoint(tmp_path / "base")
    assert check_identity(nested, base)[0]


def test_derivation_requires_difference(tmp_path: Path) -> None:
    base = _checkpoint(tmp_path / "base", b"\x01" * 2048)
    same = _checkpoint(tmp_path / "same", b"\x01" * 2048)
    trained = _checkpoint(tmp_path / "trained", b"\x02" * 2048)
    assert not check_derivation(same, base)[0]
    assert "nothing trained" in check_derivation(same, base)[1]
    assert check_derivation(trained, base)[0]


def test_location_gates_paths(tmp_path: Path) -> None:
    inside = tmp_path / "ws" / "ckpt"
    inside.mkdir(parents=True)
    outside = tmp_path / "downloads" / "model"
    outside.mkdir(parents=True)
    roots = [tmp_path / "ws"]
    assert check_location(inside, roots)[0]
    ok, detail = check_location(outside, roots)
    assert not ok and "outside" in detail


def test_recipe_structure(tmp_path: Path) -> None:
    weights = _checkpoint(tmp_path / "w")
    full = _submission(tmp_path, weights)
    assert check_recipe(full)[0]
    bare = _submission(tmp_path / "bare", weights, with_recipe=False)
    ok, detail = check_recipe(bare)
    assert not ok and "no train/" in detail


def test_verify_all_gates_pass(tmp_path: Path) -> None:
    base = _checkpoint(tmp_path / "base", b"\x01" * 2048)
    weights = _checkpoint(tmp_path / "ws" / "trained", b"\x02" * 2048)
    submission = _submission(tmp_path, weights)
    report = verify(submission, base, [tmp_path / "ws"])
    assert report.ok, report.failures


def test_verify_reports_each_failure(tmp_path: Path) -> None:
    base = _checkpoint(tmp_path / "base", b"\x01" * 2048)
    untrained = _checkpoint(tmp_path / "ws" / "untrained", b"\x01" * 2048)
    submission = _submission(tmp_path, untrained, with_recipe=False)
    report = verify(submission, base, [tmp_path / "ws"])
    assert not report.ok
    names = {name for name, _, _ in report.gates}
    assert "derivation" in names and "recipe" in names


def test_verify_rejects_missing_checkpoint(tmp_path: Path) -> None:
    submission = tmp_path / "out" / "ghost"
    submission.mkdir(parents=True)
    (submission / "model_path.txt").write_text("/nowhere\n")
    report = verify(submission, tmp_path / "base")
    assert not report.ok
    assert "full checkpoint" in report.failures[0]


def test_operator_control_skips_recipe(tmp_path: Path) -> None:
    base = _checkpoint(tmp_path / "base", b"\x01" * 2048)
    weights = _checkpoint(tmp_path / "ws" / "ctrl", b"\x02" * 2048)
    submission = _submission(tmp_path, weights, with_recipe=False)
    report = verify(submission, base, [tmp_path / "ws"], require_recipe=False)
    assert report.ok, report.failures
