"""tau / appworld / bfcl adapter contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coevobench.bench.appworld.adapter import AppWorldBench
from coevobench.bench.bfcl.adapter import BfclBench
from coevobench.bench.tau.adapter import TEST_TRIALS, TauBench, TelecomTask


def _write_splits(tmp_path: Path, name: str, sizes: dict[str, int]) -> None:
    ids = {split: [f"{name}-{split}-{i:03d}" for i in range(n)] for split, n in sizes.items()}
    (tmp_path / f"{name}_splits.json").write_text(json.dumps(ids))


def _tau(tmp_path: Path) -> TauBench:
    _write_splits(tmp_path, "tau", {"visible": 23, "selection": 46, "test": 45})
    tasks = {
        f"tau-{split}-{i:03d}": TelecomTask(instruction="Activate the plan.")
        for split, n in (("visible", 23), ("selection", 46), ("test", 45))
        for i in range(n)
    }
    return TauBench(tmp_path, tasks=tasks)


def test_tau_all_checks_rule(tmp_path: Path) -> None:
    bench = _tau(tmp_path)
    all_pass = json.dumps(
        {"checks": [{"name": "a", "passed": True}, {"name": "b", "passed": True}]}
    )
    one_fail = json.dumps(
        {"checks": [{"name": "a", "passed": True}, {"name": "b", "passed": False}]}
    )
    assert bench.judge("tau-test-000", all_pass, None).resolved
    assert not bench.judge("tau-test-000", one_fail, None).resolved
    assert bench.judge("tau-test-000", one_fail, None).exit_status == "ChecksFailed"


def test_tau_format_error_on_garbage(tmp_path: Path) -> None:
    verdict = _tau(tmp_path).judge("tau-test-000", "not json", None)
    assert verdict.exit_status == "FormatError"
    assert verdict.score == 0.0


def test_tau_four_trials_per_test_task(tmp_path: Path) -> None:
    assert TEST_TRIALS == 4
    assert _tau(tmp_path).test_trials() == 4


def _appworld(tmp_path: Path) -> AppWorldBench:
    _write_splits(tmp_path, "appworld", {"visible": 27, "selection": 51, "test": 51})
    tasks = {
        f"appworld-{split}-{i:03d}": {"instruction": "Book the flight."}
        for split, n in (("visible", 27), ("selection", 51), ("test", 51))
        for i in range(n)
    }
    return AppWorldBench(tmp_path, tasks=tasks)


def test_appworld_tgc(tmp_path: Path) -> None:
    bench = _appworld(tmp_path)
    assert bench.judge("a", json.dumps({"tgc": True}), None).resolved
    verdict = bench.judge("a", json.dumps({"tgc": False}), None)
    assert verdict.exit_status == "GoalIncomplete"


def _bfcl(tmp_path: Path) -> BfclBench:
    _write_splits(tmp_path, "bfcl", {"visible": 235, "selection": 470, "test": 560})
    tasks = {
        f"bfcl-{split}-{i:03d}": {"prompt": "Call the tool."}
        for split, n in (("visible", 235), ("selection", 470), ("test", 560))
        for i in range(n)
    }
    return BfclBench(tmp_path, tasks=tasks)


def test_bfcl_component_score_is_eq9(tmp_path: Path) -> None:
    bench = _bfcl(tmp_path)
    records = {
        "multi_turn:miss": [True, True],
        "multi_turn:base": [True, True],
        "multi_turn:turn": [True, True],
        "multi_turn:miss_func": [True, True],
        "memory:backends_a": [True],
        "memory:backends_b": [True],
        "memory:backends_c": [True],
    }
    assert bench.component_score(records) == pytest.approx(1.0)
    # Half of each multi-turn category fails: 0.6 * 0.5 + 0.4 * 1.0
    half = dict(records)
    for key in list(half):
        if key.startswith("multi_turn:"):
            half[key] = [True, False]
    assert bench.component_score(half) == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)


def test_bfcl_missing_component_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both"):
        _bfcl(tmp_path).component_score({"multi_turn:base": [True]})


def test_bfcl_record_judging(tmp_path: Path) -> None:
    bench = _bfcl(tmp_path)
    assert bench.judge("a", json.dumps({"correct": True}), None).score == 1.0
    verdict = bench.judge("a", json.dumps({"correct": "yes"}), None)
    assert verdict.exit_status == "FormatError"
