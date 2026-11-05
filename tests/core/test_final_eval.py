"""Final evaluation: bundle preparation and Test-split replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from coevobench.bench.toy.adapter import SPLITS as TOY_SPLITS
from coevobench.core.lifecycle.final_eval import evaluate, prepare, summarize_rows
from coevobench.core.protocol.spec import Verdict
from coevobench.core.protocol.splits import from_mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.test_freeze import _candidate


@pytest.fixture
def submission(tmp_path: Path) -> Path:
    root = _candidate(tmp_path)
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text('{"model_type": "qwen3_5"}')
    (weights / "model.safetensors").write_bytes(b"\x00" * 1024)
    (root / "model_path.txt").write_text(str(weights) + "\n")
    (root / "serve_config.json").write_text(json.dumps({"max_model_len": 32768}))
    return root


def test_prepare_builds_a_self_contained_bundle(tmp_path: Path, submission: Path) -> None:
    bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")
    assert (bundle / "submission" / "harness").is_dir()
    assert (bundle / "weights" / "config.json").is_file()
    # model_path rewritten to the bundle's own copy: evaluation never
    # reads the researcher workspace
    rewritten = (bundle / "submission" / "model_path.txt").read_text().strip()
    assert rewritten == str(bundle / "weights")
    assert (bundle / "evaluation.json").is_file()


def test_prepare_refuses_workspace_nested_destination(
    tmp_path: Path, submission: Path
) -> None:
    with pytest.raises(ValueError, match="outside the submission"):
        prepare(submission, "tb", submission / "inside")


def test_prepare_rejects_failed_gates(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(ValueError, match="freeze gates"):
        prepare(broken, "tb", tmp_path / "final")


def test_prepare_is_idempotent_hostile(tmp_path: Path, submission: Path) -> None:
    destination = tmp_path / "final"
    prepare(submission, "tb", destination, label="cand1")
    with pytest.raises(ValueError, match="bundle already exists"):
        prepare(submission, "tb", destination, label="cand1")


def test_evaluate_replays_and_validates(tmp_path: Path, submission: Path) -> None:
    bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")
    splits = from_mapping(TOY_SPLITS)

    def judge(task_id: str) -> Verdict:
        resolved = task_id.endswith("5")
        return Verdict(resolved=resolved, score=float(resolved), exit_status="Submitted")

    # toy split's test has toy-sum-4 and toy-sum-5 -> 1/2 resolved
    report = evaluate(bundle, bench="tb", splits=splits, judge_fn=judge)
    assert report.n_tasks == 2
    assert report.score == pytest.approx(0.5)
    assert report.reverted is False
    parsed = json.loads(report.to_json())
    assert parsed["metric"] == "score_mean"


def test_evaluate_refuses_tampered_bundle(tmp_path: Path, submission: Path) -> None:
    bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")
    (bundle / "submission" / "smuggled.txt").write_text("late edit")
    with pytest.raises(ValueError, match="changed after freezing"):
        evaluate(bundle, bench="tb", splits=from_mapping(TOY_SPLITS), judge_fn=lambda t: None)


def test_evaluate_refuses_bench_mismatch(tmp_path: Path, submission: Path) -> None:
    bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")
    with pytest.raises(ValueError, match="for bench"):
        evaluate(
            bundle,
            bench="swe",
            splits=from_mapping(TOY_SPLITS),
            judge_fn=lambda t: Verdict(True, 1.0, "Submitted"),
        )


def test_evaluate_infra_failures_are_not_scores(tmp_path: Path, submission: Path) -> None:
    bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")

    def judge(task_id: str) -> Verdict:
        return Verdict(resolved=False, score=0.0, exit_status="HarnessError:InternalServerError")

    with pytest.raises(ValueError, match="infrastructure failures"):
        evaluate(bundle, bench="tb", splits=from_mapping(TOY_SPLITS), judge_fn=judge)


def test_weight_reversion_marks_the_report(tmp_path: Path, submission: Path) -> None:
    bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")
    report = evaluate(
        bundle,
        bench="tb",
        splits=from_mapping(TOY_SPLITS),
        judge_fn=lambda t: Verdict(True, 1.0, "Submitted"),
        weights_override=Path("/base/weights"),
    )
    assert report.reverted is True
    assert report.weights == "/base/weights"


def test_summarize_rows_completeness() -> None:
    rows = [
        {"task_id": "a", "sample": 0, "score": 1.0, "exit_status": "Submitted"},
        {"task_id": "b", "sample": 0, "score": 0.0, "exit_status": "Submitted"},
    ]
    assert summarize_rows(rows, "tb")["n_rows"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows(rows + [dict(rows[0])], "tb")
    with pytest.raises(ValueError, match="infra failure"):
        bad = [dict(rows[0], exit_status="HarnessError:APIConnectionError")]
        summarize_rows(bad, "tb")
    with pytest.raises(ValueError, match="out of range"):
        summarize_rows([dict(rows[0], score=1.5)], "tb")


def test_replay_registry_resolves_the_bench_binding(tmp_path: Path, submission: Path) -> None:
    """judge_fn=None resolves through the bench layer's registered binding."""
    from coevobench.bench.swe.replay import replay_judge
    from coevobench.core.lifecycle.final_eval import _REPLAY_BINDINGS, register_replay_binding

    register_replay_binding("swe", replay_judge)
    try:
        bundle = prepare(submission, "tb", tmp_path / "final", label="cand1")
        # tb has no registered binding: a loud, actionable error.
        with pytest.raises(ValueError, match="not registered"):
            evaluate(bundle, bench="tb", splits=from_mapping(TOY_SPLITS))
        assert "swe" in _REPLAY_BINDINGS
    finally:
        _REPLAY_BINDINGS.pop("swe", None)
