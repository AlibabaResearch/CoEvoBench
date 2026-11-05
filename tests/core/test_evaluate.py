"""The development evaluation loop: records, badness, and the scalar contract."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.core.evaluation.evaluate import evaluate_split, scalar_feedback
from coevobench.core.protocol.spec import Verdict


def test_record_and_persistence(tmp_path: Path) -> None:
    def judge(task_id: str) -> Verdict:
        return Verdict(resolved=True, score=1.0, exit_status="Submitted")

    record = evaluate_split(
        run_id="glm53_tb_1",
        candidate="cand1",
        split="selection",
        task_ids=["t1", "t2", "t3"],
        judge_fn=judge,
        k=2,
        out_dir=tmp_path / "select_runs",
    )
    assert record.score == 1.0
    assert record.n_bad == 0
    files = list((tmp_path / "select_runs").glob("*.jsonl"))
    assert len(files) == 1
    summary = json.loads(files[0].read_text().splitlines()[0])
    assert summary["split"] == "selection"
    assert summary["k"] == 2
    assert summary["exit_status_counts"] == {"Submitted": 6}


def test_infra_failures_suppress_score(tmp_path: Path) -> None:
    statuses = ["Submitted", "Submitted", "HarnessError:InternalServerError"]

    def judge(task_id: str) -> Verdict:
        status = statuses.pop(0) if statuses else "Submitted"
        return Verdict(resolved=False, score=0.0, exit_status=status)

    record = evaluate_split(
        run_id="r", candidate="c", split="selection",
        task_ids=["a", "b", "c"], judge_fn=judge,
    )
    assert record.n_bad == 1
    assert record.score is None
    feedback = scalar_feedback(record)
    assert feedback == "incomplete: n_bad=1"


def test_dominance_marks_unreliable(tmp_path: Path) -> None:
    # The 2026-09-02 lesson: bad == 0 with a single exit status over 90%
    # of samples is an unreliable run (e.g. serving swallowed the content).
    def judge(task_id: str) -> Verdict:
        return Verdict(resolved=False, score=0.0, exit_status="FormatError")

    record = evaluate_split(
        run_id="r", candidate="c", split="selection",
        task_ids=[f"t{i}" for i in range(20)], judge_fn=judge,
    )
    assert record.n_bad == 0
    assert record.score is None
    assert scalar_feedback(record) == "unreliable: dominant exit status"


def test_scalar_feedback_hides_everything_but_the_number(tmp_path: Path) -> None:
    def judge(task_id: str) -> Verdict:
        resolved = task_id.endswith("1")
        return Verdict(resolved=resolved, score=float(resolved), exit_status="Submitted")

    record = evaluate_split(
        run_id="r", candidate="c", split="selection",
        task_ids=["t0", "t1"], judge_fn=judge,
    )
    feedback = scalar_feedback(record)
    assert feedback.startswith("resolved_rate=0.5")
    # The Selection contract: no task ids, no trajectories, no verdicts.
    assert "t0" not in feedback and "t1" not in feedback
