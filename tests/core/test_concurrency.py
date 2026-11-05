"""Concurrent episode driver: crash discipline, ordering, progress."""

from __future__ import annotations

import threading
import time

import pytest

from coevobench.core.evaluation.concurrency import run_concurrent
from coevobench.core.protocol.spec import Verdict


def test_concurrent_runs_all_jobs_and_sorts() -> None:
    def judge(task_id: str, sample: int) -> Verdict:
        time.sleep(0.01)
        return Verdict(resolved=True, score=1.0, exit_status="Submitted")

    jobs = [(f"t{i}", 0) for i in range(10)]
    results = run_concurrent(jobs, judge, workers=4)
    assert len(results) == 10
    assert all(r.resolved for r in results)
    # determinism: concurrent results match the verdict set
    assert {r.exit_status for r in results} == {"Submitted"}


def test_concurrent_crashed_job_becomes_bad_record() -> None:
    def judge(task_id: str, sample: int) -> Verdict:
        if task_id == "boom":
            raise RuntimeError("exploded")
        return Verdict(resolved=True, score=1.0, exit_status="Submitted")

    jobs = [("ok", 0), ("boom", 0), ("ok", 1)]
    results = run_concurrent(jobs, judge, workers=2)
    assert len(results) == 3
    bad = [r for r in results if r.exit_status.startswith("WorkerError")]
    assert len(bad) == 1
    assert bad[0].exit_status.startswith("WorkerError:RuntimeError")
    assert bad[0].resolved is False
    assert bad[0].score == 0.0


def test_concurrent_rejects_duplicate_jobs() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_concurrent([("t", 0), ("t", 0)], lambda a, b: None, workers=1)


def test_concurrent_progress_callback() -> None:
    seen: list[int] = []
    lock = threading.Lock()

    def on_progress(done: int, total: int, record: dict) -> None:
        with lock:
            seen.append(done)

    run_concurrent(
        [("t1", 0), ("t2", 0)], lambda a, b: Verdict(True, 1.0, "Submitted"),
        workers=2, on_progress=on_progress,
    )
    assert sorted(seen) == [1, 2]


def test_evaluate_split_concurrent_matches_serial() -> None:
    from coevobench.core.evaluation.evaluate import evaluate_split

    def judge(task_id: str) -> Verdict:
        return Verdict(resolved=True, score=1.0, exit_status="Submitted")

    serial = evaluate_split(
        run_id="serial", candidate="c", split="visible",
        task_ids=["a", "b", "c"], judge_fn=judge, k=2,
    )
    concurrent = evaluate_split(
        run_id="conc", candidate="c", split="visible",
        task_ids=["a", "b", "c"], judge_fn=judge, k=2, workers=3,
    )
    assert serial.score == concurrent.score
    assert serial.n_bad == concurrent.n_bad
    assert serial.exit_status_counts == concurrent.exit_status_counts


def test_evaluate_split_concurrent_crashes_count_bad() -> None:
    from coevobench.core.evaluation.evaluate import evaluate_split

    def judge(task_id: str) -> Verdict:
        if task_id == "die":
            raise RuntimeError("boom")
        return Verdict(resolved=True, score=1.0, exit_status="Submitted")

    record = evaluate_split(
        run_id="conc-bad", candidate="c", split="visible",
        task_ids=["ok1", "die", "ok2"], judge_fn=judge, k=1, workers=2,
    )
    assert record.n_bad >= 1
    assert "WorkerError:RuntimeError" in record.exit_status_counts
    # infra failures suppress the score
    assert record.score is None
