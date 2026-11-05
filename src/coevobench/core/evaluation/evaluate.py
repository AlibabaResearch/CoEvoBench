"""Development evaluation loop (design document Sections 5.2, 8.5).

Runs a candidate over a split's tasks through a judge callable, records
per-task verdicts and the exit-status distribution, and persists an
EvalRecord. Selection runs write records to the operator-side registry
directory (never mounted into a run envelope); the researcher receives
only the scalar summary.

The engine is bench-agnostic: the caller supplies judge_fn and the task
list. Infra failures (n_bad) suppress the score — an incomplete run
reports None rather than a misleading number.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from coevobench.core.protocol.exit_status import counts_summary, is_dominant, n_bad
from coevobench.core.protocol.spec import EvalRecord, Verdict

JudgeFn = Callable[[str], Verdict]


def evaluate_split(
    *,
    run_id: str,
    candidate: str,
    split: str,
    task_ids: list[str],
    judge_fn: JudgeFn,
    k: int = 1,
    seed: int = 0,
    out_dir: Path | None = None,
    workers: int = 1,
) -> EvalRecord:
    """Evaluate every (task, sample) pair; return the persisted record.

    workers > 1 runs jobs concurrently through the episode driver
    (crash discipline: WorkerError records count as bad, never a
    whole-evaluation crash, never a silent zero).
    """
    samples: list[str] = []
    verdicts: list[Verdict] = []
    started = time.time()
    # A judge may own its batching (serve once, episodes pooled, k
    # repeats managed internally) — evaluate_batch takes the whole
    # (task_ids, k, workers) contract and returns every verdict.
    batch = getattr(judge_fn, "evaluate_batch", None)
    if batch is not None:
        verdicts = batch(task_ids, k, workers)
        if len(verdicts) != len(task_ids) * k:
            raise ValueError("batch judge returned an incomplete trial set")
        samples = [v.exit_status for v in verdicts]
    elif workers > 1:
        from coevobench.core.evaluation.concurrency import run_concurrent

        jobs = [(task_id, sample) for sample in range(k) for task_id in task_ids]

        def _one(task_id: str, sample: int) -> Verdict:
            return judge_fn(task_id)

        verdicts = run_concurrent(jobs, _one, workers=workers)
        samples = [v.exit_status for v in verdicts]
    else:
        for _sample in range(k):
            for task_id in task_ids:
                verdict = judge_fn(task_id)
                verdicts.append(verdict)
                samples.append(verdict.exit_status)
    elapsed = time.time() - started

    bad = n_bad(samples)
    dominant = is_dominant(samples)
    score: float | None = None
    if verdicts and bad == 0 and not dominant:
        score = sum(v.score for v in verdicts) / len(verdicts)

    record = EvalRecord(
        run_id=run_id,
        candidate=candidate,
        split=split,
        task_ids=tuple(task_ids),
        k=k,
        seed=seed,
        exit_status_counts=counts_summary(samples),
        n_bad=bad,
        score=score,
    )
    if out_dir is not None:
        _persist(record, verdicts, elapsed, out_dir)
    return record


def _persist(record: EvalRecord, verdicts: list[Verdict], elapsed: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{record.split}_{record.candidate}_{int(time.time())}"
    path = out_dir / f"{tag}.jsonl"
    summary = {
        "run_id": record.run_id,
        "candidate": record.candidate,
        "split": record.split,
        "n_tasks": len(record.task_ids),
        "k": record.k,
        "seed": record.seed,
        "exit_status_counts": dict(record.exit_status_counts),
        "n_bad": record.n_bad,
        "score": record.score,
        "wall_seconds": round(elapsed, 3),
    }
    # verdicts arrive as k sample-major blocks over the task ids
    rows = [
        {
            "task_id": task_id,
            "sample": sample,
            "resolved": verdict.resolved,
            "score": verdict.score,
            "exit_status": verdict.exit_status,
        }
        for sample in range(record.k)
        for task_id, verdict in zip(
            record.task_ids,
            verdicts[sample * len(record.task_ids) : (sample + 1) * len(record.task_ids)],
            strict=True,
        )
        if verdict.exit_status
    ]
    with path.open("w") as handle:
        handle.write(json.dumps(summary) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def scalar_feedback(record: EvalRecord) -> str:
    """The Selection-set contract: one scalar, no task contents.

    Unreliable runs (bad > 0 or dominant exit statuses) say so instead of
    reporting a number.
    """
    if record.score is None:
        if record.n_bad:
            return f"incomplete: n_bad={record.n_bad}"
        return "unreliable: dominant exit status"
    return f"resolved_rate={record.score:.4f} n={len(record.task_ids) * record.k} k={record.k}"


__all__ = ["JudgeFn", "evaluate_split", "scalar_feedback"]
