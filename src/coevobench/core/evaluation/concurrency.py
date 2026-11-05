"""Concurrent episode driver with per-job failure accounting.

The evaluate loop runs one judge_fn call per (task, sample) job; this
module runs them in a ThreadPoolExecutor: a job that raises becomes a bad record with a
WorkerError exit status — never a crash of the whole evaluation, and
never a silent zero.

Thread-safety contract: judge_fn must be independent per call (the
broker's judges create their own sandboxes per call; the serving
pool is entered once and shared read-only). The swe image catalog and
the vLLM endpoint are shared — the catalog is resolved before the
pool starts.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

_lock = threading.Lock()


def run_concurrent(
    jobs: list[tuple[str, int]],
    run_one: Callable[[str, int], dict],
    *,
    workers: int = 8,
    on_progress: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    """Run (task_id, sample) jobs concurrently; every job yields a record.

    Crashed jobs become {bad: True, exit_status: WorkerError:...} so
    the caller's n_bad accounting picks them up.
    """
    if not jobs:
        return []
    if len(jobs) != len(set(jobs)):
        raise ValueError("jobs must be unique (task, sample) pairs")
    if workers < 1:
        raise ValueError("workers must be positive")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, task_id, sample): (task_id, sample)
            for task_id, sample in jobs
        }
        for future in as_completed(futures):
            task_id, sample = futures[future]
            done += 1
            try:
                record = future.result()
                if isinstance(record, dict):
                    record = _as_verdict(record, task_id, sample)
            except Exception as exc:  # noqa: BLE001 - the driver's job
                from coevobench.core.protocol.spec import Verdict

                record = Verdict(
                    resolved=False,
                    score=0.0,
                    exit_status=f"WorkerError:{type(exc).__name__}",
                )
                with _lock:
                    print(
                        f"[driver] job crashed: {task_id}#{sample}: "
                        f"{type(exc).__name__}: {str(exc)[:160]}",
                        flush=True,
                    )
            if on_progress is not None:
                on_progress(done, len(jobs), record)
            results.append(record)
    return results


def _as_verdict(payload: dict, task_id: str, sample: int):
    from coevobench.core.protocol.spec import Verdict

    status = payload.get("exit_status") or "WorkerError:Unknown"
    return Verdict(
        resolved=bool(payload.get("resolved")),
        score=float(payload.get("score") or 0.0),
        exit_status=status,
    )


__all__ = ["run_concurrent"]
