"""ceb submit / ceb track / ceb results: researcher submission signal,
research-effort tracking, and the results scoreboard.

Thin CLIs over core functions; every command is safe to run inside the
run envelope (no secrets read).
"""

from __future__ import annotations

import sys
from pathlib import Path


def submit_cmd(workspace: str, bench: str, *, confirm: bool = False) -> int:
    from coevobench.core.grading.submission import confirm as confirm_submission
    from coevobench.core.grading.submission import request as request_submission

    root = Path(workspace).resolve()
    try:
        if confirm:
            record = confirm_submission(root, bench)
            print(f"[submit] confirmed (digest {record['digest'][:12]})")
        else:
            target = request_submission(root, bench)
            print(f"[submit] request written: {target}")
            print("[submit] DONE.md plus this request is the explicit signal;")
            print("[submit] edits after this point invalidate the request.")
    except ValueError as exc:
        print(f"[submit] FAIL {exc}", file=sys.stderr)
        return 1
    return 0


def track_cmd(output: str, command: list[str] | None = None) -> int:
    """Record one timed research span around a command, or dump a summary.

    ./track make train     -> spans for the command, appended to out/
    ./track --summary      -> per-label counts and durations
    """
    from coevobench.core.lifecycle.timing import Recorder, summarize

    path = Path(output)
    if command:
        import subprocess
        import time

        recorder = Recorder(path)
        label = " ".join(command)
        start = time.time()
        with recorder.span("researcher", label):
            proc = subprocess.run(command, start_new_session=True)
        elapsed = time.time() - start
        print(f"[track] {label}: exit={proc.returncode} ({elapsed:.1f}s) -> {path}")
        return proc.returncode
    summary = summarize(path)
    for key, row in summary["spans"].items():
        print(
            f"[track] {key:32s} n={row['count']:4d} "
            f"total={row['total_s']:9.1f}s p95={row['p95_s']:7.1f}s"
        )
    if summary["open"]:
        print(f"[track] WARNING: {summary['open']} unfinished spans", file=sys.stderr)
    return 0


def results_cmd(*, bench: str | None = None, run_id: str | None = None) -> int:
    from coevobench.core.lifecycle.results import ResultStore

    store = ResultStore(Path.home() / ".ceb" / "results")
    rows = store.entries(run_id=run_id, bench=bench)
    if not rows:
        print("[results] no recorded results yet")
        return 0
    for row in rows:
        score = row.get("score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "-"
        print(
            f"[results] {row.get('run_id', '?'):24s} {row.get('bench', '?'):8s} "
            f"{row.get('split', '?'):10s} score={score_text} n={row.get('n_tasks', '?')}"
        )
    return 0


__all__ = ["results_cmd", "submit_cmd", "track_cmd"]
