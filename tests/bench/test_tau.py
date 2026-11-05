"""τ² telecom adapter: by-kind diagnostics (paper Table 3 stratification)."""

from __future__ import annotations


def test_by_kind_diagnostics() -> None:
    from coevobench.bench.tau.adapter import summarize_by_kind, task_kind

    assert task_kind("telecom:read-db:readonly") == "readonly"
    assert task_kind("telecom:update-plan") == "write"

    rows = [
        {"task_id": "telecom:read-a:readonly", "score": 1.0},
        {"task_id": "telecom:read-b:readonly", "score": 0.0},
        {"task_id": "telecom:write-a", "score": 1.0},
    ]
    summary = summarize_by_kind(rows)
    assert summary["readonly"] == {"n": 2, "resolved": 1, "mean": 0.5}
    assert summary["write"] == {"n": 1, "resolved": 1, "mean": 1.0}
