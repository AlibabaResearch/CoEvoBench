"""Result aggregation for final-eval outputs (paper Table 1 shape).

Reads per-bench JSON files ({bench, base, instruct, reported, n_runs})
and returns the domain rows plus the weighted Avg. Shared by the CLI
report command and the MCP tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coevobench.core.grading.scoring import BenchResult, DomainScores, aggregate


def aggregate_dir(results_dir: Path | str) -> dict[str, Any]:
    root = Path(results_dir)
    results: dict[str, BenchResult] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        bench = payload["bench"]
        results[bench] = BenchResult(
            bench=bench,
            scores=DomainScores(
                base=payload["base"],
                instruct=payload["instruct"],
                reported=payload["reported"],
            ),
            n_runs=payload.get("n_runs", 1),
        )
    if not results:
        raise ValueError(f"no result files under {root}")
    rows = [
        {
            "bench": result.bench,
            "base": result.scores.base,
            "instruct": result.scores.instruct,
            "reported": result.scores.reported,
            "improvement": result.scores.improvement,
            "n_runs": result.n_runs,
        }
        for result in results.values()
    ]
    from coevobench.core.grading.scoring import is_exploratory

    return {
        "benches": rows,
        "avg": aggregate(results),
        "exploratory": is_exploratory(results),
    }


__all__ = ["aggregate_dir"]
