"""AppWorld easy adapter (design document Section 4.3; paper C.1).

In-process environment: the agent generates Python code calling
application APIs; the official task goal completion (TGC) measures
complete-task success. The adapter judges from the recorded final state:
AppWorld's own evaluator output, parsed — never the agent's self-report.

Split semantics (paper): variants of the same scene stay in one split;
129 difficulty-1 tasks across 43 scenes -> 9/17/17 scenes.
"""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.bench.adapter import FileBench
from coevobench.core.protocol.spec import TaskView, Verdict
from coevobench.core.protocol.splits import validate_frozen

STEP_LIMIT = 50
OUTPUT_TOKENS = 3000
CONTEXT_LENGTH = 65536


class AppWorldBench(FileBench):
    name = "appworld"

    def __init__(self, data_root: Path, tasks: dict[str, dict] | None = None) -> None:
        super().__init__(data_root)
        self._tasks = tasks or {}
        validate_frozen(self.name, self.splits)

    def public_view(self, task_id: str) -> TaskView:
        task = self._tasks[task_id]
        return TaskView(task_id=task_id, prompt=task["instruction"])

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict:
        """submission = the final evaluator JSON: {"tgc": bool, ...}."""
        try:
            payload = json.loads(submission)
            tgc = payload["tgc"]
        except (ValueError, KeyError):
            return Verdict(resolved=False, score=0.0, exit_status="FormatError")
        if not isinstance(tgc, bool):
            return Verdict(resolved=False, score=0.0, exit_status="FormatError")
        return Verdict(
            resolved=tgc,
            score=float(tgc),
            exit_status="Submitted" if tgc else "GoalIncomplete",
        )


__all__ = ["AppWorldBench"]
