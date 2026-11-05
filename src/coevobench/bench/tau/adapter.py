"""τ²-bench Telecom adapter (design document Section 4.3; paper C.1).

Telecom is an in-process environment: no sandbox images. The frozen
protocol (paper Appendix D.2): GPT-4o simulates the user (temperature 0,
600-token output limit), GPT-4.1 judges natural-language assertions, and
each of the 45 test tasks evaluates four times (Eq. 8: mean single-trial
success). External-model endpoints are provisioned through operator
proxies — credentials never live in the run envelope.

The adapter judges from a recorded trajectory: reward is the environment's
own task-check outcome (all checks must pass), never an agent self-report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coevobench.bench.adapter import FileBench
from coevobench.core.protocol.spec import TaskView, Verdict
from coevobench.core.protocol.splits import validate_frozen

WALL_LIMIT_S = 1200
STEP_LIMIT = 200
OUTPUT_TOKENS = 1024
CONTEXT_LENGTH = 32768
TEST_TRIALS = 4  # paper: each of the 45 test tasks evaluated four times


@dataclass(frozen=True)
class TelecomTask:
    instruction: str
    user_sim_notes: str = ""


class TauBench(FileBench):
    name = "tau"

    def __init__(self, data_root: Path, tasks: dict[str, TelecomTask] | None = None) -> None:
        super().__init__(data_root)
        self._tasks = tasks or {}
        validate_frozen(self.name, self.splits)

    def public_view(self, task_id: str) -> TaskView:
        task = self._tasks[task_id]
        return TaskView(task_id=task_id, prompt=task.instruction)

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict:
        """Judge a recorded trajectory.

        `submission` carries the trajectory's final environment state as
        JSON: {"checks": [{"name": ..., "passed": bool}, ...]}. The
        environment (tau2's telecom domain) produces it; this adapter only
        decides resolved = all checks passed (paper: success requires
        passing all task checks).
        """
        import json

        try:
            payload = json.loads(submission)
            checks = payload["checks"]
        except (ValueError, KeyError):
            return Verdict(resolved=False, score=0.0, exit_status="FormatError")
        if not checks:
            return Verdict(resolved=False, score=0.0, exit_status="NoChecks")
        resolved = all(check.get("passed", False) for check in checks)
        return Verdict(
            resolved=resolved,
            score=float(resolved),
            exit_status="Submitted" if resolved else "ChecksFailed",
        )

    def test_trials(self) -> int:
        return TEST_TRIALS


__all__ = ["TauBench", "TelecomTask", "TEST_TRIALS"]


def task_kind(task_id: str) -> str:
    """The paper's telecom stratification dimension: read-only vs write.

    Read-only tasks exercise database queries (the task id prefix in the
    frozen pool carries the kind); write tasks mutate telecom state.
    Reporting by kind keeps the Table 3 stratification observable in
    results (Eq. 8 averages all trials; kind breakdown is diagnostic).
    """
    return "readonly" if task_id.endswith(":readonly") else "write"


def summarize_by_kind(records: list[dict]) -> dict[str, dict]:
    """Per-kind diagnostic over eval rows: {kind: {n, resolved, mean}}.

    Rows carry task_id and score; kind derives from the task id. The
    Eq. 8 headline stays the all-trials mean — this is the stratified
    view the frozen split was built to support.
    """
    grouped: dict[str, list[float]] = {}
    for row in records:
        kind = task_kind(str(row.get("task_id", "")))
        grouped.setdefault(kind, []).append(float(row.get("score", 0.0)))
    return {
        kind: {
            "n": len(scores),
            "resolved": sum(1 for s in scores if s >= 1.0),
            "mean": round(sum(scores) / len(scores), 4),
        }
        for kind, scores in grouped.items()
    }


__all__ = ["TauBench", "TelecomTask", "TEST_TRIALS", "summarize_by_kind", "task_kind"]
