"""BFCL-V4 multi-turn and memory adapter (design document Section 4.3; paper C.2).

In-process grading over scoring records. The frozen protocol: the pool
combines 800 multi-turn records (200 per category) and 465 memory QA
records (155 per backend); multi-turn variants of the same task family
share a split, and each memory scenario groups with its prerequisite
conversations and QA records. The primary score is Eq. (9):

    s_BFCL = 100 * (0.6 * mean(multi_turn categories) + 0.4 * mean(memory backends))

Judge input is a per-record correctness payload, grouped by component.
"""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.bench.adapter import FileBench
from coevobench.core.protocol.spec import TaskView, Verdict
from coevobench.core.protocol.splits import validate_frozen

STEP_LIMIT = 20          # count > limit loop semantics (paper Table 5)
OUTPUT_TOKENS = 4096
CONTEXT_LENGTH = 65536

MULTI_TURN_WEIGHT = 0.6
MEMORY_WEIGHT = 0.4
MULTI_TURN_CATEGORIES = 4
MEMORY_BACKENDS = 3


class BfclBench(FileBench):
    name = "bfcl"

    def __init__(self, data_root: Path, tasks: dict[str, dict] | None = None) -> None:
        super().__init__(data_root)
        self._tasks = tasks or {}
        validate_frozen(self.name, self.splits)

    def public_view(self, task_id: str) -> TaskView:
        task = self._tasks[task_id]
        return TaskView(task_id=task_id, prompt=task.get("prompt", ""))

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict:
        """submission = one record's grading payload: {"correct": bool}."""
        try:
            payload = json.loads(submission)
            correct = payload["correct"]
        except (ValueError, KeyError):
            return Verdict(resolved=False, score=0.0, exit_status="FormatError")
        if not isinstance(correct, bool):
            return Verdict(resolved=False, score=0.0, exit_status="FormatError")
        return Verdict(
            resolved=correct,
            score=float(correct),
            exit_status="Submitted" if correct else "RecordIncorrect",
        )

    def component_score(self, records: dict[str, list[bool]]) -> float:
        """Eq. 9 over grouped records: multi-turn categories and memory backends.

        `records` maps a component key ("multi_turn:<category>" or
        "memory:<backend>") to its per-record correctness. Equal record
        counts within a component make this the weighted mean the paper
        describes; unequal counts fall back to per-record averaging inside
        the component first.
        """
        multi_keys = [k for k in records if k.startswith("multi_turn:")]
        memory_keys = [k for k in records if k.startswith("memory:")]
        if not multi_keys or not memory_keys:
            raise ValueError("need both multi_turn:* and memory:* components")
        multi_mean = _mean_of_means(records, multi_keys)
        memory_mean = _mean_of_means(records, memory_keys)
        return MULTI_TURN_WEIGHT * multi_mean + MEMORY_WEIGHT * memory_mean


def _mean_of_means(records: dict[str, list[bool]], keys: list[str]) -> float:
    per_component = [
        sum(records[key]) / len(records[key]) for key in keys if records[key]
    ]
    if not per_component:
        raise ValueError("empty component groups")
    return sum(per_component) / len(per_component)


__all__ = ["BfclBench", "MEMORY_WEIGHT", "MULTI_TURN_WEIGHT"]
