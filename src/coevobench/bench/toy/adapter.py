"""Toy bench: built-in smoke loop (design doc Section 4.5).

Five tasks: compute the sum of squares 1..N and write answer.json; the
verifier checks it. Not part of the paper protocol; never aggregated into
`ceb report`. Exists so the full stack (sandbox -> judge -> report plumbing)
runs in CI without GPUs or real benchmark data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from coevobench.core.protocol.spec import TaskView, Verdict

TASKS: dict[str, int] = {
    "toy-sum-1": 10,
    "toy-sum-2": 100,
    "toy-sum-3": 500,
    "toy-sum-4": 1000,
    "toy-sum-5": 2500,
}

SPLITS = {
    "visible": ["toy-sum-1", "toy-sum-2"],
    "selection": ["toy-sum-3"],
    "test": ["toy-sum-4", "toy-sum-5"],
}

IMAGE = "python:3.11-slim"


@dataclass(frozen=True)
class ToyBench:
    name: str = "toy"

    def task_ids(self, split: str) -> list[str]:
        return list(SPLITS[split])

    def public_view(self, task_id: str) -> TaskView:
        n = TASKS[task_id]
        prompt = (
            f"Compute the sum of squares of integers 1 through {n}, "
            f"then write the integer result to answer.json in the working directory."
        )
        return TaskView(task_id=task_id, prompt=prompt, image=IMAGE)

    def rollout_script(self, task_id: str) -> str:
        n = TASKS[task_id]
        total = sum(i * i for i in range(1, n + 1))
        script = f'import json; json.dump({total}, open("answer.json", "w"))'
        return f"python3 -c '{script}'"

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict:
        """submission is the sandbox's final stdout; the source of truth is answer.json."""
        n = TASKS[task_id]
        expected = sum(i * i for i in range(1, n + 1))
        try:
            text = sandbox.read_file("answer.json").strip()
            resolved = json.loads(text) == expected
        except Exception:
            resolved = False
        exit_status = "Submitted" if resolved else "FormatError"
        return Verdict(resolved=resolved, score=float(resolved), exit_status=exit_status)
