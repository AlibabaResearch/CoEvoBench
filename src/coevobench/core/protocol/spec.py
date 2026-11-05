"""Core domain types. Zero infrastructure dependencies.

RunSpec instantiates the paper's Eq. (1): P_b = (S_b^(0), I_b, U_b, E_b^dev, B).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Budget:
    """Paper Section 3.2: 24 hours, four GPUs, $100 teacher-model calling budget."""

    hours: float
    gpus: tuple[int, ...]
    teacher_spend_usd: float

@dataclass(frozen=True)
class RunSpec:
    """A development run: P_b = (S_b^(0), I_b, U_b, E_b^dev, B).

    initial_system is (weights_path, harness_dir); materials is the accessible
    set I_b (TASK brief, docs, visible task pool); action_space is the guard
    policy encoding U_b; feedback is the evaluation contract E_b^dev.
    """

    bench: str
    researcher_model: str
    researcher_cli: str
    initial_weights: str
    initial_harness: str
    materials: tuple[str, ...] = ()
    budget: Budget | None = None
    replicate: int = 1
    seed: int = 0

    @property
    def run_id(self) -> str:
        return f"{self.researcher_model}_{self.bench}_{self.replicate}"


@dataclass(frozen=True)
class TaskView:
    """Public view of a task: gold stripped. Paper: visible set provides task
    statements; patches and private tests are managed by the evaluator."""

    task_id: str
    prompt: str
    image: str = ""


@dataclass(frozen=True)
class Verdict:
    resolved: bool
    score: float
    exit_status: str


class BenchAdapter(Protocol):
    name: str

    def public_view(self, task_id: str) -> TaskView: ...

    def task_ids(self, split: str) -> list[str]: ...

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict: ...


@dataclass(frozen=True)
class EvalRecord:
    """One development evaluation, persisted under runs/<id>/ (Section 8.5).

    Paper Appendix E.2: "Development evaluations record task subsets and
    repetition counts."
    """

    run_id: str
    candidate: str
    split: str
    task_ids: tuple[str, ...]
    k: int
    seed: int
    exit_status_counts: Mapping[str, int] = field(default_factory=dict)
    n_bad: int = 0
    score: float | None = None
