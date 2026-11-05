"""Comparison controls: the M0H0 / M0H1 / M1H1 matched-configuration
matrix (B4; paper Tables 7 and 9's source).

Three roles over one run's artifacts, all graded by the same
final-eval path:

- M0_H0: the seed harness + base weights (the floor; matches the
  initial system)
- M0_H1: the seed harness + trained weights (training-gain isolation)
- M1_H1: the evolved harness + trained weights (the submission)

prepare() stages the three bundles outside the researcher workspace;
evaluate() grades each on the Test split through the shared engine.
Differences between roles read directly as harness-gain (M1H1 vs
M0H1) and training-gain (M0H1 vs M0H0).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from coevobench.core.lifecycle.final_eval import FinalReport, evaluate, prepare

ROLES = ("M0_H0", "M0_H1", "M1_H1")


@dataclass(frozen=True)
class ComparisonResult:
    bench: str
    scores: dict[str, dict]

    @property
    def training_gain(self) -> float | None:
        m0h1, m0h0 = self.scores.get("M0_H1"), self.scores.get("M0_H0")
        if not m0h1 or not m0h0:
            return None
        return m0h1["score"] - m0h0["score"]

    @property
    def harness_gain(self) -> float | None:
        m1h1, m0h1 = self.scores.get("M1_H1"), self.scores.get("M0_H1")
        if not m1h1 or not m0h1:
            return None
        return m1h1["score"] - m0h1["score"]


def prepare_comparison(
    submission_dir: Path,
    bench: str,
    destination: Path,
    *,
    base_weights: Path,
    seed_harness_src: Path,
) -> dict[str, Path]:
    """Stage the three role bundles (CPU/filesystem only).

    seed_harness_src is the SAVED initial harness package root (the
    operator supplies the original snapshot, never the live template).
    """
    submission_dir = Path(submission_dir).resolve()
    destination = Path(destination).resolve()
    if destination.is_relative_to(submission_dir):
        raise ValueError("comparison destination must live outside the submission")

    weights = Path((submission_dir / "model_path.txt").read_text().strip()).resolve()
    if not (weights / "config.json").is_file():
        raise ValueError("trained checkpoint missing config.json")

    bundles: dict[str, Path] = {}
    staging = Path(tempfile.mkdtemp(prefix=".comparison-", dir=destination))
    try:
        # M1_H1: the submission itself
        bundles["M1_H1"] = prepare(
            submission_dir, bench, staging / "M1_H1", base_weights=base_weights
        )

        # M0_H1 and M0_H0 share the submission staging structure but
        # swap the harness and/or weights
        for role in ("M0_H1", "M0_H0"):
            role_dir = staging / role / "submission"
            shutil.copytree(
                bundles["M1_H1"] / "submission",
                role_dir,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            harness_target = _harness_package_root(role_dir)
            shutil.rmtree(harness_target.parent.parent)
            shutil.copytree(
                seed_harness_src,
                role_dir / "harness" / bench / "src",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            if role == "M0_H0":
                (role_dir / "model_path.txt").write_text(str(base_weights) + "\n")
            (staging / role / "evaluation.json").write_text(
                json.dumps({"bench": bench, "role": role}, indent=2)
            )
            bundles[role] = staging / role

        for role, bundle in bundles.items():
            target = destination / role
            if target.exists():
                raise ValueError(f"{role} already staged")
            bundle.rename(target)
            bundles[role] = target
        return bundles
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _harness_package_root(submission: Path) -> Path:
    for candidate in submission.rglob("minisweagent/__init__.py"):
        return candidate.parent.parent
    raise ValueError("no minisweagent package in the submission")


def evaluate_comparison(
    bundles: dict[str, Path],
    bench: str,
    splits,
    *,
    k: int = 1,
) -> ComparisonResult:
    """Grade all three roles on the Test split."""
    scores: dict[str, dict] = {}
    for role in ROLES:
        bundle = bundles.get(role)
        if bundle is None:
            continue
        try:
            report: FinalReport = evaluate(
                bundle, bench=bench, splits=splits, k=k
            )
            scores[role] = {
                "score": report.score,
                "n_tasks": report.n_tasks,
                "n_trials": report.n_trials,
                "exit": report.metric,
            }
        except ValueError as exc:
            scores[role] = {"error": str(exc)[:200]}
    return ComparisonResult(bench=bench, scores=scores)


__all__ = [
    "ComparisonResult",
    "ROLES",
    "evaluate_comparison",
    "prepare_comparison",
]
