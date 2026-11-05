"""Final evaluation: replay a frozen candidate on the secret Test split.

The authoritative score (paper Section 3.3): the operator serves the
candidate's declared weights with the candidate's declared serving
shape, runs the frozen harness over every Test task, and validates the
result completeness. Development-phase scores never reach a report.

Two phases:

- prepare: freeze the nominated candidate into a bundle outside the
  researcher workspace (weights copied in, chat template materialized,
  manifest hashed read-only) — CPU/filesystem only.
- evaluate: serve from the single command builder, execute the bench
  adapter over the Test ids, validate rows (no missing/duplicate
  trials, no infra failures, scores in [0, 1]).

--revert-weights implements Eq. (5): on confirmed cheating, evaluate
with the base weights and the evolved harness instead.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coevobench.core.evaluation.evaluate import evaluate_split
from coevobench.core.grading.freeze import freeze, validate, verify_unchanged
from coevobench.core.protocol.exit_status import is_infra_failure
from coevobench.core.protocol.splits import Splits


@dataclass(frozen=True)
class FinalReport:
    bench: str
    n_tasks: int
    n_trials: int
    score: float
    metric: str
    weights: str
    reverted: bool = False
    # bench-defined secondary metrics (appworld SGC, bfcl category
    # breakdowns); the scalar score stays the paper's comparable
    provenance: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "bench": self.bench,
                "n_tasks": self.n_tasks,
                "n_trials": self.n_trials,
                "score": round(self.score, 4),
                "metric": self.metric,
                "weights": self.weights,
                "reverted": self.reverted,
                "metrics": self.metrics,
            },
            indent=2,
        )


def prepare(
    submission_dir: Path,
    bench: str,
    destination: Path,
    *,
    label: str | None = None,
    base_weights: Path | None = None,
) -> Path:
    """Freeze the nomination into a self-contained bundle.

    The bundle carries submission (harness + model_path + meta +
    serve_config) and the weights themselves — evaluation never reads
    the researcher workspace. Bundles live outside the workspace by
    construction.
    """
    submission_dir = Path(submission_dir).resolve()
    destination = Path(destination).resolve()
    if destination.is_relative_to(submission_dir):
        raise ValueError("final bundles must be stored outside the submission")

    errors = validate(submission_dir)
    if errors:
        raise ValueError(f"candidate failed freeze gates: {list(errors)}")

    model_path = Path((submission_dir / "model_path.txt").read_text().strip())
    if not model_path.is_absolute():
        raise ValueError("model_path.txt must contain an absolute path")
    # envelope terms (/base, /workspace/...) resolve against the run
    # workspace the submission sits in
    from coevobench.core.grading.submission import checkpoint_path

    model_path = checkpoint_path(
        submission_dir.parent.parent, model_path
    )
    base_weights = (base_weights or model_path.parent / "base").resolve()
    untrained = model_path == base_weights
    # An untrained submission is legal (Base + evolved harness is a valid
    # candidate) but weights must still be materialized.
    if untrained and not (model_path / "config.json").is_file():
        raise ValueError("weights must be a full checkpoint with config.json")

    bundle = destination / (label or submission_dir.name)
    if bundle.exists():
        raise ValueError(f"bundle already exists: {bundle}")
    staging = destination / f".preparing-{bundle.name}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            submission_dir,
            staging / "submission",
            ignore=shutil.ignore_patterns("__pycache__"),
            dirs_exist_ok=True,
        )
        weights_target = staging / "weights"
        shutil.copytree(
            model_path,
            weights_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.optim", "checkpoint-*"),
        )
        (staging / "submission" / "model_path.txt").write_text(
            str(bundle / "weights") + "\n"
        )
        metadata = {
            "bench": bench,
            "prepared_from": str(submission_dir),
            "weights_source": str(model_path),
        }
        (staging / "evaluation.json").write_text(json.dumps(metadata, indent=2))
        freeze(staging / "submission", label or bundle.name)
        staging.rename(bundle)
        return bundle
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def evaluate(
    bundle: Path,
    *,
    bench: str,
    splits: Splits,
    judge_fn: Callable[[str], Any] | None = None,
    adapter_name: str = "",
    k: int = 1,
    weights_override: Path | None = None,
) -> FinalReport:
    """Replay the frozen candidate over the Test split.

    Two modes:
    - judge_fn supplied: the caller wires its own judging (tests use
      this; so do operator controls without serving).
    - judge_fn omitted: the full replay engine drives the candidate —
      serve the bundle's weights through the single command builder,
      run the candidate's harness over every Test task through the
      welded SandboxEnvironment, extract the submission, and judge it
      fresh-sandbox (bench adapter + SweBench directive grading).

    --revert-weights (Eq. 5, confirmed cheating) is weights_override=base:
    same harness, base weights.
    """
    bundle = Path(bundle).resolve()
    if not verify_unchanged(bundle / "submission"):
        raise ValueError("frozen bundle changed after freezing; refusing evaluation")
    record = json.loads((bundle / "evaluation.json").read_text())
    if record.get("bench") != bench:
        raise ValueError(f"bundle is for bench {record.get('bench')!r}, not {bench!r}")

    test_ids = list(splits.ids("test"))
    if not test_ids:
        raise ValueError("empty Test split")

    from coevobench.core.lifecycle.timing import Recorder

    recorder = Recorder(bundle.parent / "timing.jsonl")
    with recorder.span("final_eval", f"evaluate:{bundle.name}", bench=bench, k=k):
        if judge_fn is None:
            judge_fn = _resolve_replay_judge(bench, bundle, weights_override)

        run = evaluate_split(
            run_id=f"final-{bench}",
            candidate=bundle.name,
            split="test",
            task_ids=test_ids,
            judge_fn=judge_fn,
            k=k,
        )

    if run.n_bad:
        raise ValueError(
            f"infrastructure failures present ({run.n_bad}); "
            "result is incomplete, not an official score"
        )
    if run.score is None:
        raise ValueError("run unreliable (dominant exit status); not an official score")

    reverted = weights_override is not None
    weights = str(weights_override) if reverted else record["weights_source"]
    return FinalReport(
        bench=bench,
        n_tasks=len(test_ids),
        n_trials=len(test_ids) * k,
        score=run.score,
        metric="score_mean",
        weights=weights,
        reverted=reverted,
    )


_REPLAY_BINDINGS: dict[str, object] = {}


def register_replay_binding(bench: str, factory) -> None:
    """Register the bench layer's judge factory for full replay.

    core stays bench-agnostic: the upper layers (CLI, broker, tests)
    import the bench module that provides the binding and register it
    here. evaluate() with judge_fn=None resolves through this registry.
    """
    _REPLAY_BINDINGS[bench] = factory


def _resolve_replay_judge(bench: str, bundle: Path, weights_override: Path | None):
    factory = _REPLAY_BINDINGS.get(bench)
    if factory is None:
        raise ValueError(
            f"final-eval replay is not registered for {bench!r}; "
            "register the bench binding (bench.<name>.replay) or supply judge_fn"
        )
    return factory(bundle, bench, weights_override)


def summarize_rows(rows: list[dict], bench: str) -> dict:
    """Completeness validation: every (task, sample) present exactly once,
    no infra failures, every score finite in [0, 1]."""
    expected_pairs = {(r["task_id"], r["sample"]) for r in rows}
    if len(rows) != len(expected_pairs):
        raise ValueError("duplicate trials in final rows")
    for row in rows:
        if is_infra_failure(row.get("exit_status", "")):
            raise ValueError(f"infra failure in final rows: {row.get('task_id')}")
        score = float(row["score"])
        if score < 0 or score > 1:
            raise ValueError(f"score out of range for {row.get('task_id')}: {score}")
    return {
        "n_rows": len(rows),
        "metric": "score_mean",
    }


__all__ = [
    "FinalReport",
    "evaluate",
    "prepare",
    "summarize_rows",
]
