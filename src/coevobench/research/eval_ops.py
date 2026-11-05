"""Contamination and evaluation op handlers."""

from __future__ import annotations

import contextlib
import math
from typing import Any

from coevobench.research.context import BrokerContext
from coevobench.runtime.broker_wire import BrokerError

_HANDLERS: dict[str, Any] = {}


def register(into: dict[str, Any]) -> None:
    into.update(_HANDLERS)


def op(name: str):
    def wrap(fn):
        _HANDLERS[name] = fn
        return fn

    return wrap


# ---- contamination ----


class _ContaminationState:
    """One bench's screening service: frozen reference + receipt ledger.

    State persists under <root>/research/contamination (operator-side;
    the directory is never mounted into the run container). The
    researcher receives only the receipt's public view.
    """

    def __init__(self, workspace_root, bench: str) -> None:
        self._workspace_root = workspace_root
        self._bench = bench
        self._reference_items = None
        from coevobench.core.contamination.policy import ReceiptLedger

        root = Path(workspace_root) if workspace_root else Path("/root/.ceb")
        self._ledger = ReceiptLedger(root / "research" / "contamination")

    def check(self, inputs: list[str]) -> dict[str, Any]:
        from coevobench.core.contamination import (
            ScreeningError,
            build_reference_items,
        )
        from coevobench.core.contamination.policy import ReceiptLedger as _L

        if self._reference_items is None:
            task_reader = self._load_task_reader()
            from coevobench.core.protocol.protocol_assets import frozen_splits
            from coevobench.core.protocol.splits import from_mapping

            frozen = frozen_splits(self._bench)
            if frozen is None:
                raise ScreeningError(f"bench {self._bench!r} has no frozen splits")
            splits = from_mapping(
                {
                    "visible": list(frozen.visible),
                    "selection": list(frozen.selection),
                    "test": list(frozen.test),
                }
            )
            self._ledger.freeze_reference(
                self._bench, splits, task_reader=task_reader
            )
            self._reference_items = build_reference_items(
                self._bench, splits, task_reader=task_reader
            )
        receipt = self._ledger.check(inputs, self._reference_items)
        return _L.public_view(receipt)

    def _load_task_reader(self) -> dict[str, str]:
        from coevobench.config.schema import Config
        from coevobench.core.protocol.protocol_assets import cache_root

        root = cache_root(Config())
        reader: dict[str, str] = {}
        path = root / f"{self._bench}_tasks.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = row.get("instance_id") or row.get("task_id")
                    text = row.get("problem_statement") or row.get("instruction") or ""
                    if key:
                        reader[key] = text
        return reader


import json  # noqa: E402
from pathlib import Path  # noqa: E402

_state: _ContaminationState | None = None


@op("contamination.check")
def contamination_check(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    global _state

    if ctx.bench not in ("swe", "tb"):
        raise BrokerError("corpus screening is not configured for this experiment")
    # the wire envelope carries the op name; the payload check applies
    # to what remains
    if set(request) - {"op"} != {"inputs"}:
        raise BrokerError("corpus screening accepts only the complete input file list")
    # the researcher names corpus files in envelope paths (/workspace/...,
    # relative to the workspace); the screening engine reads host-side
    inputs = [
        str(ctx.local_path(p, must_exist=True)) for p in request["inputs"]
    ]
    with ctx.lock:
        if _state is None:
            _state = _ContaminationState(ctx.workspace_root, ctx.bench)
    return _state.check(inputs)


# ---- evaluation ----


def _prepare_eval(request: dict[str, Any]) -> tuple[dict, list[str], int]:
    from coevobench.core.protocol.protocol_assets import frozen_splits
    from coevobench.core.protocol.splits import from_mapping

    request = dict(request)
    bench = request.get("bench", "")
    split = request.get("split", "")
    if split not in ("visible", "selection"):
        raise BrokerError(f"split must be visible|selection (got {split!r})")
    frozen = frozen_splits(bench)
    if frozen is None and bench == "toy":
        from coevobench.bench.toy.adapter import SPLITS as TOY_SPLITS

        frozen = from_mapping(TOY_SPLITS)
    if frozen is None:
        raise BrokerError(f"bench {bench!r} is not configured")
    splits = from_mapping(
        {
            "visible": list(frozen.visible),
            "selection": list(frozen.selection),
            "test": list(frozen.test),
        }
    )
    ids = list(splits.ids(split))
    tasks_arg = request.get("tasks")
    if tasks_arg:
        from coevobench.bench.anon import anon_to_real, load_mapping, make_anon_id

        mapping = load_mapping(bench) or {
            real: make_anon_id(real, bench) for real in ids
        }
        # resolve ONLY within the requested split: a mapping that ships
        # every split must not let an anonymous test id into a
        # selection query
        mapping = {real: anon for real, anon in mapping.items() if real in ids}
        ids = anon_to_real(tasks_arg, mapping)
    n_prefix = request.get("n")
    if n_prefix is not None:
        if int(n_prefix) < 1:
            raise BrokerError("n must be positive")
        ids = ids[: int(n_prefix)]
    k = int(request.get("k", 1))
    workers = int(request.get("workers", 1))
    if k < 1 or workers < 1 or not ids:
        raise BrokerError("evaluation needs positive k/workers and nonempty tasks")
    if request.get("formal"):
        k = _apply_formal_presets(request)
    wall = float(request.get("wall_cap_s", 900))
    if not math.isfinite(wall) or wall < 0:
        raise BrokerError("wall_cap_s must be finite and nonnegative")
    request.update(k=k, workers=workers)
    return request, ids, k


@op("eval.plan")
def eval_plan(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    """Return an effective wait budget, without exposing task identities."""
    prepared, ids, k = _prepare_eval(request)
    workers = prepared["workers"]
    wall = float(prepared.get("wall_cap_s", 900))
    # Per wave: provisioning, episode (including driver grace), verifier,
    # teardown. Serving is started once outside the worker pool.
    timeout = 600 + math.ceil(len(ids) * k / workers) * (120 + wall + 120 + 600 + 120) + 120
    return {
        "n_tasks": len(ids), "k": k, "workers": workers,
        "wall_cap_s": wall, "timeout_s": max(3600, timeout) if wall else None,
    }


@op("eval.run")
def eval_run(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    # A disconnected client may retry while its previous request still
    # owns a serving process. Never start two evaluations on those GPUs.
    if not ctx.evaluation_lock.acquire(blocking=False):
        raise BrokerError("an evaluation is already running; wait for its result before retrying")
    try:
        return _evaluate(ctx, request)
    finally:
        ctx.evaluation_lock.release()


def _evaluate(ctx: BrokerContext, request: dict[str, Any]) -> dict[str, Any]:
    from coevobench.core.evaluation.evaluate import evaluate_split, scalar_feedback

    request, ids, k = _prepare_eval(request)
    bench, split = request["bench"], request["split"]
    gpu = getattr(ctx, "gpu_jobs", None)
    if gpu is not None and gpu.status().get("status") in ("running", "starting"):
        raise BrokerError("a GPU job holds the cards; gpu.stop it before evaluating")
    submission = request.get("submission_dir")
    model = request.get("model")

    if submission or model:
        judge = ctx.candidate_judge_for(
            bench, submission, request,
        )
    else:
        judge = ctx.judge_for(bench, request)
    # The record persists engine-side (under the run's workspace) even
    # when the client dies waiting: an orphaned evaluation must leave
    # its verdicts behind, not vanish with the socket. The researcher
    # finds them under out/eval/.
    out_dir = None
    if ctx.workspace_root:
        out_dir = (ctx.workspace_root / "out" / "eval" if split == "visible"
                   else ctx.workspace_root.parent / "select_runs")
    session = getattr(judge, "evaluation_session", contextlib.nullcontext)
    with session():
        record = evaluate_split(
            run_id=str(request.get("run_id", "broker-eval")),
            candidate=str(request.get("candidate", "cand0")),
            split=split,
            task_ids=ids,
            judge_fn=judge,
            k=k,
            workers=request["workers"],
            out_dir=out_dir,
        )
    return {
        "n_tasks": len(ids),
        "k": k,
        "n_bad": record.n_bad,
        "exit_status_counts": dict(record.exit_status_counts),
        "score": record.score,
        "feedback": scalar_feedback(record) if split == "selection" else None,
    }


def _apply_formal_presets(request: dict[str, Any]) -> int:
    """The announced final-evaluation settings (paper Table 5): the
    researcher's declared overrides are refused in favor of the
    published values — self-evaluation stays free-form, formal-mode
    numbers are comparable by construction."""
    from coevobench.bench.swe.adapter import (
        CONTEXT_LENGTH,
        OUTPUT_TOKENS,
        STEP_LIMIT,
        WALL_LIMIT_S,
    )

    request["temperature"] = 0.0
    request["max_tokens"] = OUTPUT_TOKENS
    request["wall_cap_s"] = WALL_LIMIT_S
    request["step_limit"] = STEP_LIMIT
    request["max_model_len"] = CONTEXT_LENGTH
    return 1
