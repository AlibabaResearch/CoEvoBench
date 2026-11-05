"""ceb eval: development evaluation over Visible or Selection.

The CLI front of the shared evaluation loop. Toy bench drives real
sandboxes without GPUs or served models; other benches need a served
candidate (wire via the bench adapter; engine: core/evaluate).
Selection output lands operator-side; the terminal shows the scalar
contract only.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path


def run_eval(
    bench: str,
    split: str,
    *,
    k: int = 1,
    n: int | None = None,
    tasks: str | None = None,
    candidate: str = "cand0",
    out_dir: Path | None = None,
    registry_root: Path | None = None,
    workers: int = 1,
    model: str | None = None,
    serve_config: str | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    wall_cap: int | None = None,
    step_limit: int | None = None,
    max_model_len: int | None = None,
    formal: bool = False,
    list_tasks: bool = False,
) -> int:
    # Inside the run envelope (broker socket mounted), forward every
    # setting to the host engine — the envelope holds no judging.

    if os.environ.get("CEB_BROKER_SOCKET") or os.path.exists("/broker.sock"):
        from coevobench.research.eval_client import run_eval as forward_eval

        return forward_eval(
            bench,
            split,
            k=k,
            tasks=tasks,
            candidate=candidate,
            workers=workers,
            n=n,
            model=model,
            serve_config=serve_config,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            chat_template=chat_template,
            temperature=temperature,
            max_tokens=max_tokens,
            wall_cap=wall_cap,
            step_limit=step_limit,
            max_model_len=max_model_len,
            formal=formal,
            list_tasks=list_tasks,
            run_id=os.environ.get("CEB_RUN_ID", ""),
        )
    if list_tasks:
        # The workspace's anonymous pool file first (the envelope's only
        # source of truth — bench data never installs there); the frozen
        # splits are the host fallback.
        ws = os.environ.get("CEB_WORKSPACE")
        for root in ([Path(ws)] if ws else []) + [Path.cwd()]:
            candidate = root / f"anonymous_pool_{bench}.txt"
            if candidate.is_file():
                print(candidate.read_text(), end="")
                return 0
        from coevobench.bench.anon import generate_mapping, load_mapping, pool_file_contents
        from coevobench.core.protocol.protocol_assets import frozen_splits

        frozen = frozen_splits(bench)
        if frozen is None:
            print(f"[eval] unknown bench {bench!r}", file=sys.stderr)
            return 2
        mapping = load_mapping(bench) or generate_mapping(bench, list(frozen.selection))
        print(pool_file_contents(bench, mapping), end="")
        return 0
    if formal:
        print(
            "[eval] formal presets apply on the host engine "
            "(k/workers/temperature/wall/step follow the announced values)",
            file=sys.stderr,
        )
    if split == "test":
        print("[eval] REFUSED: test is the secret final-test set", file=sys.stderr)
        return 2
    if split not in ("visible", "selection"):
        print(f"[eval] split must be visible|selection (got {split!r})", file=sys.stderr)
        return 2

    # host side only: the engine import stays below the broker gate so
    # the envelope distribution (the client surface alone) never loads it
    from coevobench.core.evaluation.evaluate import evaluate_split, scalar_feedback

    ids = _resolve_tasks(bench, split, tasks, n)
    judge = _bench_judge(bench)
    record = evaluate_split(
        run_id=f"cli-{bench}",
        candidate=candidate,
        split=split,
        task_ids=ids,
        judge_fn=judge,
        k=k,
        out_dir=out_dir,
        workers=workers,
    )
    print(f"[eval] {bench} split={split} n={len(ids)} k={k}")
    if split == "selection":
        print(f"[eval] {scalar_feedback(record)}")
    else:
        counts = dict(record.exit_status_counts)
        score = record.score if record.score is not None else float("nan")
        print(f"[eval] score={score:.4f} exit_status={counts} n_bad={record.n_bad}")
    return 0


def _resolve_tasks(bench: str, split: str, tasks_arg: str | None, n: int | None) -> list[str]:
    splits = _load_splits(bench)
    ids = list(splits.ids(split))
    if tasks_arg:
        if split != "selection":
            raise SystemExit("--tasks is only valid with --split selection")
        anon_ids = [t.strip() for t in tasks_arg.split(",") if t.strip()]
        return _resolve_anonymous(bench, anon_ids, ids)
    if n:
        rng = random.Random(f"20260901:{bench}:{split}")
        rng.shuffle(ids)
        return ids[:n]
    return ids


def _resolve_anonymous(bench: str, anon_ids: list[str], real_ids: list[str]) -> list[str]:
    from coevobench.bench.anon import anon_to_real, load_mapping, make_anon_id

    mapping = load_mapping(bench) or {}
    if not mapping:
        mapping = {real: make_anon_id(real, bench) for real in real_ids}
    return anon_to_real(anon_ids, mapping)


def _load_splits(bench: str):
    from coevobench.core.protocol.splits import from_mapping, validate_frozen

    if bench == "toy":
        from coevobench.bench.toy.adapter import SPLITS

        return from_mapping(SPLITS)

    from coevobench.config.schema import Config
    from coevobench.core.protocol.protocol_assets import cache_root, frozen_splits

    frozen = frozen_splits(bench)
    cache = cache_root(Config()) / f"{bench}_splits.json"
    if cache.is_file():
        splits = from_mapping(json.loads(cache.read_text()))
        try:
            validate_frozen(bench, splits)
            return splits
        except ValueError:
            pass
    if frozen is None:
        raise SystemExit(f"no valid splits for {bench}; run: ceb data fetch {bench}")
    return frozen


def _sandbox_runtime():
    """The sandbox runtime for the operator-side eval paths.

    The shared selector with the loaded config: `[sandbox] runtime`
    selects when set, otherwise the environment probe — docker when
    the host has it, Modal sandboxes inside a remote run (the run
    envelope has no docker socket by design).
    """
    from coevobench.config.loader import load_config
    from coevobench.runtime.sandbox.select import sandbox_runtime

    return sandbox_runtime(load_config())


def _bench_judge(bench: str):
    """Wire the in-envelope judge (delegates to the shared builder)."""
    from pathlib import Path

    from coevobench.research.judges import env_judge_for

    data_root = Path(os.environ.get("CEB_DATA_ROOT", "/workspace/.ceb-data/tb"))
    return env_judge_for(bench, data_root=data_root)


__all__ = ["run_eval"]


def calibrate_cmd(
    bench: str,
    *,
    tasks: str = "",
    n: int = 5,
    state: Path | None = None,
) -> int:
    """Two-gate oracle calibration: the grader proves itself before it
    scores anyone (gold must resolve, empty must fail)."""
    from coevobench.bench.swe.adapter import SweBench, calibrate
    from coevobench.bench.swe.tasks import load_swe_instances
    from coevobench.config.schema import Config
    from coevobench.core.protocol.protocol_assets import cache_root

    instances = load_swe_instances()
    if tasks:
        ids = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        # Sample across repositories: one repo failing should be visible.
        by_repo: dict[str, list[str]] = {}
        for iid in instances:
            by_repo.setdefault(iid.split("__")[0], []).append(iid)
        ids = []
        for repo in sorted(by_repo)[:n]:
            ids.append(sorted(by_repo[repo])[0])
        if not ids:
            print("[calibrate] no instances to calibrate", file=sys.stderr)
            return 2

    swe = SweBench(cache_root(Config()), instances=instances)
    runtime = _sandbox_runtime()
    state = state or cache_root(Config()) / "swe_oracle_trusted.json"
    result = calibrate(swe, ids, runtime=runtime, trusted_state=state)
    for iid in result["gold_failed"]:
        print(f"[calibrate] GOLD FAILED: {iid} (grader or image broken; untrusted)")
    for iid in result["empty_passed"]:
        print(f"[calibrate] EMPTY PASSED: {iid} (tests pass without the fix!)")
    print(
        f"[calibrate] {result['n']} instances -> trusted={result['trusted']} "
        f"(state: {state})"
    )
    return 0 if result["trusted"] else 1


__all__ = ["calibrate_cmd", "run_eval"]
