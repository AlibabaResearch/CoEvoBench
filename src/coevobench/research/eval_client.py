"""In-envelope evaluation client: forward to the host broker.

The frozen ./eval entry runs INSIDE the run envelope where the
researcher lives — so it must not judge anything, hold split data, or
touch secrets (design doc Sections 2.1.1, 5.2). It packages the
request (split, candidate identity, task selection, the researcher's
declared serving recipe and decoding overrides) and sends it over the
broker socket; the host runs the real engine (core/evaluate with the
bench adapter and SandboxRuntime) and returns the scalar or the
visible feedback.

Self-evaluation settings are the researcher's to choose (the paper
announces final-evaluation settings separately); everything declared
here is recorded with the run — subsets, k, and the serving recipe.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BROKER_SOCKET = "/broker.sock"


def _serving_recipe(model: str | None, explicit: str | None) -> dict | None:
    """An explicit recipe replaces the workspace default; paths are recipe-relative."""
    workspace = Path(os.environ.get("CEB_WORKSPACE", Path.cwd()))
    path = Path(explicit) if explicit else workspace / "serve_config.json"
    if not explicit and (not model or not path.is_file()):
        return None
    recipe = json.loads(path.read_text())
    if not isinstance(recipe, dict):
        raise ValueError("serving recipe must be a JSON object")
    template = recipe.get("chat_template")
    if template:
        if not isinstance(template, str):
            raise ValueError("chat_template must be a path string")
        # Do not resolve symlinks here: /base and /workspace name mounts
        # whose host-side paths are known only to the broker.
        recipe["chat_template"] = os.path.abspath(path.parent / template)
    return recipe


def run_eval(
    bench: str,
    split: str,
    *,
    k: int = 1,
    tasks: str | None = None,
    candidate: str = "cand0",
    workers: int = 1,
    n: int | None = None,
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
    run_id: str = "",
) -> int:
    if split == "test":
        print("[eval] REFUSED: test is the secret final-test set", file=sys.stderr)
        return 2
    if split not in ("visible", "selection"):
        print(f"[eval] split must be visible|selection (got {split!r})", file=sys.stderr)
        return 2

    from coevobench.runtime.broker_wire import BrokerError, call

    request: dict = {
        "op": "eval.run",
        "bench": bench,
        "split": split,
        "k": k,
        "candidate": candidate,
        "workers": workers,
    }
    if run_id:
        request["run_id"] = run_id
    if tasks:
        if split != "selection":
            print("[eval] ERROR: --tasks is only valid with --split selection.", file=sys.stderr)
            return 2
        request["tasks"] = [t.strip() for t in tasks.split(",") if t.strip()]
    if n is not None:
        request["n"] = n

    # The candidate's identity: weights + serving recipe. A model_path
    # pins the weights under evaluation; the serve config declares the
    # recipe the candidate asks the platform to replay.
    if model:
        request["model"] = model
    try:
        recipe = _serving_recipe(model, serve_config)
    except (ValueError, OSError) as exc:
        print(f"[eval] invalid serving recipe: {exc}", file=sys.stderr)
        return 2
    if recipe is not None:
        request["serve_config"] = recipe
    if chat_template:
        chat_template = os.path.abspath(chat_template)
    for key, value in (
        ("tool_call_parser", tool_call_parser),
        ("reasoning_parser", reasoning_parser),
        ("chat_template", chat_template),
        ("temperature", temperature),
        ("max_tokens", max_tokens),
        ("wall_cap_s", wall_cap),
        ("step_limit", step_limit),
        ("max_model_len", max_model_len),
    ):
        if value is not None:
            request[key] = value
    if formal:
        request["formal"] = True

    if list_tasks:
        # In-envelope the split definitions are not installed (bench
        # secrets); the workspace's anonymous pool file is the only
        # source of truth, and it must satisfy the request on its own.
        # The bench imports below are the HOST fallback, so they stay
        # under the file check — the envelope never reaches them.
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
            print("[eval] unknown bench", file=sys.stderr)
            return 2
        mapping = load_mapping(bench) or generate_mapping(bench, list(frozen.selection))
        print(pool_file_contents(bench, mapping), end="")
        return 0

    socket_path = os.environ.get("CEB_BROKER_SOCKET", BROKER_SOCKET)
    try:
        # The host owns the split size and formal presets. Ask it for the
        # effective workload, including verifier/startup time, before waiting.
        plan = call(socket_path, {**request, "op": "eval.plan"}, timeout=30)
        if "error" in plan:
            print(f"[eval] ERROR: {plan['error']}", file=sys.stderr)
            return 4
        response = call(socket_path, request, timeout=plan["timeout_s"])
    except TimeoutError:
        print(
            "[eval] broker wait timed out; check the run's evaluation records "
            "before retrying (the host may still be evaluating)", file=sys.stderr,
        )
        return 3
    except (BrokerError, OSError) as exc:
        print(f"[eval] broker unavailable: {exc}", file=sys.stderr)
        return 3
    if "error" in response:
        print(f"[eval] ERROR: {response['error']}", file=sys.stderr)
        return 4

    print(f"[eval] {bench} split={split} n={response.get('n_tasks')} k={k}")
    if split == "selection":
        print(f"[eval] {response.get('feedback', 'no feedback')}")
    else:
        counts = response.get("exit_status_counts")
        bad = response.get("n_bad")
        print(f"[eval] score={response.get('score')} exit_status={counts} n_bad={bad}")
    return 0


__all__ = ["run_eval"]
