"""Judge factories for the broker's eval.run op (design Section 7).

Two shapes:

- `judge_for`: platform-side judging of a submission artifact per task
  (fresh sandbox, official images, adapter grading). No candidate
  episodes — the caller already holds the artifact.
- `candidate_judge_for`: the A1 path — each judge call drives one full
  candidate episode: serve the candidate's weights, run its harness's
  agent loop through the welded SandboxEnvironment in the task
  sandbox, extract the real submission artifact, and grade it
  fresh-sandbox.

Both are wired per bench; the broker delegates here so server.py stays
the op surface (dispatch, bookkeeping, event ledger) and this module
owns evaluation semantics.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coevobench.runtime.broker_wire import BrokerError

if TYPE_CHECKING:
    from coevobench.core.evaluation.candidate import Candidate


def judge_for(
    bench: str,
    request: dict[str, Any],
    *,
    runtime,
    config,
    workspace_root: Path | None,
) -> Any:
    """Build the per-task judge for a platform-side evaluation."""
    if bench == "toy":
        from coevobench.bench.toy.adapter import ToyBench
        from coevobench.runtime.sandbox.base import SandboxSpec

        toy = ToyBench()

        def judge(task_id: str):
            spec = SandboxSpec(
                image=toy.public_view(task_id).image, cpu=1, mem_gib=1, ttl_s=300
            )
            sandbox = runtime.create(spec)
            try:
                sandbox.run(toy.rollout_script(task_id))
                return toy.judge(task_id, "", sandbox)
            finally:
                sandbox.kill()

        return judge
    if bench == "swe":
        from coevobench.bench.swe.adapter import SweBench
        from coevobench.runtime.sandbox.base import SandboxSpec

        swe = SweBench(_data_root(workspace_root, "swe"))
        submissions = request.get("submissions") or {}
        mirror_cfg = _instance_id_mirror(config)
        base = _mirror_base(mirror_cfg)
        from coevobench.research.catalog import swe_eval_image_catalog

        image_catalog = swe_eval_image_catalog()

        def judge(task_id: str):
            image = (
                f"{base}{task_id}" if mirror_cfg is not None else image_catalog.get(task_id)
            )
            if image is None:
                from coevobench.core.protocol.spec import Verdict

                return Verdict(
                    resolved=False, score=0.0, exit_status="EvalImageMissing"
                )
            spec = SandboxSpec(image=image, cpu=2, mem_gib=8, ttl_s=1800)
            sandbox = runtime.create(spec, ready_timeout_s=600)
            try:
                submission = submissions.get(task_id, "")
                return swe.judge(task_id, submission, sandbox)
            finally:
                sandbox.kill()

        return judge
    if bench == "tb":
        from coevobench.bench.tb.adapter import TbBench
        from coevobench.bench.tb.tasks import load_tb_tasks
        from coevobench.runtime.registry_auth import tag_source_mirror
        from coevobench.runtime.sandbox.base import SandboxSpec

        tasks = load_tb_tasks()
        tb = TbBench(_data_root(workspace_root, "tb"), tasks=tasks)
        mirror_cfg = tag_source_mirror(
            (config.eval_images.mirrors if config else []), "task_id"
        )
        base = _mirror_base(mirror_cfg)

        def judge(task_id: str):
            task = tasks.get(task_id)
            if task is None:
                from coevobench.core.protocol.spec import Verdict

                return Verdict(resolved=False, score=0.0, exit_status="TaskDataMissing")
            image = task.get("image") or (f"{base}{task_id}" if base else None)
            if image is None:
                from coevobench.core.protocol.spec import Verdict

                return Verdict(
                    resolved=False, score=0.0, exit_status="EvalImageMissing"
                )
            # terminal-bench graders install their own venv at runtime
            # (the testing venv is separate from the agent's by design),
            # so the judge sandbox carries network like the official
            # harness.
            spec = SandboxSpec(
                image=image, cpu=2, mem_gib=8, ttl_s=1800, network=True
            )
            sandbox = runtime.create(spec, ready_timeout_s=300)
            try:
                return tb.judge(task_id, "", sandbox)
            finally:
                sandbox.kill()

        return judge
    raise BrokerError(f"bench {bench!r} judging is not wired into the broker yet")


def candidate_judge_for(
    bench: str,
    submission_dir,
    request: dict[str, Any],
    *,
    runtime,
    workspace_root: Path | None,
    config=None,
) -> Any:
    """A1: judge a researcher candidate (weights + harness pair).

    Each judge call drives one full candidate episode against the
    served candidate weights. Two episode shapes by bench family:

    - sandboxed (swe, tb): the platform spawns the task sandbox, the
      candidate harness's agent loop runs inside it through the welded
      SandboxEnvironment; the submission is the sandbox's state (swe:
      the git diff, extracted while the sandbox lives; tb: judged on
      the sandbox directly — verifiers upload after the agent).
    - in-process (appworld, bfcl, tau): the episode runs inside the
      judge process against the served endpoint; the submission is the
      trajectory record the bench adapter grades.

    Two candidate forms: a frozen `submission_dir` (weights + harness
    + serve_config.json), or the live workspace candidate — `model`
    pointing at weights under the workspace and the harness taken
    from the workspace tree the researcher is editing (edits apply on
    the next episode).
    """
    if bench not in ("swe", "tb", "appworld", "bfcl", "tau"):
        raise BrokerError(
            f"candidate evaluation is not defined for bench {bench!r}: "
            "toy runs script rollouts without a harness or model calls"
        )

    from coevobench.core.evaluation.candidate import Candidate

    def _candidate() -> Candidate:
        if submission_dir:
            from dataclasses import replace

            candidate = Candidate.from_dir(Path(submission_dir), bench=bench)
            owner = workspace_root.parent if workspace_root else None
            return replace(candidate, owner_dir=owner)
        model = request.get("model")
        if not model:
            raise BrokerError(
                "candidate evaluation needs a submission_dir or model"
            )
        workspace = workspace_root or Path("/root/.ceb")
        harness_root = _workspace_harness_root(workspace, bench)
        overrides = dict(request.get("serve_config") or {})
        for key in ("tool_call_parser", "reasoning_parser", "chat_template"):
            if request.get(key) is not None:
                overrides[key] = request[key]
        if request.get("max_model_len") is not None:
            overrides["max_len"] = request["max_model_len"]
        # the researcher speaks envelope paths (/base, /workspace/...,
        # relative-to-workspace); the broker serves weights host-side,
        # so the model path crosses the same domain translation every
        # other researcher-supplied path does
        from coevobench.research.context import BrokerContext

        weights = BrokerContext._translate_model_path(workspace_root, model)
        if overrides.get("chat_template"):
            template = BrokerContext._translate_model_path(
                workspace_root, overrides["chat_template"]
            )
            if not template.is_file():
                raise BrokerError("serving chat_template file does not exist")
            overrides["chat_template"] = str(template)
        return Candidate(
            label=str(Path(weights).name),
            weights=weights,
            harness_src=harness_root,
            serve_shape=overrides,
            bench=bench,
            owner_dir=(workspace_root.parent if workspace_root
                       and (workspace_root.parent / "manifest.json").is_file() else None),
        )

    limits = {
        "step_limit": int(request.get("step_limit", 80)),
        "wall_cap_s": int(request.get("wall_cap_s", 900)),
        "max_tokens": int(request.get("max_tokens", 5120)),
        "temperature": (
            float(request["temperature"])
            if request.get("temperature") is not None
            else None
        ),
    }

    if bench in ("swe", "tb"):
        return _sandboxed_candidate_judge(
            bench, _candidate(), request, runtime, workspace_root, limits, config
        )
    if bench in ("appworld", "bfcl"):
        # the policy-sandbox protocol: the candidate's harness runs in an
        # isolated sandbox and reaches the model ONLY through the
        # platform-injected completion — it never sees an endpoint, an
        # identity, or the official grader
        from coevobench.core.evaluation.candidate import Candidate as _C

        if submission_dir:
            candidate = _C.from_dir(Path(submission_dir), bench=bench)
        else:
            candidate = _candidate()
        return _policy_sandbox_judge(bench, candidate, request, workspace_root, runtime)
    return _inprocess_candidate_judge(
        bench, _candidate(), request, workspace_root, limits
    )


def _workspace_harness_root(workspace: Path, bench: str) -> Path:
    """The live harness tree the researcher is editing (by bench)."""
    packages = {
        "swe": "minisweagent/__init__.py",
        "tb": "minisweagent/__init__.py",
        "appworld": "appworldharness/__init__.py",
        "bfcl": "bfclharness/__init__.py",
        "tau": "tauharness/__init__.py",
    }
    rel = packages[bench]
    harness = next((p for p in workspace.rglob(rel)), None)
    if harness is None:
        raise BrokerError(
            f"no harness package ({rel.split('/')[0]}) under the workspace {workspace}"
        )
    return harness.parent.parent


def _sandboxed_candidate_judge(
    bench: str,
    candidate: Candidate,
    request: dict[str, Any],
    runtime,
    workspace_root: Path | None,
    limits: dict,
    config,
) -> Any:
    """SWE/TB: one sandboxed episode per task, graded on the sandbox."""
    from coevobench.config.schema import Config
    from coevobench.core.evaluation.candidate import run_candidate_episode
    from coevobench.core.protocol.spec import Verdict

    if bench == "swe":
        from coevobench.bench.swe.adapter import SweBench
        from coevobench.bench.swe.tasks import load_swe_instances
        from coevobench.core.protocol.protocol_assets import cache_root

        instances = load_swe_instances()
        bench_adapter = SweBench(cache_root(Config()), instances=instances)
        tasks = instances
    else:
        from coevobench.bench.tb.adapter import TbBench
        from coevobench.bench.tb.tasks import load_tb_tasks
        from coevobench.runtime.registry_auth import tag_source_mirror

        tasks = load_tb_tasks()
        # task records already carry resolved image references (shaped at
        # materialize time); the task-id mirror is the fallback for
        # records without one
        mirrors = config.eval_images.mirrors if config else []
        base = _mirror_base(tag_source_mirror(mirrors, "task_id"))
        bench_adapter = TbBench(_data_root(workspace_root, "tb"), tasks=tasks)

    port = int(request.get("serve_port", 18900))
    if bench == "tb":
        from coevobench.research.eval_ops import _prepare_eval

        _, wanted, _ = _prepare_eval({"bench": "tb", "split": "visible", **request})
        if any(not tasks.get(t, {}).get("image") and not base for t in wanted):
            raise BrokerError(
                "TB evaluation images are missing; operator must run ceb data build-images tb "
                "or configure a task-ID image mirror before evaluation"
            )
    serving = _ServingPool(candidate, port)

    def judge(task_id: str):
        task_data = tasks.get(task_id)
        if task_data is None:
            return Verdict(False, 0.0, "TaskDataMissing")
        if bench == "tb":
            # tb records must carry the image (or the task-id mirror
            # resolves it); there is no generic fallback for verifiers
            image = task_data.get("image") or (
                f"{base}{task_id}" if base else None
            )
            if image is None:
                return Verdict(False, 0.0, "EvalImageMissing")
            task_data = {**task_data, "image": image}
        with serving.endpoint() as served:
            episode = run_candidate_episode(
                candidate,
                task_id,
                served,
                bench=bench,
                task_data=task_data,
                sandbox_runtime=runtime,
                **limits,
            )
            episode_sandbox = runtime.connect(episode.get("sandbox_id", ""))
            try:
                if bench == "swe":
                    # the real git diff while the sandbox is alive (integrity #3)
                    diff_out = episode_sandbox.run(
                        "cd /testbed && git add -N . && git diff HEAD "
                        "2>/dev/null | head -c 200000",
                        timeout_s=60,
                    )
                    return bench_adapter.judge(task_id, diff_out["text"], episode_sandbox)
                # tb: verifiers upload onto the episode sandbox and run there
                return bench_adapter.judge(task_id, "", episode_sandbox)
            finally:
                episode_sandbox.kill()

    # Keep an outer lease across all waves, including workers=1. Per-task
    # leases must not stop the model between sequential episodes.
    judge.evaluation_session = serving.endpoint
    return judge


def _policy_sandbox_judge(
    bench: str,
    candidate: Candidate,
    request: dict[str, Any],
    workspace_root: Path | None,
    runtime,
) -> Any:
    """appworld/bfcl: the replay engines from the cloud-experiment line.

    The candidate's harness executes inside an isolated policy sandbox
    (mailbox protocol — code and observations cross, nothing else);
    the model is reached only through the platform-injected
    completion; the official evaluator/world run operator-side.
    Paths resolve envelope->host the same way every other researcher
    path does.
    """
    if workspace_root is None:
        raise BrokerError(f"{bench} evaluation needs a run workspace")
    if bench == "appworld":
        from coevobench.bench.appworld.replay import live_judge

        return live_judge(
            request.get("submission_dir"),
            {**request, "split": request.get("split", "")},
            workspace_root,
            runtime=runtime,
        )
    from coevobench.bench.bfcl.replay import live_judge as bfcl_live_judge

    return bfcl_live_judge(
        request.get("submission_dir"), request, workspace_root
    )


def _inprocess_candidate_judge(
    bench: str,
    candidate: Candidate,
    request: dict[str, Any],
    workspace_root: Path | None,
    limits: dict,
) -> Any:
    """tau: the episode runs in this process, against the served
    candidate; the adapter grades the trajectory record."""
    from coevobench.core.protocol.spec import Verdict

    if bench == "tau":
        from coevobench.bench.tau import runtime as tau_rt

        def judge(task_id: str):
            try:
                with _served(candidate, request) as served:
                    import os

                    os.environ["TAU_AGENT_BASE_URL"] = served["base_url"]
                    os.environ["TAU_AGENT_LLM"] = f"hosted_vllm/{served['model_name']}"
                    name = tau_rt.register_agent(candidate.harness_src)
                    tasks = tau_rt.load_tasks([task_id])
                    settings = tau_rt.load_settings(
                        temperature=limits["temperature"] or 0.0,
                        max_tokens=limits["max_tokens"],
                    )
                    record = tau_rt.run_simulation(tasks[0], settings, name)
                    checks = (record.get("reward_info") or {}).get("checks") or []
                    submission = json.dumps({"checks": checks})
                from coevobench.bench.tau.adapter import TauBench

                return TauBench().judge(task_id, submission, None)
            except Exception as exc:
                return Verdict(
                    False, 0.0, "HarnessError:" + type(exc).__name__
                )
        return judge

    raise BrokerError(f"in-process candidate evaluation for {bench!r} is not wired")


@contextlib.contextmanager
def _served(candidate, request: dict[str, Any]):
    """Serve the candidate for one in-process episode round."""
    from coevobench.core.evaluation.candidate import candidate_serve

    port = int(request.get("serve_port", 18900))
    with candidate_serve(candidate, port) as served:
        yield served


def _appworld_task(task) -> dict:
    """The task record shape the appworld harness Agent expects."""
    instruction = (
        getattr(task, "instruction", None)
        or (task.task_data or {}).get("instruction", "")
    )
    main_user = getattr(task, "main_user", None) or ""
    if not instruction:
        data = getattr(task, "task_data", None) or {}
        instruction = data.get("instruction", "")
        main_user = data.get("supervisor", main_user)
    return {"instruction": instruction, "supervisor": main_user}

class _ServingPool:
    """One vLLM serve shared across every judge call of one evaluation.

    The serve must outlive a single episode — starting vLLM per task
    costs minutes of cold start per task and (worse) concurrent judge
    workers racing the lazy init used to stack two serves onto the
    same GPUs until one OOMed. The lock serializes the bring-up; the
    refcount tears the serve down only after the last episode leaves.
    """

    def __init__(self, candidate, port: int) -> None:
        import threading

        self._candidate = candidate
        self._port = port
        self._lock = threading.Lock()
        self._stack: contextlib.ExitStack | None = None
        self._endpoint = None
        self._holders = 0

    @contextlib.contextmanager
    def endpoint(self):
        with self._lock:
            if self._endpoint is None:
                from coevobench.core.evaluation.candidate import candidate_serve

                with contextlib.ExitStack() as stack:
                    self._endpoint = stack.enter_context(
                        candidate_serve(self._candidate, self._port)
                    )
                    self._stack = stack.pop_all()
            self._holders += 1
        try:
            yield self._endpoint
        finally:
            with self._lock:
                self._holders -= 1
                if self._holders <= 0 and self._stack is not None:
                    self._stack.close()
                    self._stack = None
                    self._endpoint = None
                    self._holders = 0


class _JudgeSandbox:
    """Adapter presenting the episode sandbox to the bench judge."""

    def __init__(self, episode: dict, runtime) -> None:
        self._episode = episode
        self._runtime = runtime

    def write_files(self, files) -> None:
        sb = self._runtime.connect(self._episode.get("sandbox_id", ""))
        try:
            sb.write_files(files)
        finally:
            sb.close()

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None):
        sb = self._runtime.connect(self._episode.get("sandbox_id", ""))
        try:
            return sb.run(command, timeout_s=timeout_s, cwd=cwd)
        finally:
            sb.close()


def _data_root(workspace_root: Path | None, bench: str) -> Path:
    root = workspace_root or Path("/root/.ceb")
    return root / "data" / bench


def _instance_id_mirror(config) -> object | None:
    """The SWE fast-path mirror (tag_source=instance_id), if any."""
    if config is None:
        return None
    from coevobench.runtime.registry_auth import tag_source_mirror

    return tag_source_mirror(config.eval_images.mirrors, "instance_id")


def _mirror_base(mirror) -> str:
    if mirror is None:
        return ""
    rw = mirror.rewrite.rstrip("/")
    return rw if rw.endswith(":") else rw + ":"


def default_sandbox_runtime():
    """The sandbox runtime for envelope-side judges: the shared selector
    with no config — docker when the host has it, Modal sandboxes
    inside a remote run.

    The run envelope has no docker socket (by design); a remote (Modal)
    run gets co-located Modal sandboxes — env-ops never cross back to
    the operator host. Operator-side call sites resolve through the
    same selector WITH the config, so `[sandbox] runtime` selects
    there.
    """
    from coevobench.runtime.sandbox.select import sandbox_runtime

    return sandbox_runtime()


def env_judge_for(bench: str, *, data_root=None):
    """The in-envelope judge: co-located sandboxes, bench adapters.

    Shares the judge-body construction with the broker path (the same
    sandbox spawn/kill discipline and Verdict plumbing); the difference
    is where the task data lives — the workspace's CEB_DATA_ROOT
    rather than the operator host's cache.
    """
    from pathlib import Path

    from coevobench.runtime.sandbox.base import SandboxSpec

    runtime = default_sandbox_runtime()

    if bench == "toy":
        from coevobench.bench.toy.adapter import ToyBench

        toy = ToyBench()

        def judge(task_id: str):
            spec = SandboxSpec(
                image=toy.public_view(task_id).image, cpu=1, mem_gib=1, ttl_s=300
            )
            sandbox = runtime.create(spec)
            try:
                sandbox.run(toy.rollout_script(task_id))
                return toy.judge(task_id, "", sandbox)
            finally:
                sandbox.kill()

        return judge

    if bench == "tb":
        from coevobench.bench.tb.adapter import TbBench
        from coevobench.bench.tb.tasks import load_tb_tasks

        tasks = load_tb_tasks()
        root = data_root or Path("/workspace/.ceb-data/tb")
        tb = TbBench(root, tasks=tasks)

        def judge(task_id: str):
            task = tasks.get(task_id)
            if task is None:
                from coevobench.core.protocol.spec import Verdict

                return Verdict(False, 0.0, "TaskDataMissing")
            image = task.get("image")
            if not image:
                from coevobench.core.protocol.spec import Verdict

                return Verdict(False, 0.0, "EvalImageMissing")
            # terminal-bench graders install their venv at runtime (the
            # testing venv is separate from the agent's by design), so
            # the judge sandbox carries network like the official harness.
            spec = SandboxSpec(
                image=image, cpu=2, mem_gib=8, ttl_s=1800, network=True
            )
            sandbox = runtime.create(spec, ready_timeout_s=600)
            try:
                return tb.judge(task_id, "", sandbox)
            finally:
                sandbox.kill()

        return judge

    raise RuntimeError(
        f"bench {bench!r} evaluation needs a served candidate; "
        "wire it via the bench adapter (engine: core/evaluate)"
    )


__all__ = [
    "candidate_judge_for",
    "default_sandbox_runtime",
    "env_judge_for",
    "judge_for",
]
