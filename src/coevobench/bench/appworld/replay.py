"""AppWorld candidate evaluation shared by the broker and frozen final replay."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from coevobench.bench.appworld.assets import verify_source
from coevobench.bench.appworld.policy import PolicyError, PolicySession
from coevobench.bench.appworld.runtime import WorldSession, settings
from coevobench.bench.appworld.tasks import load_tasks
from coevobench.core.evaluation.candidate import Candidate, candidate_serve
from coevobench.core.protocol.protocol_assets import cache_root
from coevobench.core.protocol.spec import Verdict


class _LazySandboxRuntime:
    """Resolves the sandbox family at first use, not at judge build time.

    Judge construction happens in contexts with no sandbox needs (freeze
    verification, bundle preparation); resolving eagerly dragged a
    provider lookup into all of them.
    """

    def __init__(self, owner_dir: Path | None = None) -> None:
        self._owner_dir = owner_dir
        self._inner = None

    def _resolve(self):
        if self._inner is None:
            from coevobench.config.loader import load_config
            from coevobench.runtime.registry import resolve

            cfg = load_config()
            default = (
                "modal" if cfg.runhost.provider == "modal" else cfg.sandbox.runtime
            )
            name = os.environ.get("CEB_APPWORLD_SANDBOX", default or "docker")
            self._inner = resolve("sandbox", name)(owner_dir=self._owner_dir)
        return self._inner

    def create(self, spec, **kwargs):
        return self._resolve().create(spec, **kwargs)

    def connect(self, sandbox_id):
        return self._resolve().connect(sandbox_id)


def sandbox_runtime(owner_dir: Path | None = None):
    return _LazySandboxRuntime(owner_dir)


class CompletionClient:
    """Only message content crosses from candidate to operator; endpoint/limits stay pinned."""

    def __init__(self, endpoint: dict, options: dict, deadline=float("inf")):
        self.endpoint, self.options = endpoint, options
        self.deadline = deadline
        self.usage = []

    def __call__(self, messages: list[dict]) -> dict:
        if (
            not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(row, dict) and row.get("role") in ("system", "user", "assistant", "tool")
                for row in messages
            )
        ):
            raise PolicyError("invalid completion messages")
        body = {
            "model": self.endpoint["model_name"],
            "messages": messages,
            "temperature": self.options["temperature"],
            "seed": self.options["seed"],
            "max_tokens": self.options["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        token = self.endpoint.get("api_key", "")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(
            self.endpoint["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        # Match the initial harness's two retries for transient model transport errors.
        for attempt in range(3):
            try:
                remaining = self.deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("run deadline reached")
                with urllib.request.urlopen(
                    request, timeout=min(self.options["request_timeout_s"], remaining)
                ) as response:
                    payload = json.loads(response.read(8 * 1024 * 1024))
                break
            except OSError as exc:
                if attempt == 2 or (
                    isinstance(exc, urllib.error.HTTPError)
                    and exc.code not in (408, 429, 500, 502, 503, 504)
                ):
                    raise
                time.sleep(0.5 * (attempt + 1))
        usage = payload.get("usage") or {}
        self.usage.append(usage)
        return {"content": payload["choices"][0]["message"].get("content") or "", "usage": usage}


def scene_metrics(ids: list[str], verdicts: list[Verdict], k: int) -> dict:
    successes = []
    for sample in range(k):
        groups = {}
        for key, verdict in zip(
            ids, verdicts[sample * len(ids) : (sample + 1) * len(ids)], strict=True
        ):
            groups.setdefault(key.rsplit("_", 1)[0], []).append(verdict.resolved)
        successes.extend(all(values) for values in groups.values() if len(values) == 3)
    return {
        "task_goal_completion": 100 * sum(v.score for v in verdicts) / len(verdicts),
        "scenario_goal_completion": 100 * sum(successes) / len(successes) if successes else None,
        "n_complete_scene_trials": len(successes),
    }


class AppWorldJudge:
    def __init__(
        self,
        candidate: Candidate,
        request=None,
        private_root=None,
        gpu_ids=(),
        owner_dir=None,
        *,
        runtime=None,
        served=None,
    ):
        self.request = request or {}
        shape = {"max_len": 65536, **candidate.serve_shape}
        if self.request.get("formal"):
            shape["max_len"] = 65536
        self.candidate = replace(candidate, serve_shape=shape)
        self.private_root = Path(private_root or cache_root() / "appworld-evaluations")
        self.owner_dir, self.gpu_ids = owner_dir, gpu_ids
        self.runtime = runtime or sandbox_runtime(owner_dir)
        self.served = served
        self.visible_results = []
        self.metrics = {}
        self.provenance = {}

    def __call__(self, task_id):
        return self.evaluate_batch([task_id], 1, 1)[0]

    def _options(self):
        path = self.candidate.harness_src.parent / "config.json"
        config = json.loads(path.read_text()) if path.is_file() else {}
        values = {
            name: config[name]
            for name in ("temperature", "max_tokens", "max_steps", "seed")
            if name in config
        }
        for source, target in (
            ("wall_cap", "wall_cap_s"),
            ("request_timeout", "request_timeout_s"),
        ):
            if source in config:
                values[target] = config[source]
        for source, target in (
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("step_limit", "max_steps"),
            ("wall_cap_s", "wall_cap_s"),
        ):
            if self.request.get(source) is not None:
                values[target] = self.request[source]
        if self.request.get("formal"):
            values = dict(
                temperature=0.0,
                max_tokens=3000,
                max_steps=50,
                seed=100,
                wall_cap_s=0.0,
                request_timeout_s=600,
            )
        return settings(**values)

    def evaluate_batch(self, ids, k, workers):
        self.provenance = verify_source()
        tasks = load_tasks()
        if set(ids) - tasks.keys():
            raise ValueError("unknown AppWorld task")
        options = self._options()
        private = self.private_root / uuid.uuid4().hex
        private.mkdir(parents=True, mode=0o700)
        (private / "provenance.json").write_text(json.dumps(self.provenance, indent=2))
        pairs = [(key, sample) for sample in range(k) for key in ids]
        with (cache_root() / "appworld-serving.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            serving = (
                contextlib.nullcontext(self.served)
                if self.served is not None
                else candidate_serve(
                    self.candidate,
                    gpu_ids=self.gpu_ids,
                    owner_dir=self.owner_dir,
                )
            )
            with serving as endpoint, ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(
                    pool.map(
                        lambda pair: self._episode(pair, endpoint, options, private),
                        pairs,
                    )
                )
        verdicts = [result[0] for result in results]
        rows = [result[1] for result in results]
        self.metrics = scene_metrics(ids, verdicts, k)
        (private / "results.json").write_text(
            json.dumps({"metrics": self.metrics, "rows": rows}, indent=2)
        )
        if self.request.get("split") == "visible":
            self.visible_results = rows
        return verdicts

    def _episode(self, pair, endpoint, options, private):
        task_id, sample = pair
        directory = private / f"{task_id}-{sample}"
        started = time.monotonic()
        world = policy = None
        row = {"task_id": task_id, "sample": sample, "trajectory": []}
        try:
            deadline = time.time() + 86400
            if self.owner_dir is not None:
                state = json.loads((self.owner_dir / "lifecycle.json").read_text())
                deadline = state["deadline_at"]
                if state["phase"] != "running" or time.time() >= deadline:
                    raise TimeoutError("run deadline reached")
            world = WorldSession(task_id, directory, owner_dir=self.owner_dir)
            policy = PolicySession(
                self.runtime,
                os.environ.get("CEB_APPWORLD_POLICY_IMAGE", "coevobench-appworld-policy:latest"),
                self.candidate.harness_src,
                world.public,
                options,
                ttl_s=max(1, int(deadline - time.time())),
            )
            completion = CompletionClient(endpoint, options, deadline)
            observation = None
            row["stop_reason"] = "step_limit"
            for step in range(options["max_steps"]):
                if options["wall_cap_s"] and time.monotonic() - started >= options["wall_cap_s"]:
                    row["stop_reason"] = "wall_cap"
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("run deadline reached")
                code = policy.next_code(observation, completion, timeout_s=remaining)
                output = world.call("execute", code=code)
                observation = output["observation"]
                row["trajectory"].append({"step": step, "code": code, "observation": observation})
                if output["completed"]:
                    row["stop_reason"] = "task_completed"
                    break
            grade = world.call("evaluate")
            row["usage"] = completion.usage
            (directory / "official-evaluation.json").write_text(json.dumps(grade, indent=2))
            verdict = Verdict(bool(grade["tgc"]), float(grade["tgc"]), "Submitted")
        except PolicyError as exc:
            verdict = Verdict(False, 0.0, "HarnessError:" + type(exc).__name__)
            row["error"] = str(exc)
        except Exception as exc:
            verdict = Verdict(False, 0.0, "WorkerError:" + type(exc).__name__)
            row["error"] = type(exc).__name__
        finally:
            cleanup_errors = []
            for resource in (policy, world):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        cleanup_errors.append(type(exc).__name__)
            if cleanup_errors:
                row["cleanup_errors"] = cleanup_errors
                verdict = Verdict(False, 0.0, "WorkerError:CleanupFailed")
        row.update(
            correct=verdict.resolved,
            exit_status=verdict.exit_status,
            wall_seconds=time.monotonic() - started,
        )
        return verdict, row


def live_judge(submission_dir, request, workspace_root, *, runtime=None):
    workspace = workspace_root.resolve()
    base_path = workspace.parent / "base-model.json"
    base = (
        Path(json.loads(base_path.read_text())["path"]).resolve() if base_path.is_file() else None
    )
    manifest_path = workspace.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if base is None and manifest.get("base_dir"):
        base = Path(manifest["base_dir"]).resolve()

    def resolve(value):
        path = Path(value)
        if str(path) in ("/base", "/workspace/base"):
            if base is None:
                raise ValueError("run has no operator-recorded base weights")
            return base
        if path.is_relative_to("/workspace"):
            path = workspace / path.relative_to("/workspace")
        elif not path.is_absolute():
            path = workspace / path
        path = path.resolve()
        if not path.is_relative_to(workspace) and path != base:
            raise ValueError("AppWorld candidate path is outside the run workspace")
        return path

    if submission_dir:
        candidate = Candidate.from_dir(resolve(submission_dir), "appworld")
        candidate = replace(candidate, weights=resolve(str(candidate.weights)))
    else:
        packages = list(workspace.glob("harness/**/appworldharness/__init__.py"))
        if len(packages) != 1:
            raise ValueError("AppWorld needs exactly one harness package")
        candidate = Candidate(
            "live",
            resolve(request.get("model") or "/base"),
            packages[0].parent.parent,
            request.get("serve_config") or {},
            "appworld",
        )
    if not candidate.harness_src.resolve().is_relative_to(workspace):
        raise ValueError("AppWorld harness escapes workspace")
    if request.get("max_model_len") is not None:
        candidate = replace(
            candidate,
            serve_shape={**candidate.serve_shape, "max_len": int(request["max_model_len"])},
        )
    path = workspace.parent / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    return AppWorldJudge(
        candidate,
        request,
        workspace.parent / "appworld-evaluations",
        tuple(manifest.get("gpus", ())),
        workspace.parent,
        runtime=runtime,
    )


def replay_judge(bundle, bench, weights_override, *, owner_dir=None):
    if bench != "appworld":
        raise ValueError("AppWorld replay received a different benchmark")
    candidate = Candidate.from_dir(bundle / "submission", bench)
    if weights_override is not None:
        candidate = replace(candidate, weights=Path(weights_override))
    return AppWorldJudge(
        candidate, {"formal": True}, bundle.parent / "appworld-evaluations", owner_dir=owner_dir
    )
