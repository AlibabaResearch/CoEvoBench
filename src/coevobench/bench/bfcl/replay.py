"""One BFCL candidate session shared by broker evaluation and frozen replay."""

from __future__ import annotations

import fcntl
import json
import uuid
from dataclasses import replace
from pathlib import Path

from coevobench.bench.bfcl.provenance import UPSTREAM_COMMIT
from coevobench.bench.bfcl.runtime import read_results, run_repeat, settings, token_usage
from coevobench.bench.bfcl.tasks import component
from coevobench.core.evaluation.candidate import Candidate, candidate_serve
from coevobench.core.protocol.protocol_assets import cache_root
from coevobench.core.protocol.spec import Verdict


def aggregate_score(ids: list[str], verdicts: list[Verdict], k: int) -> float:
    groups: dict[str, list[float]] = {}
    for task_id, verdict in zip(ids * k, verdicts, strict=True):
        groups.setdefault(component(task_id), []).append(verdict.score)
    domains = {}
    for domain in ("multi_turn", "memory"):
        means = [
            sum(values) / len(values)
            for key, values in groups.items()
            if key.startswith(domain + ":")
        ]
        if means:
            domains[domain] = sum(means) / len(means)
    if len(domains) == 1:
        return next(iter(domains.values()))
    return 0.6 * domains["multi_turn"] + 0.4 * domains["memory"]


class BfclJudge:
    """Batching preserves memory dependencies and serves once across repeats."""

    aggregate_score = staticmethod(aggregate_score)

    def __init__(
        self,
        candidate: Candidate,
        request: dict | None = None,
        private_root: Path | None = None,
        gpu_ids: tuple[int, ...] = (),
        owner_dir: Path | None = None,
    ):
        self.candidate = replace(candidate, serve_shape={"max_len": 65536, **candidate.serve_shape})
        self.request = request or {}
        self.private_root = private_root or cache_root() / "bfcl-evaluations"
        self.gpu_ids = gpu_ids
        self.owner_dir = owner_dir
        self.visible_results = []
        self.provenance = {
            "implementation": "reconstructed",
            "upstream_commit": UPSTREAM_COMMIT,
            "historical_equivalence": "unverified",
        }

    def __call__(self, task_id: str) -> Verdict:
        return self.evaluate_batch([task_id], 1, 1)[0]

    def evaluate_batch(self, ids: list[str], k: int, workers: int) -> list[Verdict]:
        request = self.request
        if request.get("wall_cap_s"):
            raise ValueError("BFCL does not support a nonzero per-task wall cap")
        config_path = self.candidate.harness_src.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        defaults = {"temperature": 0.001, "max_tokens": 4096, "max_steps": 20}
        defaults.update({key: config[key] for key in defaults if key in config})
        for source, target in (
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("step_limit", "max_steps"),
        ):
            if request.get(source) is not None:
                defaults[target] = request[source]
        options = settings(
            tokenizer=str(self.candidate.weights),
            context_limit=int(self.candidate.serve_shape.get("max_len", 65536)),
            **defaults,
        )
        private = self.private_root / uuid.uuid4().hex
        private.mkdir(parents=True, mode=0o700)
        verdicts = []
        # Sessions share the local serving port and may share allocated cards.
        with (cache_root() / "bfcl-serving.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with candidate_serve(
                self.candidate, gpu_ids=self.gpu_ids, owner_dir=self.owner_dir
            ) as endpoint:
                options.update(name=endpoint["model_name"], base_url=endpoint["base_url"])
                for sample in range(k):
                    project = run_repeat(
                        ids,
                        sample,
                        options,
                        self.candidate.harness_src / "bfclharness",
                        private,
                        workers=workers,
                        owner_dir=self.owner_dir,
                    )
                    rows = {row["id"]: row for row in read_results(project)}
                    (project / "usage.json").write_text(json.dumps(token_usage(project), indent=2))
                    if request.get("split") == "visible":
                        with (project / "generation.jsonl").open() as stream:
                            for line in stream:
                                generated = json.loads(line)
                                key = generated["id"]
                                if key in rows:
                                    self.visible_results.append(
                                        {
                                            "task_id": key,
                                            "sample": sample,
                                            "correct": rows[key]["correct"],
                                            "exit_status": rows[key]["exit_status"],
                                            "trajectory": generated.get("inference_log", []),
                                        }
                                    )
                    verdicts.extend(
                        Verdict(
                            rows[key]["correct"],
                            float(rows[key]["correct"]),
                            rows[key]["exit_status"],
                        )
                        for key in ids
                    )
        return verdicts


def replay_judge(bundle: Path, bench: str, weights_override: Path | None):
    if bench != "bfcl":
        raise ValueError("BFCL replay received a different benchmark")
    candidate = Candidate.from_dir(bundle / "submission", bench="bfcl")
    if weights_override is not None:
        candidate = replace(candidate, weights=Path(weights_override))
    return BfclJudge(
        candidate,
        request={"temperature": 0.001, "max_tokens": 4096, "step_limit": 20},
        private_root=bundle.parent / "bfcl-evaluations",
    )


def live_judge(submission_dir, request: dict, workspace_root: Path):
    workspace = workspace_root.resolve()
    base_record = workspace.parent / "base-model.json"
    base = (
        Path(json.loads(base_record.read_text())["path"]).resolve()
        if base_record.is_file()
        else None
    )
    manifest_path = workspace.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if base is None and manifest.get("base_dir"):
        base = Path(manifest["base_dir"]).resolve()

    def resolve(path: str) -> Path:
        value = Path(path)
        if value == Path("/base") or value == Path("/workspace/base"):
            if base is None:
                raise ValueError("run has no operator-recorded base model")
            return base
        if value.is_relative_to("/workspace"):
            value = workspace / value.relative_to("/workspace")
        elif not value.is_absolute():
            value = workspace / value
        value = value.resolve()
        if not value.is_relative_to(workspace) and value != base:
            raise ValueError("BFCL candidate path is outside this run's workspace")
        return value

    if submission_dir:
        submission = resolve(submission_dir)
        candidate = Candidate.from_dir(submission, bench="bfcl")
        candidate = replace(candidate, weights=resolve(str(candidate.weights)))
    else:
        package = next(workspace.glob("harness/**/bfclharness/__init__.py"), None)
        if package is None:
            raise ValueError("BFCL harness package missing from workspace")
        candidate = Candidate(
            "live",
            resolve(request.get("model") or "/base"),
            package.parent.parent,
            request.get("serve_config") or {},
            "bfcl",
        )
    if not candidate.harness_src.resolve().is_relative_to(workspace):
        raise ValueError("BFCL harness escapes workspace")
    if request.get("max_model_len") is not None:
        candidate = replace(
            candidate,
            serve_shape={**candidate.serve_shape, "max_len": int(request["max_model_len"])},
        )
    manifest_path = workspace.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return BfclJudge(
        candidate,
        request,
        workspace.parent / "bfcl-evaluations",
        tuple(manifest.get("gpus", ())),
        workspace.parent,
    )
