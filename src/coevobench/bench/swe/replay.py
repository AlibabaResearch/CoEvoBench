"""SWE final-eval replay binding (the bench layer's judge factory).

Serves the (possibly reverted) bundle weights, drives the frozen
submission's harness over each task through the welded
SandboxEnvironment, extracts the real submission artifact, and grades
it fresh-sandbox. core/final_eval stays bench-agnostic: it consumes
this through the replay-binding registry, and only the upper layers
(cli, broker) import this module to register it.
"""

from __future__ import annotations

import contextlib
from pathlib import Path


def replay_judge(bundle: Path, bench: str, weights_override: Path | None):
    from coevobench.bench.swe.adapter import SweBench
    from coevobench.bench.swe.tasks import load_swe_instances
    from coevobench.config.schema import Config
    from coevobench.core.evaluation.candidate import (
        Candidate,
        candidate_serve,
        run_candidate_episode,
    )
    from coevobench.core.protocol.protocol_assets import cache_root
    from coevobench.core.protocol.spec import Verdict

    submission = bundle / "submission"
    candidate = Candidate.from_dir(submission, bench=bench)
    if weights_override is not None:
        from dataclasses import replace

        candidate = replace(candidate, weights=Path(weights_override))
    instances = load_swe_instances()
    from coevobench.runtime.sandbox.select import sandbox_runtime

    runtime = sandbox_runtime(Config())
    swe = SweBench(cache_root(Config()), instances=instances)
    serving: dict | None = None

    def judge(task_id: str):
        nonlocal serving
        task_data = instances.get(task_id)
        if task_data is None:
            return Verdict(False, 0.0, "TaskDataMissing")
        with contextlib.ExitStack() as stack:
            if serving is None:
                # serve once for the whole Test pass
                serving_ref = stack.enter_context(candidate_serve(candidate))
                _ReplayState.endpoint = serving_ref
                serving = serving_ref
            served = _ReplayState.endpoint
            episode = run_candidate_episode(
                candidate, task_id, served,
                bench=bench, task_data=task_data, sandbox_runtime=runtime,
            )
            episode_sandbox = runtime.connect(episode.get("sandbox_id", ""))
            try:
                diff_out = episode_sandbox.run(
                    "cd /testbed && git add -N . && git diff HEAD "
                    "2>/dev/null | head -c 200000",
                    timeout_s=60,
                )
                return swe.judge(task_id, diff_out["text"], episode_sandbox)
            finally:
                episode_sandbox.kill()

    return judge


class _ReplayState:
    endpoint = None
