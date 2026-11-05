"""Example researcher CLI: the teaching template (design document Section 3.5).

Walking through this file is how you add a new researcher CLI. A real
adapter does two things and nothing else:

1. provision() — idempotent setup inside the run envelope at init time.
2. entrypoint() — the (argv, env) pair the RunHost will exec.

This example wraps a plain python agent loop reading tasks from stdin,
which is enough to drive the toy bench end to end without any commercial
CLI. It also demonstrates the audit-hook contract: every command the
researcher executes must be streamed as one JSON line to stderr.
"""

from __future__ import annotations

from coevobench.runtime.researcher.base import (
    Endpoint,
    ProvisionContext,
    ResearcherConfig,
)
from coevobench.runtime.runhost.base import Entrypoint

_LOOP = """#!/usr/bin/env python3
import json
import sys
import time


def audit(tool: str, command: str) -> None:
    print(json.dumps({"time": time.time(), "tool": tool, "command": command}),
          file=sys.stderr, flush=True)


for line in sys.stdin:
    task = json.loads(line)
    audit("TaskStart", task.get("task_id", ""))
    # A real adapter drives the model here; the example just echoes a
    # deterministic answer so the smoke loop finishes without credentials.
    audit("TaskDone", task.get("task_id", ""))
    print(json.dumps({"task_id": task.get("task_id"), "ok": True}), flush=True)
"""


class ExampleCLI:
    name = "example"

    def provision(self, config: ResearcherConfig, ctx: ProvisionContext) -> None:
        from pathlib import Path

        workspace = Path(ctx.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "example_loop.py").write_text(_LOOP)

    def entrypoint(
        self, config: ResearcherConfig, endpoint: Endpoint, budget_hours: float
    ) -> Entrypoint:
        # The container's main process stays alive independently; the
        # example loop is driven by ceb via RunHandle.exec, not by stdin.
        argv = ["sleep", "infinity"]
        return Entrypoint(argv=argv, env={"HOME": "/home/researcher"})


__all__ = ["ExampleCLI"]
