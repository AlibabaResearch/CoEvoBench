"""Codex CLI researcher adapter (binary distribution, no cargo needed).

Provision verifies the codex binary and writes the config (model,
approval policy, sandbox mode, wire API). The entrypoint runs one exec
session with JSON events on stdout for operator log capture. The AppWorld
cloud entry wraps it in a budget-aware continuation loop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from coevobench.runtime.researcher.base import (
    Endpoint,
    ProvisionContext,
    ResearcherConfig,
)
from coevobench.runtime.runhost.base import Entrypoint


class CodexCLI:
    name = "codex"

    def provision(self, config: ResearcherConfig, ctx: ProvisionContext) -> None:
        if shutil.which("codex") is None:
            raise RuntimeError("codex binary not found; install from the official release channel")
        home = Path(ctx.home)
        (home / ".codex").mkdir(parents=True, exist_ok=True)
        wire = config.cli_endpoint or "https://api.openai.com/v1"
        (home / ".codex" / "config.toml").write_text(
            "\n".join(
                [
                    f"model = {json.dumps(config.model)}",
                    'model_provider = "research"',
                    f"model_reasoning_effort = {json.dumps(config.reasoning_effort or 'high')}",
                    'approval_policy = "never"',
                    'sandbox_mode = "danger-full-access"',
                    "[model_providers.research]",
                    'name = "Research"',
                    f"base_url = {json.dumps(wire)}",
                    'wire_api = "responses"',
                    'env_key = "RESEARCHER_TOKEN"',
                ]
            )
            + "\n"
        )

    def entrypoint(
        self, config: ResearcherConfig, endpoint: Endpoint, budget_hours: float
    ) -> Entrypoint:
        argv = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--model",
            config.model,
        ]
        # Pin the researcher settings on every invocation, including resumed sessions.
        overrides = {
            "model_provider": "research",
            "model_reasoning_effort": config.reasoning_effort or "high",
            "approval_policy": "never",
            "sandbox_mode": "danger-full-access",
            "model_providers.research.name": "Research",
            "model_providers.research.base_url": endpoint.base_url.rstrip("/"),
            "model_providers.research.wire_api": "responses",
            "model_providers.research.env_key": "RESEARCHER_TOKEN",
        }
        for key, value in overrides.items():
            argv.extend(["--config", f"{key}={json.dumps(value)}"])
        argv.append(
            "Read TASK.md and RESEARCH_TOOLS.md and continue the research autonomously. "
            "Work under the persistent workspace. Check ./timer.sh status. "
            "Keep the nomination current; before finishing write out/DONE.md and run ./submit."
        )
        env = {
            "RESEARCHER_TOKEN": f"${endpoint.token_env}",
            "CEB_RESEARCHER_CLI": "codex",
            "HOME": "/home/researcher",
        }
        return Entrypoint(argv=argv, env=env)


__all__ = ["CodexCLI"]
