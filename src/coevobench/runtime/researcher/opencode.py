"""OpenCode researcher adapter (single binary, fully open chain).

Provision verifies the binary and writes the project config (model,
provider endpoint from env). The entrypoint runs the agent in
non-interactive mode; the audit plugin streams tool calls as JSON on
stderr (same contract as the other adapters).
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

_PLUGIN = """import { Plugin } from "@opencode/plugin"
import { appendFileSync, mkdirSync } from "node:fs"
import { join } from "node:path"
export default Plugin.define({
  id: "ceb-audit",
  async setup(ctx) {
    const out = join(ctx.location.directory, "out")
    mkdirSync(out, { recursive: true })
    await ctx.tool.hook("execute.before", (event) => {
      const line = JSON.stringify({time: Date.now() / 1000,
        tool: event.tool, command: event.input?.command ?? ""})
      appendFileSync(join(out, "opencode-tools.jsonl"), line + "\\n")
      console.error(line)
    });
  },
});
"""


class OpenCodeCLI:
    name = "opencode"

    def provision(self, config: ResearcherConfig, ctx: ProvisionContext) -> None:
        if shutil.which("opencode") is None:
            raise RuntimeError(
                "opencode binary not found; install from https://opencode.ai"
            )
        workspace = Path(ctx.workspace)
        plugin = workspace / ".opencode/plugins/ceb-audit"
        plugin.mkdir(parents=True, exist_ok=True)
        (plugin / "index.ts").write_text(_PLUGIN)
        (workspace / "opencode.json").write_text(
            json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "model": f"research/{config.model}",
                    "providers": {
                        "research": {
                            "package": "@opencode/ai/providers/openai-compatible",
                            "env": ["RESEARCHER_TOKEN"],
                            "settings": {
                                "baseURL": "{env:RESEARCHER_BASE_URL}",
                                "apiKey": "{env:RESEARCHER_TOKEN}",
                            },
                            "models": {config.model: {"capabilities": {"tools": True}}},
                        }
                    },
                },
                indent=2,
            )
            + "\n"
        )

    def entrypoint(
        self, config: ResearcherConfig, endpoint: Endpoint, budget_hours: float
    ) -> Entrypoint:
        argv = ["opencode", "--standalone", "run", "--auto", "--format", "json",
                "--model", f"research/{config.model}",
                "Read TASK.md and RESEARCH_TOOLS.md and work through them autonomously."]
        env = {
            "RESEARCHER_BASE_URL": endpoint.base_url,
            "RESEARCHER_TOKEN": f"${endpoint.token_env}",
            "HOME": "/home/researcher",
        }
        return Entrypoint(argv=argv, env=env)


__all__ = ["OpenCodeCLI"]
