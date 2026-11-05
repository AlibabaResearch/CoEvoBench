"""Claude Code researcher CLI adapter.

Provisioning writes the settings file with permission denies and the audit
hook; the entrypoint runs the CLI in print mode against the configured
gateway. The audit hook (Section 5.6) streams every Bash tool call to
stderr, which docker captures host-side: append-only from the run's
perspective.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from coevobench.runtime.researcher.base import (
    Endpoint,
    ProvisionContext,
    ResearcherConfig,
)
from coevobench.runtime.runhost.base import Entrypoint

_SETTINGS = {
    "permissions": {
        "deny": [
            "Write(/base/**)",
            "Edit(/base/**)",
        ]
    },
}

_HOOK = """#!/usr/bin/env python3
import json
import sys
import time

for line in sys.stdin:
    try:
        event = json.loads(line)
    except ValueError:
        continue
    tool = event.get("tool_name", "?")
    command = ""
    if tool == "Bash":
        command = event.get("tool_input", {}).get("command", "")
    print(json.dumps({"time": time.time(), "tool": tool, "command": command}),
          file=sys.stderr, flush=True)
"""


def _gateway_base(url: str) -> str:
    """Claude Code appends /v1/messages itself; a base ending in /v1
    would produce /v1/v1/messages and the gateway rejects it as an
    unknown model. Strip the trailing /v1."""
    while url.rstrip("/").endswith("/v1"):
        url = url.rstrip("/")[: -len("/v1")]
    return url.rstrip("/")


class ClaudeCodeCLI:
    name = "claude-code"

    def provision(self, config: ResearcherConfig, ctx: ProvisionContext) -> None:
        home = Path(ctx.home)
        (home / ".claude").mkdir(parents=True, exist_ok=True)
        hook_path = home / ".claude" / "audit_hook.py"
        hook_path.write_text(_HOOK)
        hook_path.chmod(0o755)
        settings = {**_SETTINGS, "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [
            {"type": "command", "command": f"python3 {shlex.quote(str(hook_path))}"},
        ]}]}}
        (home / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2))

    def entrypoint(
        self, config: ResearcherConfig, endpoint: Endpoint, budget_hours: float
    ) -> Entrypoint:
        argv = [
            "bash",
            "/opt/researcher-loop.sh",
        ]
        env = {
            "RESEARCHER_MODEL": config.model,
            "ANTHROPIC_BASE_URL": _gateway_base(endpoint.base_url),
            "ANTHROPIC_AUTH_TOKEN": f"${endpoint.token_env}",
            # Non-Anthropic models (glm/qwen/...) are unknown to Claude
            # Code's catalog; this restores pass-through to the gateway.
            "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
            "HOME": "/home/researcher",
        }
        return Entrypoint(argv=argv, env=env)


__all__ = ["ClaudeCodeCLI", "_gateway_base"]
