"""Provision the selected CLI inside its own environment, then replace this process."""

from __future__ import annotations

import importlib
import os
import sys

from coevobench.runtime.researcher.base import ProvisionContext, ResearcherConfig

ADAPTERS = {
    "claude-code": ("claude_code", "ClaudeCodeCLI"),
    "codex": ("codex", "CodexCLI"),
    "opencode": ("opencode", "OpenCodeCLI"),
}


def main() -> None:
    name, model, endpoint, *command = sys.argv[1:]
    module, symbol = ADAPTERS[name]
    adapter = getattr(importlib.import_module("coevobench.runtime.researcher." + module), symbol)()
    adapter.provision(
        ResearcherConfig(model=model, cli=name, cli_endpoint=endpoint),
        ProvisionContext(workspace=os.environ.get("CEB_WORKSPACE", os.getcwd()),
                         home=os.environ["HOME"]),
    )
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
