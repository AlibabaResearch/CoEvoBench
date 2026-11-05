"""Researcher CLI adapters: provision writes, entrypoint shapes."""

from __future__ import annotations

import json
import shutil

import pytest

from coevobench.runtime.researcher.base import (
    Endpoint,
    ProvisionContext,
    ResearcherConfig,
)
from coevobench.runtime.researcher.codex import CodexCLI
from coevobench.runtime.researcher.example import ExampleCLI
from coevobench.runtime.researcher.opencode import OpenCodeCLI

CTX = ProvisionContext(workspace="/workspace", home="/home/researcher")
CFG = ResearcherConfig(model="gpt-6", cli="codex")
ENDPOINT = Endpoint(base_url="https://gw.example/v1")


def _with_binary(monkeypatch, name: str) -> None:
    monkeypatch.setattr(shutil, "which", lambda n: f"/usr/bin/{n}" if n == name else None)


def test_example_provision_writes_loop(tmp_path) -> None:
    cli = ExampleCLI()
    cli.provision(CFG, ProvisionContext(workspace=str(tmp_path), home=str(tmp_path / "h")))
    assert (tmp_path / "example_loop.py").is_file()
    entry = cli.entrypoint(CFG, ENDPOINT, 24.0)
    assert entry.argv == ["sleep", "infinity"]


def test_codex_provision_writes_config(tmp_path, monkeypatch) -> None:
    _with_binary(monkeypatch, "codex")
    cli = CodexCLI()
    home = tmp_path / "home"
    cli.provision(CFG, ProvisionContext(workspace=str(tmp_path), home=str(home)))
    config = (home / ".codex" / "config.toml").read_text()
    assert 'model = "gpt-6"' in config
    assert 'wire_api = "responses"' in config


def test_codex_provision_requires_binary(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="codex binary not found"):
        CodexCLI().provision(CFG, CTX)


def test_claude_entrypoint_uses_persistent_loop() -> None:
    from coevobench.runtime.researcher.claude_code import ClaudeCodeCLI

    entry = ClaudeCodeCLI().entrypoint(CFG, ENDPOINT, 24.0)
    assert entry.argv == ["bash", "/opt/researcher-loop.sh"]
    assert entry.env["RESEARCHER_MODEL"] == "gpt-6"
    assert entry.env["ANTHROPIC_AUTH_TOKEN"] == "$RESEARCHER_TOKEN"


def test_codex_entrypoint_uses_scoped_token(tmp_path) -> None:
    entry = CodexCLI().entrypoint(CFG, ENDPOINT, 24.0)
    assert entry.argv[0] == "codex" and "exec" in entry.argv
    assert entry.env["RESEARCHER_TOKEN"] == "$RESEARCHER_TOKEN"


def test_opencode_provision_writes_plugin_and_config(tmp_path, monkeypatch) -> None:
    _with_binary(monkeypatch, "opencode")
    cli = OpenCodeCLI()
    ctx = ProvisionContext(workspace=str(tmp_path), home=str(tmp_path / "h"))
    cli.provision(CFG, ctx)
    plugin = (tmp_path / ".opencode/plugins/ceb-audit/index.ts").read_text()
    assert 'ctx.tool.hook("execute.before"' in plugin
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["model"] == "research/gpt-6"
    assert config["providers"]["research"]["settings"]["baseURL"] == "{env:RESEARCHER_BASE_URL}"
    assert config["providers"]["research"]["settings"]["apiKey"] == "{env:RESEARCHER_TOKEN}"


def test_opencode_entrypoint_carries_endpoint(tmp_path) -> None:
    entry = OpenCodeCLI().entrypoint(CFG, ENDPOINT, 24.0)
    assert entry.env["RESEARCHER_BASE_URL"] == "https://gw.example/v1"
    assert entry.argv[0] == "opencode"


def test_gateway_base_strips_trailing_v1() -> None:
    from coevobench.runtime.researcher.claude_code import _gateway_base

    assert _gateway_base("https://gw.example/apps/anthropic/v1") == "https://gw.example/apps/anthropic"
    assert _gateway_base("https://gw.example/apps/anthropic/v1/") == "https://gw.example/apps/anthropic"
    assert _gateway_base("https://gw.example/apps/anthropic") == "https://gw.example/apps/anthropic"
    assert _gateway_base("https://gw.example/v1/v1") == "https://gw.example"
    assert _gateway_base("https://api.anthropic.com") == "https://api.anthropic.com"


def test_claude_entrypoint_normalizes_gateway_base() -> None:
    from coevobench.runtime.researcher.claude_code import ClaudeCodeCLI

    cli = ClaudeCodeCLI()
    entry = cli.entrypoint(
        ResearcherConfig(model="glm-5.3", cli="claude-code"),
        Endpoint(base_url="https://gw.example/apps/anthropic/v1"),
        1.0,
    )
    assert entry.env["ANTHROPIC_BASE_URL"] == "https://gw.example/apps/anthropic"


def test_bootstrap_provisions_in_the_child_environment(tmp_path):
    import os
    import subprocess
    import sys

    home = tmp_path / "home"
    home.mkdir()
    command = (
        "import json,os;from pathlib import Path;"
        "p=Path(os.environ['HOME'])/'.claude/settings.json';"
        "assert json.loads(p.read_text())['hooks']['PreToolUse'];print('provisioned')"
    )
    result = subprocess.run(
        [sys.executable, "-m", "coevobench.runtime.researcher.bootstrap", "claude-code",
         "fixture", "https://gateway.example", sys.executable, "-c", command],
        cwd=tmp_path, env={**os.environ, "HOME": str(home)}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "provisioned"
