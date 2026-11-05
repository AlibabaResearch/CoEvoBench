"""Sandbox SPI contract tests, run against every registered implementation.

Any new SandboxRuntime must pass this suite unchanged. Requires a working
docker daemon; skipped otherwise.
"""

from __future__ import annotations

import pytest

from coevobench.runtime.sandbox.base import FileWrite, SandboxSpec
from coevobench.runtime.sandbox.docker import DockerSandboxRuntime

pytestmark = pytest.mark.sandbox

IMAGE = "python:3.11-slim"


@pytest.fixture(scope="module")
def runtime() -> DockerSandboxRuntime:
    return DockerSandboxRuntime()


@pytest.fixture
def sandbox(runtime: DockerSandboxRuntime):
    sb = runtime.create(SandboxSpec(image=IMAGE, cpu=1, mem_gib=1, ttl_s=300))
    yield sb
    sb.kill()


def _docker_available() -> bool:
    import subprocess

    try:
        return subprocess.run(
            ["docker", "version", "--format", "x"], capture_output=True, timeout=15
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="docker daemon not available"
)


def test_run_returns_exec_result(sandbox) -> None:
    result = sandbox.run("echo hello")
    assert result["exit_code"] == 0
    assert "hello" in result["text"]


def test_shell_state_does_not_persist(sandbox) -> None:
    sandbox.run("export TOY_VAR=42")
    result = sandbox.run("echo $TOY_VAR")
    assert result["text"].strip() == ""


def test_files_persist_across_runs(sandbox) -> None:
    sandbox.write_files([FileWrite(path="/tmp/persist.txt", data="marker", mode=0o644)])
    result = sandbox.run("cat /tmp/persist.txt")
    assert result["text"].strip() == "marker"


def test_timeout_kills_process_tree(sandbox) -> None:
    result = sandbox.run(
        "sh -c 'sleep 30 & sleep 30'", timeout_s=3
    )
    assert result["exit_code"] == -1
    # After the sweep, the only survivor with comm=sleep must be the
    # container entrypoint (sleep infinity); the two task sleeps are gone.
    left = sandbox.run(
        "grep -l sleep /proc/[0-9]*/comm 2>/dev/null | wc -l"
    )
    assert left["text"].strip() == "1"


def test_write_file_mode_is_octal(sandbox) -> None:
    sandbox.write_files([FileWrite(path="/tmp/exec.sh", data="#!/bin/sh\necho ok", mode=0o755)])
    result = sandbox.run("/tmp/exec.sh")
    assert result["exit_code"] == 0


def test_kill_is_idempotent(sandbox) -> None:
    sandbox.kill()
    sandbox.kill()


def test_docker_run_host_expands_env_references(monkeypatch) -> None:
    from coevobench.runtime.runhost.docker_host import DockerRunHost

    monkeypatch.setenv("RESEARCHER_TOKEN", "fixture-token")
    monkeypatch.setenv("OTHER", "value2")
    host = DockerRunHost()
    assert host._expand("$RESEARCHER_TOKEN") == "fixture-token"
    assert host._expand("${RESEARCHER_TOKEN}-suffix") == "fixture-token-suffix"
    assert host._expand("plain-$OTHER") == "plain-value2"
    assert host._expand("no-refs-here") == "no-refs-here"
    monkeypatch.delenv("RESEARCHER_TOKEN")
    assert host._expand("$RESEARCHER_TOKEN") == ""
