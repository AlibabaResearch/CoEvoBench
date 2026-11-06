"""Docker CPU ceilings must fit the daemon that runs the container."""

import subprocess

import pytest

from coevobench.runtime.gpu.profile import GPURequest
from coevobench.runtime.runhost import docker_host
from coevobench.runtime.runhost.base import Entrypoint, RunEnvelope
from coevobench.runtime.sandbox import docker
from coevobench.runtime.sandbox.base import SandboxSpec


@pytest.fixture
def daemon(monkeypatch):
    """A four-CPU daemon; no Docker command leaves the test process."""
    state = {"calls": [], "unavailable": False}

    def run(args, **kwargs):
        state["calls"].append(args)
        if args[0] == "info":
            if state["unavailable"]:
                raise docker._CliError("docker info failed: daemon unavailable")
            return subprocess.CompletedProcess(args, 0, "4\n", "")
        return subprocess.CompletedProcess(args, 0, "sandbox-fixture\n", "")

    docker._daemon_cpus.cache_clear()
    monkeypatch.setattr(docker, "_run_cli", run)
    monkeypatch.setattr(docker_host, "_run_cli", run)
    monkeypatch.setattr(docker.DockerSandbox, "run",
                        lambda self, command, **kwargs: {"exit_code": 0, "text": "READY"})
    yield state
    docker._daemon_cpus.cache_clear()


def _cpus(argv):
    return argv[argv.index("--cpus") + 1]


def test_ceiling_is_capped_at_daemon_capacity_and_cached(daemon):
    assert docker.cpu_limit(8) == "4"
    assert docker.cpu_limit(2) == "2"
    assert sum(call[0] == "info" for call in daemon["calls"]) == 1


def test_unknown_capacity_keeps_request_and_is_retried(daemon):
    daemon["unavailable"] = True
    assert docker.cpu_limit(8) == "8"
    daemon["unavailable"] = False
    assert docker.cpu_limit(8) == "4"


def test_run_container_and_sandbox_fit_a_small_host(daemon, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docker_host.DockerRunHost().launch(
        RunEnvelope(image="fixture", gpus=GPURequest(device_ids=())),
        Entrypoint(["true"], {}),
        str(workspace),
    )
    docker.DockerSandboxRuntime().create(SandboxSpec(image="fixture", cpu=2, ttl_s=60))
    run = next(call for call in daemon["calls"] if call[0] == "run")
    create = next(call for call in daemon["calls"] if call[0] == "create")
    assert _cpus(run) == "4"
    assert _cpus(create) == "2"
