"""Exercise Daytona adaptation through a local Docker-backed SDK double, never paid APIs."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from coevobench.runtime.sandbox.base import FileWrite, SandboxSpec
from coevobench.runtime.sandbox.daytona import DaytonaSandboxRuntime
from coevobench.runtime.sandbox.docker import DockerSandboxRuntime


class FakeFs:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def upload_file(self, data, path):
        self.sandbox.write_files([FileWrite(path, data)])

    def set_file_permissions(self, path, *, mode):
        self.sandbox.run(f"chmod {mode} {path}")

    def download_file(self, path):
        return self.sandbox.read_file(path).encode()


class FakeApi:
    def create(self, params, *, timeout):
        self.params = params
        sandbox = DockerSandboxRuntime().create(
            SandboxSpec(image="python:3.11-slim", cpu=1, mem_gib=1)
        )

        def execute(command, *, timeout):
            result = sandbox.run(command, timeout_s=timeout)
            return SimpleNamespace(exit_code=result["exit_code"], result=result["text"])

        return SimpleNamespace(
            id=sandbox.id, process=SimpleNamespace(exec=execute), fs=FakeFs(sandbox), inner=sandbox
        )

    def delete(self, sandbox, **_kwargs):
        sandbox.inner.kill()


@pytest.fixture
def sandbox():
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode:
            pytest.skip("local Docker required for isolated SDK-double tests")
    except OSError:
        pytest.skip("local Docker unavailable")
    api = FakeApi()
    runtime = DaytonaSandboxRuntime(api=api)
    sb = runtime.create(SandboxSpec(image="python:3.11-slim", ttl_s=125, network=False))
    assert api.params.auto_stop_interval == 0
    assert api.params.ttl_minutes == 3
    assert api.params.network_block_all is True
    try:
        yield sb
    finally:
        sb.kill()


@pytest.mark.parametrize(
    "name",
    [
        "test_run_returns_exec_result",
        "test_shell_state_does_not_persist",
        "test_files_persist_across_runs",
        "test_timeout_kills_process_tree",
        "test_write_file_mode_is_octal",
        "test_kill_is_idempotent",
    ],
)
def test_existing_sandbox_contract_unchanged(sandbox, name):
    path = Path(__file__).with_name("test_sandbox_contract.py")
    spec = importlib.util.spec_from_file_location("sandbox_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    getattr(module, name)(sandbox)


def test_build_context_uses_public_sdk_without_cloud(tmp_path, monkeypatch):
    import json

    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\nCOPY data /task/data\n")
    (tmp_path / "data").write_text("fixture")
    calls = []
    raw = SimpleNamespace(id="daytona-fixture", process=SimpleNamespace(
        exec=lambda *a, **k: SimpleNamespace(exit_code=0, result=json.dumps(
            {"exit_code": 0, "text": "Python 3.11"}
        ))
    ))
    class Api:
        def create(self, params, **kwargs):
            calls.append(params)
            return raw
    runtime = DaytonaSandboxRuntime(api=Api(), owner_dir=tmp_path)
    box = runtime.create_from_context(tmp_path, SandboxSpec(cpu=3, mem_gib=5, ttl_s=120))
    assert box.id == raw.id
    assert calls[0].resources.cpu == 3 and calls[0].network_block_all
    assert calls[0].image._context_list[0].source_path == str(tmp_path / "data")
    record = json.loads(next((tmp_path / "eval-resources").glob("*.json")).read_text())
    assert record["identity"] == raw.id and record.get("expires_at")


@pytest.mark.parametrize(
    "researcher_cli,image",
    [
        ("claude-code", "coevobench-run:latest"),
        ("codex", "coevobench-run:latest"),
    ],
)
def test_researcher_launch_and_binary_workspace_roundtrip(
    tmp_path, monkeypatch, researcher_cli, image
):
    import json
    import time

    from coevobench.runtime.gpu.profile import GPURequest
    from coevobench.runtime.runhost import daytona_host
    from coevobench.runtime.runhost.base import Entrypoint, RunEnvelope

    try:
        present = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
        if present.returncode:
            pytest.skip("build Dockerfile.run to test researcher workspace transfer")
    except OSError:
        pytest.skip("Docker unavailable")

    class BinaryFs(FakeFs):
        def download_file(self, path):
            return subprocess.check_output(["docker", "exec", self.sandbox.id, "cat", path])

    class ResearcherApi(FakeApi):
        def create(self, params, *, timeout):
            sb = DockerSandboxRuntime().create(SandboxSpec(image=image, cpu=2, mem_gib=4))
            self.params = params

            def execute(command, *, timeout):
                result = sb.run(command, timeout_s=timeout)
                return SimpleNamespace(exit_code=result["exit_code"], result=result["text"])

            return SimpleNamespace(
                id=sb.id, process=SimpleNamespace(exec=execute), fs=BinaryFs(sb), inner=sb
            )

    api = ResearcherApi()
    monkeypatch.setattr(daytona_host, "client", lambda: api)
    workspace = tmp_path / "workspace"
    (workspace / "harness").mkdir(parents=True)
    (workspace / "harness/agent.py").write_text("print(1)")
    (workspace / "eval").write_text("operator entry")
    (tmp_path / "lifecycle.json").write_text(json.dumps({"deadline_at": time.time() + 120}))
    handle = daytona_host.DaytonaRunHost().launch(
        RunEnvelope(image="fixed-image", gpus=GPURequest(device_ids=()), budget_hours=1),
        Entrypoint(
            argv=["sh", "-c", "id -un > who; pwd > where; sleep 120"],
            env={"CEB_RESEARCHER_CLI": researcher_cli},
        ),
        str(workspace),
    )
    try:
        for _ in range(20):
            if handle.exec("test -f /workspace/who")["exit_code"] == 0:
                break
            time.sleep(0.1)
        handle.pull(workspace)
        assert (workspace / "who").read_text().strip() == "researcher"
        assert (workspace / "where").read_text().strip() == "/workspace"
        assert api.params.os_user == "root"
        assert api.params.auto_stop_interval == 0
        (workspace / "harness/agent.py").write_text("new code")
        handle.push(workspace)
        assert handle.exec("cat /workspace/harness/agent.py")["text"] == "new code"
        handle.exec("rm /workspace/harness/agent.py; echo tampered >/workspace/eval")
        handle.pull(workspace)
        assert not (workspace / "harness/agent.py").exists()
        assert (workspace / "eval").read_text() == "operator entry"
    finally:
        handle.stop()
