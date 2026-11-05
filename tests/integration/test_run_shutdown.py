"""Real Docker recovery: daemon death, TTL and deadline cleanup stay run-scoped."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from coevobench.cli.run import launch_run, stop_run
from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.orchestrate.supervisor import LifecycleStore
from coevobench.runtime.broker_wire import call
from coevobench.runtime.sandbox.base import SandboxSpec
from coevobench.runtime.sandbox.docker import DockerSandboxRuntime


@pytest.fixture
def docker():
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker daemon required")
    if result.returncode:
        pytest.skip("Docker daemon required")
    return DockerSandboxRuntime()


def exists(identity):
    return subprocess.run(["docker", "inspect", identity], capture_output=True).returncode == 0


@pytest.fixture
def same_host_docker(docker, tmp_path):
    marker = tmp_path / "operator-marker"
    marker.write_text("shared-operator-filesystem")
    probe = subprocess.run(
        ["docker", "run", "--rm", "--mount", f"type=bind,src={marker},dst=/marker,readonly",
         "python:3.11-slim", "cat", "/marker"], capture_output=True, text=True, timeout=30,
    )
    if probe.returncode or probe.stdout != "shared-operator-filesystem":
        pytest.skip("Docker daemon does not share the operator filesystem")
    return docker


def wait_for(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("condition not reached before deadline")


@pytest.mark.sandbox
@pytest.mark.parametrize("ending", ["deadline", "researcher-exit", "broker-killed",
                                    "receipt-lost", "submitted", "ttl"])
def test_run_shutdown_captures_submission_and_reaps_only_owned(same_host_docker, tmp_path, ending):
    docker = same_host_docker
    foreign = docker.create(SandboxSpec(image="python:3.11-slim", cpu=1, mem_gib=1, ttl_s=180))
    # Short root also exercises broker socket creation on macOS.
    with tempfile.TemporaryDirectory(prefix="ceb-stop-") as td:
        root = Path(td)
        weights = root / "base"
        weights.mkdir()
        (weights / "config.json").write_text("{}")
        info = None
        try:
            info = launch_run(bench="toy", researcher_model="example", researcher_cli="example",
                              budget_hours=0.1, gpus=(), registry_root=root,
                              image="python:3.11-slim", base_dir=str(weights))
            owner = Path(info["workspace"]).parent
            sock = str(owner / "broker.sock")
            response = call(sock, {"op": "sandbox.create", "image": "python:3.11-slim",
                                   "cpu": 1, "mem_gib": 1,
                                   "ttl_s": 3 if ending == "ttl" else 120})
            owned = response["sandbox_id"]
            assert exists(owned)
            receipts = [json.loads(p.read_text())
                        for p in (owner / "eval-resources").glob("*.json")]
            assert any(r["identity"] == owned for r in receipts)
            read_base = subprocess.run(["docker", "exec", info["container"], "cat",
                                        "/workspace/base/config.json"], capture_output=True)
            assert read_base.returncode == 0
            store = LifecycleStore(RunRegistry(root).root)
            state = store.load(info["run_id"])
            if ending == "deadline":
                state.deadline_at = time.time() + 1
                store.save(state)
                public = subprocess.check_output(["docker", "exec", info["container"], "cat",
                                                   "/run/ceb/lifecycle.json"], text=True)
                assert json.loads(public)["deadline_at"] == state.deadline_at
            elif ending in ("researcher-exit", "submitted"):
                if ending == "submitted":
                    from coevobench.core.grading.submission import request

                    workspace = Path(info["workspace"])
                    final = workspace / "out/toy_final"
                    (final / "harness").mkdir(parents=True)
                    (final / "model_path.txt").write_text("/base")
                    for name in ("meta.json", "serve_config.json"):
                        (final / name).write_text("{}")
                    (workspace / "out/DONE.md").write_text("Test researcher submitted")
                    request(workspace, "toy")
                subprocess.run(["docker", "stop", "--timeout", "1", info["container"]],
                               check=True, capture_output=True)
            elif ending == "ttl":
                wait_for(lambda: not exists(owned))
                assert call(sock, {"op": "ping"})["ok"]
                assert json.loads((owner / "manifest.json").read_text())["phase"] == "running"
                assert stop_run(info["run_id"], registry_root=root)["stopped"]
            else:
                os.kill(int((owner / "broker.pid").read_text()), signal.SIGKILL)
                if ending == "receipt-lost":
                    for receipt in (owner / "eval-resources").glob("*.json"):
                        if json.loads(receipt.read_text())["identity"] == owned:
                            receipt.unlink()
                assert stop_run(info["run_id"], registry_root=root)["stopped"]
            wait_for(lambda: not exists(info["container"]) and not exists(owned))
            assert exists(foreign.id)
            report = json.loads((owner / "audit/submission-status.json").read_text())
            assert report["confirmed"] == (ending == "submitted")
            if ending != "submitted":
                assert report["status"] == "incomplete"
                assert not (owner / "workspace/out/toy_final/submission.json").exists()
            assert (owner / "audit/researcher/container.log").is_file()
            assert stop_run(info["run_id"], registry_root=root)["stopped"]
        finally:
            if info is not None:
                stop_run(info["run_id"], registry_root=root)
            foreign.kill()


@pytest.mark.sandbox
def test_docker_ttl_is_enforced_without_broker(docker):
    sandbox = docker.create(SandboxSpec(image="python:3.11-slim", cpu=1, mem_gib=1, ttl_s=2))
    try:
        wait_for(lambda: not exists(sandbox.id), seconds=8)
    finally:
        sandbox.kill()


def test_launch_failure_after_broker_start_closes_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("CEB_SERVE__DEFAULTS__TENSOR_PARALLEL", "2")
    monkeypatch.setenv("CEB_SERVE__DEFAULTS__MAX_MODEL_LEN", "8192")
    monkeypatch.setenv("CEB_SERVE__DEFAULTS__ENFORCE_EAGER", "true")
    def broken_adapter(name):
        raise RuntimeError("researcher adapter failed")

    monkeypatch.setattr("coevobench.cli.run._researcher_cli", broken_adapter)
    with pytest.raises(RuntimeError, match="adapter failed"):
        launch_run(bench="toy", researcher_model="example", researcher_cli="example",
                   budget_hours=1, gpus=(), registry_root=tmp_path)
    owner = tmp_path / "runs/example_toy_1"
    assert not (owner / "broker.pid").exists()
    assert json.loads((owner / "manifest.json").read_text())["phase"] == "done"
    assert (owner / "eval-resources/closed").is_file()
    recipe = json.loads((owner / "workspace/serve_config.json").read_text())
    assert recipe["tensor_parallel"] == 2 and recipe["max_len"] == 8192
    assert recipe["enforce_eager"] is True
    assert "max_model_len" not in recipe


def test_broker_survives_cleanup_exception_and_retries(tmp_path):
    from coevobench.orchestrate.broker_daemon import start_broker_daemon, stop_broker_daemon
    from coevobench.runtime.run_resources import process_start, register

    (tmp_path / "workspace").mkdir()
    (tmp_path / "manifest.json").write_text('{"bench":"toy"}')
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              start_new_session=True)
    try:
        register(tmp_path, "process", worker.pid, ttl_s=-1)
        receipt = next((tmp_path / "eval-resources").glob("*.json"))
        original = receipt.read_text()
        receipt.write_text("{broken receipt")
        info = start_broker_daemon(tmp_path)
        socket = str(Path(info["socket"]).resolve())
        identity = process_start(info["pid"])
        history = tmp_path / "eval-resources/cleanup-events.jsonl"
        wait_for(lambda: history.exists() and "JSONDecodeError" in history.read_text())
        assert call(socket, {"op": "ping"})["ok"]
        assert process_start(info["pid"]) == identity
        assert not (tmp_path / "eval-resources/closed").exists()
        repaired = receipt.with_suffix(".repaired")
        repaired.write_text(original)
        repaired.replace(receipt)
        assert worker.wait(timeout=15) < 0
        wait_for(lambda: "released_at" in receipt.read_text())
        assert call(socket, {"op": "ping"})["ok"]
        assert process_start(info["pid"]) == identity
        assert "JSONDecodeError" in history.read_text()
    finally:
        stop_broker_daemon(tmp_path)
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=5)
