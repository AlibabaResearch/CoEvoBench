"""End-to-end smoke: run envelope + broker + sandbox + toy bench judge.

This is the CI loop promised in Section 4.5 of the design document: no
GPUs, no real benchmark data, no researcher credentials. Requires docker.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from coevobench.bench.toy.adapter import ToyBench
from coevobench.core.lifecycle.registry import RunManifest, RunRegistry
from coevobench.orchestrate.supervisor import LifecycleStore, Supervisor
from coevobench.research.server import Broker, BrokerServer
from coevobench.runtime.broker_wire import call
from coevobench.runtime.sandbox.base import SandboxSpec
from coevobench.runtime.sandbox.broker_client import BrokerSandboxRuntime


def _docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "version", "--format", "x"],
                capture_output=True,
                timeout=15,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="docker daemon not available"
)


def test_broker_roundtrip():
    """sandbox ops through the broker socket: create -> write -> run -> kill."""
    sock_dir = Path(tempfile.mkdtemp(prefix="ceb-t-"))
    broker = Broker()
    server = BrokerServer(str(sock_dir / "broker.sock"), broker)
    server.serve_forever_bg()
    try:
        response = call(
            str(sock_dir / "broker.sock"),
            {
                "op": "sandbox.create",
                "image": "python:3.11-slim",
                "cpu": 1,
                "mem_gib": 1,
                "ttl_s": 120,
            },
            timeout=180,
        )
        sandbox_id = response["sandbox_id"]
        runtime = BrokerSandboxRuntime(str(sock_dir / "broker.sock"))
        sb = runtime.connect(sandbox_id)
        result = sb.run("echo through-broker")
        assert result["exit_code"] == 0
        assert "through-broker" in result["text"]
        sb.kill()
    finally:
        broker.cleanup()
        server.server_close()


def test_toy_bench_full_loop():
    """The smoke loop: task view -> rollout in a broker sandbox -> judge verdict."""
    sock_dir = Path(tempfile.mkdtemp(prefix="ceb-t-"))
    broker = Broker()
    server = BrokerServer(str(sock_dir / "broker.sock"), broker)
    server.serve_forever_bg()
    try:
        runtime = BrokerSandboxRuntime(str(sock_dir / "broker.sock"))
        bench = ToyBench()
        view = bench.public_view("toy-sum-1")
        spec = SandboxSpec(image=view.image, cpu=1, mem_gib=1, ttl_s=120)
        sb = runtime.create(spec)
        try:
            sb.run(bench.rollout_script("toy-sum-1"))
            verdict = bench.judge("toy-sum-1", "", sb)
            assert verdict.resolved
            assert verdict.score == 1.0
        finally:
            sb.kill()
    finally:
        broker.cleanup()
        server.server_close()


def test_registry_and_lifecycle(tmp_path):
    registry = RunRegistry(tmp_path)
    manifest = RunManifest(
        run_id="glm53_tb_1",
        bench="tb",
        researcher_model="glm53",
        researcher_cli="claude-code",
        replicate=1,
        budget_hours=24.0,
        gpus=(0, 1, 2, 3),
        image="python:3.11-slim",
    )
    registry.register(manifest)

    assert registry.list_runs() == ["glm53_tb_1"]
    assert registry.next_replicate("glm53", "tb") == 2

    store = LifecycleStore(registry.root)
    stopped = []
    supervisor = Supervisor("glm53_tb_1", store, stop_fn=lambda: stopped.append(1))
    supervisor.start(24.0)
    assert supervisor.state.budget_left_hours() <= 24.0

    # deadline in the past: reconcile must shut down and burn to zero
    supervisor.state.deadline_at -= 48 * 3600
    supervisor.store.save(supervisor.state)
    phase = supervisor.reconcile(container_running=True)
    assert phase == "done"
    assert stopped == [1]
    assert supervisor.state.budget_left_hours() == 0.0


def test_broker_sandbox_from_dir_roundtrip(tmp_path):
    """sandbox.create --from-dir: the broker builds the context and runs
    the sandbox on it in place — no registry, no job lifecycle."""
    import subprocess as sp

    if not _docker_available():
        pytest.skip("docker daemon not available")

    ws = tmp_path / "ws"
    ctx = ws / "my-task"
    ctx.mkdir(parents=True)
    (ctx / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "COPY payload.txt /payload.txt\n"
        "RUN cat /payload.txt > /buildtime-proof.txt\n"
    )
    (ctx / "payload.txt").write_text("BUILT-AT-BUILD-TIME")

    broker = Broker(workspace_root=ws)
    sock_dir = Path(sp.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip())
    server = BrokerServer(str(sock_dir / "b.sock"), broker)
    server.serve_forever_bg()
    try:
        created = call(
            str(sock_dir / "b.sock"),
            {"op": "sandbox.create", "from_dir": str(ctx), "cpu": 1, "mem_gib": 1},
            timeout=600,
        )
        sid = created["sandbox_id"]
        probe = call(
            str(sock_dir / "b.sock"),
            {"op": "sandbox.exec", "sandbox_id": sid, "command": "cat /buildtime-proof.txt"},
            timeout=120,
        )
        assert "BUILT-AT-BUILD-TIME" in probe["text"]
    finally:
        if "sid" in locals():
            call(str(sock_dir / "b.sock"), {"op": "sandbox.kill", "sandbox_id": sid}, timeout=60)
        broker.cleanup()
        server.server_close()
