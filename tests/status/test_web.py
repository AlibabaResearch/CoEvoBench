"""Web console: same-schema invariant and redaction."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from coevobench.core.lifecycle.registry import RunManifest, RunRegistry
from coevobench.orchestrate.supervisor import LifecycleStore, Supervisor
from coevobench.status.collect import collect_run, redact
from coevobench.status.web import serve_bg


@pytest.fixture
def registry(tmp_path: Path) -> RunRegistry:
    reg = RunRegistry(tmp_path)
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
    reg.register(manifest)
    store = LifecycleStore(reg.root)
    supervisor = Supervisor("glm53_tb_1", store, stop_fn=lambda: None)
    supervisor.start(24.0)
    return reg


def test_api_and_collector_same_document(registry: RunRegistry) -> None:
    port = _free_port()
    server = serve_bg(port=port, registry_root=registry.root.parent)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/glm53_tb_1") as resp:
            api_doc = json.loads(resp.read())
    finally:
        server.server_close()
    local_doc = redact(collect_run(registry, "glm53_tb_1"))
    # timestamp and budget_left_hours are computed at collection time —
    # two reads milliseconds apart can straddle a rounding boundary.
    api_doc.pop("timestamp")
    local_doc.pop("timestamp")
    api_budget = api_doc.pop("budget_left_hours")
    local_budget = local_doc.pop("budget_left_hours")
    assert api_budget == pytest.approx(local_budget, abs=0.01)
    # everything else must be identical
    assert api_doc == local_doc
    assert api_doc["run_id"] == "glm53_tb_1"
    assert api_doc["phase"] == "running"
    assert 0 < local_budget <= 24.0


def test_health_and_runs_endpoints(registry: RunRegistry) -> None:
    port = _free_port()
    server = serve_bg(port=port, registry_root=registry.root.parent)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as resp:
            health = json.loads(resp.read())
        assert health["ok"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs") as resp:
            runs = json.loads(resp.read())["runs"]
        assert [r["run_id"] for r in runs] == ["glm53_tb_1"]
    finally:
        server.server_close()


def test_unknown_run_404(registry: RunRegistry) -> None:
    import urllib.error

    port = _free_port()
    server = serve_bg(port=port, registry_root=registry.root.parent)
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/nope")
        assert excinfo.value.code == 404
    finally:
        server.server_close()


def test_redact_strips_credential_shapes() -> None:
    payload = {
        "run_id": "r",
        "api_token": "sk-secret",
        "nested": {"registry_key": "hunter2", "phase": "running"},
    }
    cleaned = redact(payload)
    assert cleaned["api_token"] == "[REDACTED]"
    assert cleaned["nested"]["registry_key"] == "[REDACTED]"
    assert cleaned["nested"]["phase"] == "running"


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
