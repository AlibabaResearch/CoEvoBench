"""Full broker op surface through the actual socket transport."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from coevobench.research.server import Broker, BrokerServer
from coevobench.runtime.broker_wire import BrokerError, call


def _docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "version", "--format", "x"], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _short_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ceb-ops-"))


def test_ping_reports_capabilities() -> None:
    broker = Broker()
    response = broker.dispatch({"op": "ping"})
    assert response["ok"] is True
    assert "eval.run" in response["ops"]
    assert "contamination.check" in response["ops"]
    assert "sandbox.upload" in response["ops"]
    assert not any(op.startswith("image.") for op in response["ops"])


def test_unknown_op_rejected() -> None:
    with pytest.raises(BrokerError, match="unsupported op"):
        Broker().dispatch({"op": "nope"})


def test_transfer_requires_workspace_root() -> None:
    broker = Broker()
    with pytest.raises(BrokerError, match="workspace root"):
        broker.dispatch({"op": "sandbox.upload", "local": "x", "remote": "/x", "sandbox_id": "s"})


def test_transfer_rejects_paths_outside_workspace() -> None:
    broker = Broker(workspace_root=_short_dir())
    with pytest.raises(BrokerError, match="inside the workspace"):
        broker.dispatch(
            {"op": "sandbox.upload", "local": "../../etc/passwd", "remote": "/x", "sandbox_id": "s"}
        )


def test_remote_path_must_be_absolute() -> None:
    broker = Broker(workspace_root=_short_dir())
    with pytest.raises(BrokerError, match="absolute"):
        broker.dispatch(
            {"op": "sandbox.download", "local": "x", "remote": "relative", "sandbox_id": "s"}
        )


def test_create_needs_image_or_from_dir() -> None:
    broker = Broker()
    with pytest.raises(BrokerError, match="image or from_dir"):
        broker.dispatch({"op": "sandbox.create"})


def test_create_from_dir_stays_inside_the_workspace() -> None:
    broker = Broker(workspace_root=_short_dir())
    with pytest.raises(BrokerError, match="inside the workspace"):
        broker.dispatch({"op": "sandbox.create", "from_dir": "/etc"})


def test_contamination_requires_complete_input_list() -> None:
    broker = Broker(workspace_root=_short_dir(), bench="swe")
    with pytest.raises(BrokerError, match="complete input file list"):
        broker.dispatch({"op": "contamination.check", "inputs": ["a.jsonl"], "extra": 1})


def test_contamination_rejects_unconfigured_bench() -> None:
    broker = Broker(workspace_root=_short_dir(), bench="tau")
    with pytest.raises(BrokerError, match="not configured"):
        broker.dispatch({"op": "contamination.check", "inputs": ["a.jsonl"]})


def test_eval_run_rejects_unknown_bench() -> None:
    broker = Broker()
    try:
        broker.dispatch({"op": "eval.run", "bench": "nope", "split": "visible"})
        raise AssertionError("should have raised")
    except BrokerError as exc:
        assert "not configured" in str(exc)


@pytest.mark.skipif(not _docker_available(), reason="docker daemon not available")
def test_sandbox_lifecycle_ops_over_wire() -> None:
    """create -> exec -> status -> upload -> download -> export -> close,
    the broker surface, driven through the socket."""
    broker = Broker(workspace_root=_short_dir())
    sock_dir = _short_dir()
    server = BrokerServer(str(sock_dir / "b.sock"), broker)
    server.serve_forever_bg()
    wire = str(sock_dir / "b.sock")
    try:
        created = call(
            wire,
            {
                "op": "sandbox.create",
                "image": "python:3.11-slim",
                "cpu": 1,
                "mem_gib": 1,
                "ttl_s": 300,
            },
            timeout=180,
        )
        sid = created["sandbox_id"]

        result = call(
            wire, {"op": "sandbox.exec", "sandbox_id": sid, "command": "echo ops-ok"}
        )
        assert "ops-ok" in result["text"]

        status = call(wire, {"op": "sandbox.status", "sandbox_id": sid})
        assert status["alive"] is True
        assert status["n_events"] >= 1

        upload_src = broker._workspace_root / "up.txt"
        upload_src.write_text("uploaded-bytes")
        up = call(
            wire,
            {
                "op": "sandbox.upload",
                "sandbox_id": sid,
                "local": "up.txt",
                "remote": "/tmp/up.txt",
            },
        )
        assert up["bytes"] == len("uploaded-bytes")
        echoed = call(
            wire, {"op": "sandbox.exec", "sandbox_id": sid, "command": "cat /tmp/up.txt"}
        )
        assert "uploaded-bytes" in echoed["text"]

        call(
            wire,
            {
                "op": "sandbox.download",
                "sandbox_id": sid,
                "local": "down.txt",
                "remote": "/tmp/up.txt",
            },
        )
        assert (broker._workspace_root / "down.txt").read_text() == "uploaded-bytes"

        call(
            wire,
            {"op": "sandbox.export", "sandbox_id": sid, "local": "events.jsonl"},
        )
        events_file = broker._workspace_root / "events.jsonl"
        lines = [json.loads(line) for line in events_file.read_text().splitlines() if line]
        operations = [e["operation"] for e in lines]
        assert "create" in operations and "execute" in operations
        assert "upload" in operations and "download" in operations

        closed = call(wire, {"op": "sandbox.close", "sandbox_id": sid})
        assert closed["closed"] is True
        gone = call(wire, {"op": "sandbox.exec", "sandbox_id": sid, "command": "true"})
        assert "unknown sandbox" in gone.get("error", "")
    finally:
        broker.cleanup()
        server.server_close()
