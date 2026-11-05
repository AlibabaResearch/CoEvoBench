"""The envelope eval client forwards to the host; the host judges.

This is the integrity-critical path: the researcher-side ./eval entry
must never see split data or run judgments itself.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from coevobench.research.server import Broker, BrokerServer
from coevobench.runtime.broker_wire import call


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


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="docker daemon not available"
)


def _broker(tmp_name: str):
    broker = Broker()
    sock_dir = Path(tempfile.mkdtemp(prefix="ceb-ev-"))
    server = BrokerServer(str(sock_dir / "b.sock"), broker)
    server.serve_forever_bg()
    return broker, server, str(sock_dir / "b.sock")


def test_eval_run_visible_over_broker() -> None:
    broker, server, socket_path = _broker("vis")
    try:
        response = call(
            socket_path,
            {"op": "eval.run", "bench": "toy", "split": "visible", "k": 1},
            timeout=300,
        )
        assert response["n_tasks"] == 2
        assert response["score"] == pytest.approx(1.0)
        assert response["n_bad"] == 0
    finally:
        broker.cleanup()
        server.server_close()


def test_eval_run_selection_returns_only_scalar() -> None:
    broker, server, socket_path = _broker("sel")
    try:
        response = call(
            socket_path,
            {"op": "eval.run", "bench": "toy", "split": "selection", "k": 1},
            timeout=300,
        )
        assert response["feedback"].startswith("resolved_rate=")
        # The selection contract: no task ids in the response.
        assert "task_ids" not in response
    finally:
        broker.cleanup()
        server.server_close()


def test_eval_run_refuses_test_and_unknown_split() -> None:
    broker, server, socket_path = _broker("ref")
    try:
        response = call(
            socket_path,
            {"op": "eval.run", "bench": "toy", "split": "test"},
            timeout=60,
        )
        # the wire layer turns exceptions into error responses; the
        # refusal must reach the caller either way
        assert "error" in response
        assert "visible" in response["error"]
    finally:
        broker.cleanup()
        server.server_close()


def test_envelope_client_shape() -> None:
    """The client module imports without split data present and refuses
    test outright — its only job is forwarding."""
    from coevobench.research import eval_client

    assert eval_client.run_eval("toy", "test") == 2
    assert eval_client.run_eval("toy", "weird") == 2
