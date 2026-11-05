"""Real native AppWorld worlds plus isolated candidate policy; no GPU/API billing."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from coevobench.bench.appworld.assets import data_root, python_path, verify_source
from coevobench.bench.appworld.replay import AppWorldJudge
from coevobench.bench.appworld.runtime import WorldSession
from coevobench.bench.appworld.tasks import materialize_appworld
from coevobench.core.evaluation.candidate import Candidate
from coevobench.core.protocol.protocol_assets import frozen_splits
from coevobench.runtime.sandbox.docker import DockerSandboxRuntime

PROJECT = Path(__file__).resolve().parents[2]
VISIBLE_ORACLE_TASK = "3650990_1"


@pytest.fixture
def installed():
    if not python_path().exists() or not (data_root() / "data/tasks").is_dir():
        pytest.skip("run scripts/setup-appworld.py")
    verify_source()


def test_real_world_keeps_state_and_official_grading(installed, tmp_path):
    world = WorldSession(VISIBLE_ORACLE_TASK, tmp_path / "world")
    try:
        assert world.call("execute", code="remember = 37")
        assert world.call("execute", code="print(remember + 5)")["observation"].strip() == "42"
        assert world.call("evaluate")["tgc"] is False
    finally:
        world.close()
    another = WorldSession(VISIBLE_ORACLE_TASK, tmp_path / "another")
    try:
        assert "NameError" in another.call("execute", code="print(remember)")["observation"]
    finally:
        another.close()


def test_materialize_never_publishes_private_tasks(installed, tmp_path):
    result = materialize_appworld(tmp_path / "workspace", tmp_path / "private.jsonl")
    assert result["n_tasks"] == 129
    public = tmp_path / "workspace/tasks_visible/appworld"
    assert {p.stem for p in public.glob("*.json")} == set(frozen_splits("appworld").visible)
    assert all(
        set(json.loads(p.read_text())) == {"task_id", "instruction", "supervisor"}
        for p in public.glob("*.json")
    )
    assert (tmp_path / "private.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("correct", [True, False])
def test_real_policy_to_world_to_official_grade(installed, tmp_path, monkeypatch, correct):
    image = os.environ.get("CEB_APPWORLD_POLICY_IMAGE", "coevobench-appworld-policy:latest")
    try:
        available = (
            subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except OSError:
        available = False
    if not available:
        pytest.skip("build deploy/appworld-policy.Dockerfile for local integration tests")
    monkeypatch.setenv("CEB_APPWORLD_POLICY_IMAGE", image)
    assert VISIBLE_ORACLE_TASK in frozen_splits("appworld").visible
    # A deterministic operator fixture, used only for Visible integration validation.
    # This is not a measured model run and never reads Selection/Test answers.
    answer = json.loads(
        (data_root() / f"data/tasks/{VISIBLE_ORACLE_TASK}/ground_truth/answer.json").read_text()
    )
    if not correct:
        answer = "deliberately invalid answer"
    code = "apis.supervisor.complete_task(answer=" + repr(answer) + ")"
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            body = json.dumps(
                {
                    "choices": [{"message": {"content": f"```python\n{code}\n```"}}],
                    "usage": {"completion_tokens": 12},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        candidate = Candidate(
            "fixture", tmp_path, PROJECT / "templates/appworld/harness/src", {}, "appworld"
        )
        judge = AppWorldJudge(
            candidate,
            {"split": "visible"},
            tmp_path / "results",
            runtime=DockerSandboxRuntime(),
            served={
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "model_name": "fixture",
            },
        )
        verdicts = judge.evaluate_batch([VISIBLE_ORACLE_TASK], 2, 1)
        assert [v.resolved for v in verdicts] == [correct, correct]
        assert {v.exit_status for v in verdicts} == {"Submitted"}
        assert len(judge.visible_results) == 2
        assert len(requests) == 2
        assert all(row["max_tokens"] == 3000 and row["seed"] == 100 for row in requests)
        assert all(row["chat_template_kwargs"] == {"enable_thinking": False} for row in requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
