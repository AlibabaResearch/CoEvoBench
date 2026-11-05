"""Exercise the installed client, real broker wire, and evaluation scheduling together."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coevobench.core.protocol.spec import Verdict
from coevobench.research.server import Broker, BrokerError, BrokerServer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def installed_client(tmp_path_factory):
    root = tmp_path_factory.mktemp("installed-client")
    subprocess.run(
        [sys.executable, str(ROOT / "envelope/build.py"), "--out", str(root)], check=True,
        capture_output=True, text=True,
    )
    subprocess.run(["uv", "venv", str(root / "venv")], check=True, capture_output=True)
    python = root / "venv/bin/python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(next((root / "dist").glob("*.whl")))],
        check=True, capture_output=True,
    )
    return root / "venv/bin/ceb"


def test_installed_client_preserves_workers_and_recipe(installed_client, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "serve_config.json").write_text(json.dumps({
        "tensor_parallel": 2, "max_len": 8192, "enforce_eager": True,
        "enable_prefix_caching": False,
        "extra_args": ["--limit-mm-per-prompt", '{"image":0,"video":0}'],
    }))
    broker = Broker(workspace_root=workspace, bench="tb", runtime=object())
    seen = []
    barrier = threading.Barrier(4)

    def factory(bench, submission, request):
        seen.append(request)

        def judge(task):
            # A serialized client used to silently replace workers=4 with 1.
            barrier.wait(timeout=5)
            return Verdict(False, 0.0, "Submitted")

        return judge

    monkeypatch.setattr(broker._ctx, "candidate_judge_for", factory)
    with tempfile.TemporaryDirectory(prefix="ceb-rpc-") as td:
        socket = str(Path(td) / "b.sock")
        with BrokerServer(socket, broker) as server:
            server.serve_forever_bg()
            env = {**os.environ, "CEB_WORKSPACE": str(workspace), "CEB_BROKER_SOCKET": socket}
            try:
                result = subprocess.run(
                    [str(installed_client), "eval", "--bench", "tb", "--model", "./base",
                     "--n", "4", "--workers", "4"],
                    cwd=workspace, env=env, capture_output=True, text=True, timeout=15,
                )
            finally:
                server.shutdown()
    assert result.returncode == 0, result.stderr + result.stdout
    assert "n_bad=0" in result.stdout
    assert seen[0]["workers"] == 4
    assert seen[0]["serve_config"]["tensor_parallel"] == 2
    assert seen[0]["serve_config"]["max_len"] == 8192
    assert seen[0]["serve_config"]["enable_prefix_caching"] is False
    assert list((workspace / "out/eval").glob("*.jsonl"))


def test_recipe_precedence_and_relative_template(tmp_path, monkeypatch):
    from coevobench.research.eval_client import _serving_recipe

    monkeypatch.setenv("CEB_WORKSPACE", str(tmp_path))
    (tmp_path / "serve_config.json").write_text('{"tensor_parallel":2,"max_len":32768}')
    custom = tmp_path / "custom"
    custom.mkdir()
    recipe = custom / "recipe.json"
    recipe.write_text('{"max_len":8192,"chat_template":"chat.jinja"}')
    assert _serving_recipe("/base", None)["tensor_parallel"] == 2
    actual = _serving_recipe("/base", str(recipe))
    assert actual == {"max_len": 8192, "chat_template": str(custom / "chat.jinja")}
    recipe.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        _serving_recipe("/base", str(recipe))
    with pytest.raises(OSError):
        _serving_recipe("/base", str(custom / "missing.json"))


def test_cli_overrides_and_template_reach_candidate(tmp_path, monkeypatch):
    from coevobench.research import judges

    package = tmp_path / "harness/src/minisweagent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (tmp_path / "chat.jinja").write_text("{{ messages }}")
    captured = {}

    def build(bench, candidate, *args):
        captured["candidate"] = candidate
        return lambda _: Verdict(False, 0.0, "Submitted")

    monkeypatch.setattr(judges, "_sandboxed_candidate_judge", build)
    judges.candidate_judge_for("tb", None, {
        "model": "weights", "serve_config": {"max_len": 4096, "tool_call_parser": "old"},
        "max_model_len": 8192, "tool_call_parser": "new",
        "chat_template": "/workspace/chat.jinja",
    }, runtime=object(), workspace_root=tmp_path)
    candidate = captured["candidate"]
    assert candidate.serve_shape["max_len"] == 8192
    assert candidate.serve_shape["tool_call_parser"] == "new"
    assert candidate.serve_shape["chat_template"] == str(tmp_path / "chat.jinja")


def test_eval_plan_uses_full_split_and_verifier_budget():
    broker = Broker(runtime=object())
    plan = broker.dispatch({"op": "eval.plan", "bench": "tb", "split": "visible",
                            "workers": 4, "k": 2})
    assert plan["n_tasks"] == 20
    assert plan["timeout_s"] > 20 * 2 * (900 + 600) / 4
    assert plan["workers"] == 4
    assert not any(key in plan for key in ("ids", "tasks", "task_ids"))
    limited = broker.dispatch({"op": "eval.plan", "bench": "tb", "split": "visible",
                               "n": 2, "workers": 4, "formal": True, "k": 3})
    assert limited["n_tasks"] == 2 and limited["k"] == 1
    for key in ("n", "workers", "k"):
        with pytest.raises(BrokerError):
            broker.dispatch({"op": "eval.plan", "bench": "tb", "split": "visible", key: 0})


def test_selection_persistence_stays_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "run/workspace"
    broker = Broker(runtime=object(), workspace_root=workspace, bench="tb")
    monkeypatch.setattr(broker._ctx, "candidate_judge_for", lambda *a: (
        lambda task: Verdict(False, 0.0, "Submitted")
    ))
    result = broker.dispatch({"op": "eval.run", "bench": "tb", "split": "selection",
                              "model": "/base", "n": 2})
    assert result["n_tasks"] == 2
    assert list((workspace.parent / "select_runs").glob("*.jsonl"))
    assert not (workspace / "out/eval").exists()


def test_socket_timeout_is_an_explanation_not_a_traceback(monkeypatch, capsys):
    from coevobench.research.eval_client import run_eval

    def call(socket, request, *, timeout):
        if request["op"] == "eval.plan":
            return {"timeout_s": 12345}
        assert timeout == 12345
        raise TimeoutError("timed out")

    monkeypatch.setattr("coevobench.runtime.broker_wire.call", call)
    assert run_eval("tb", "visible") == 3
    assert "before retrying" in capsys.readouterr().err


def test_evaluation_outer_lease_keeps_one_serve_across_waves(monkeypatch):
    from coevobench.research.judges import _ServingPool

    events = []

    @contextlib.contextmanager
    def serve(*args):
        events.append("start")
        try:
            yield {"base_url": "http://example"}
        finally:
            events.append("stop")

    monkeypatch.setattr("coevobench.core.evaluation.candidate.candidate_serve", serve)
    pool = _ServingPool(object(), 18900)
    with pool.endpoint():
        for _ in range(3):
            with ThreadPoolExecutor(4) as executor:
                def task(_):
                    with pool.endpoint() as endpoint:
                        return endpoint

                assert len(list(executor.map(task, range(4)))) == 4
            assert events == ["start"]
    assert events == ["start", "stop"]


def test_overlapping_eval_is_rejected_and_slot_released(monkeypatch):
    broker = Broker(runtime=object())
    entered, release = threading.Event(), threading.Event()

    def evaluate(ctx, request):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("serve failed")

    monkeypatch.setattr("coevobench.research.eval_ops._evaluate", evaluate)
    with ThreadPoolExecutor(1) as executor:
        first = executor.submit(broker.dispatch, {"op": "eval.run"})
        assert entered.wait(5)
        try:
            with pytest.raises(BrokerError, match="already running"):
                broker.dispatch({"op": "eval.run"})
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="serve failed"):
            first.result()
    assert not broker._ctx.evaluation_lock.locked()


def test_parallel_episodes_do_not_change_process_environment(tmp_path, monkeypatch):
    from coevobench.core.evaluation.candidate import Candidate, run_candidate_episode

    monkeypatch.setenv("CEB_SANDBOX_ID", "operator-context")
    seen = []
    lock = threading.Lock()

    class Sandbox:
        def __init__(self, ident):
            self.id = ident

        def kill(self):
            pass

    class Runtime:
        name = "docker"

        def __init__(self, ident):
            self.ident = ident

        def create(self, spec):
            return Sandbox(self.ident)

    def execute(argv, *, env, **kwargs):
        with lock:
            seen.append((env["CEB_TASK_ID"], env["CEB_SANDBOX_ID"]))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", execute)
    candidate = Candidate("test", tmp_path, tmp_path)

    def episode(ident):
        return run_candidate_episode(
            candidate, ident, {"base_url": "http://localhost", "model_name": "m"},
            bench="tb", task_data={"instruction": "test"}, sandbox_runtime=Runtime(ident),
        )

    with ThreadPoolExecutor(4) as executor:
        list(executor.map(episode, ["one", "two", "three", "four"]))
    assert all(task == sandbox for task, sandbox in seen)
    assert os.environ["CEB_SANDBOX_ID"] == "operator-context"
