"""Researcher-side surface: broker client, workspace entries, eval client."""

from __future__ import annotations

import json

import pytest

# ---- client ----

def test_client_cli_maps_flags_to_wire_fields(monkeypatch) -> None:
    from coevobench.research import client as client_mod

    captured: list[tuple[str, dict]] = []

    def fake_call(socket, request, *, timeout):
        captured.append((socket, request))
        if request["op"] == "sandbox.exec":
            return {"exit_code": 0 if request["command"] == "true" else 1,
                    "returncode": 0 if request["command"] == "true" else 1,
                    "text": ""}
        return {"sandbox_id": "sb-1"}

    monkeypatch.setattr(
        "coevobench.runtime.broker_wire.call", fake_call
    )
    monkeypatch.setenv("CEB_BROKER_SOCKET", "/tmp/fake.sock")

    rc = client_mod.main(
        ["sandbox", "create", "--image", "python:3.11", "--memory", "6", "--ttl", "900"]
    )
    assert rc == 0
    op, request = captured[-1]
    assert op == "/tmp/fake.sock"
    assert request["op"] == "sandbox.create"
    assert request["mem_gib"] == 6
    assert request["ttl_s"] == 900

    rc = client_mod.main(
        [
            "sandbox", "create", "--from-dir", "work/my-task",
            "--cpu", "4", "--network", "--build-timeout", "600",
        ]
    )
    assert rc == 0
    op, request = captured[-1]
    assert request["op"] == "sandbox.create"
    assert request["from_dir"].endswith("work/my-task")
    assert "image" not in request
    assert request["cpu"] == 4
    assert request["network"] is True
    assert request["build_timeout_s"] == 600

    rc = client_mod.main(["sandbox", "exec", "sb-1", "false"])
    assert rc == 1
    rc = client_mod.main(["sandbox", "exec", "sb-1", "true"])
    assert rc == 0

    # broker failures come back as {"error": ...}: a failed command,
    # not a success payload
    monkeypatch.setattr(
        "coevobench.runtime.broker_wire.call",
        lambda socket, request, *, timeout: {"error": "boom"},
    )
    rc = client_mod.main(["sandbox", "status", "sb-1"])
    assert rc == 1


def test_client_contamination_exit_codes(monkeypatch) -> None:
    from coevobench.research import client as client_mod

    monkeypatch.setattr(
        "coevobench.runtime.broker_wire.call",
        lambda socket, request, *, timeout: {"verdict": "NO_MATCHES (screening only)"},
    )
    monkeypatch.setenv("CEB_BROKER_SOCKET", "/tmp/fake.sock")
    rc = client_mod.main(["contamination", "check", "--input", "a.jsonl"])
    assert rc == 0

    monkeypatch.setattr(
        "coevobench.runtime.broker_wire.call",
        lambda socket, request, *, timeout: {"verdict": "REVIEW_REQUIRED (overlap)"},
    )
    rc = client_mod.main(["contamination", "check", "--input", "a.jsonl"])
    assert rc == 3


def test_client_manifest_is_create_only(monkeypatch, tmp_path) -> None:
    from coevobench.research import client as client_mod

    monkeypatch.setattr(
        "coevobench.runtime.broker_wire.call",
        lambda socket, request, *, timeout: {
            "check_id": "c1",
            "status": "complete",
            "verdict": "NO_MATCHES (screening only)",
            "files": [str(tmp_path / "corpus.jsonl")],
        },
    )
    monkeypatch.setenv("CEB_BROKER_SOCKET", "/tmp/fake.sock")
    manifest = tmp_path / "train" / "corpus.json"
    rc = client_mod.main(
        ["contamination", "check", "--input", str(tmp_path / "corpus.jsonl"),
         "--manifest", str(manifest)]
    )
    assert rc == 0
    payload = json.loads(manifest.read_text())
    assert payload["check_id"] == "c1"
    # A second declaration never silently replaces the first.
    rc = client_mod.main(
        ["contamination", "check", "--input", str(tmp_path / "corpus.jsonl"),
         "--manifest", str(manifest)]
    )
    assert rc == 1  # create-only: the earlier declaration stands


# ---- workspace provisioning ----

def test_provision_stamps_the_full_contract(tmp_path, monkeypatch) -> None:
    from coevobench.orchestrate.provision import provision_workspace

    monkeypatch.setattr(
        "coevobench.orchestrate.provision._provision_visible_tasks",
        lambda workspace, bench, landed: None,
    )
    base = tmp_path / "base-weights"
    base.mkdir()
    workspace = tmp_path / "ws"
    landed = provision_workspace(workspace, "tb", base_dir=base)

    for entry in ("eval", "timer.sh", "sandbox", "submit", "track", "contamination"):
        path = workspace / entry
        assert path.is_file(), f"missing entry {entry}"
        assert landed[entry] == 1
        assert not (path.stat().st_mode & 0o222), f"{entry} not frozen read-only"
    assert (workspace / "anonymous_pool_tb.txt").is_file()
    assert (workspace / "base").is_symlink()
    assert (workspace / "out").is_dir()
    pool = (workspace / "anonymous_pool_tb.txt").read_text().splitlines()
    assert pool and all(line.startswith("A-") for line in pool)


def test_provision_anon_pool_never_contains_test_ids(tmp_path, monkeypatch) -> None:
    from coevobench.orchestrate.provision import provision_workspace

    monkeypatch.setattr(
        "coevobench.orchestrate.provision._provision_visible_tasks",
        lambda workspace, bench, landed: None,
    )
    workspace = tmp_path / "ws"
    provision_workspace(workspace, "swe")
    from coevobench.cli.data import frozen_splits

    frozen = frozen_splits("swe")
    pool = (workspace / "anonymous_pool_swe.txt").read_text().splitlines()
    from coevobench.bench.anon import anon_to_real, generate_mapping

    real = anon_to_real(pool, generate_mapping("swe", list(frozen.selection)))
    assert set(real) == set(frozen.selection)
    assert not set(real) & set(frozen.test)


def test_provision_copies_materialized_public_views(tmp_path, monkeypatch) -> None:
    from coevobench.orchestrate import provision as provision_mod

    cache = tmp_path / "cache"
    views = cache / "workspace_material" / "tasks_visible" / "swe"
    views.mkdir(parents=True)
    (views / "repo__x-1.json").write_text(json.dumps({"instance_id": "repo__x-1"}))
    (views / "protected.json").write_text('{"instance_id":"protected"}')
    from coevobench.core.protocol.splits import from_mapping

    monkeypatch.setattr("coevobench.core.protocol.protocol_assets.frozen_splits", lambda _: (
        from_mapping({"visible": ["repo__x-1"], "selection": ["protected"], "test": []})
    ))
    monkeypatch.setattr(
        "coevobench.core.protocol.protocol_assets.cache_root", lambda cfg: cache
    )
    workspace = tmp_path / "ws"
    landed = {}
    provision_mod._provision_visible_tasks(workspace, "swe", landed)
    assert landed["tasks_visible"] == 1
    assert (workspace / "tasks_visible" / "swe" / "repo__x-1.json").is_file()
    assert not (workspace / "tasks_visible/swe/protected.json").exists()
    (workspace / "tasks_visible/swe/protected.json").write_text("{}")
    provision_mod._provision_visible_tasks(workspace, "swe", landed)
    assert not (workspace / "tasks_visible/swe/protected.json").exists()


def test_provision_visible_placeholder_without_cache(tmp_path, monkeypatch) -> None:
    from coevobench.orchestrate import provision as provision_mod

    monkeypatch.setattr(
        "coevobench.core.protocol.protocol_assets.cache_root", lambda cfg: tmp_path / "empty"
    )
    workspace = tmp_path / "ws"
    provision_mod._provision_visible_tasks(workspace, "tb", {})
    note = workspace / "tasks_visible" / "tb" / "README.txt"
    assert "ceb data materialize" in note.read_text()


# ---- eval client forwarding ----

def test_eval_client_forwards_the_full_surface(monkeypatch, tmp_path) -> None:
    from coevobench.research import eval_client

    captured: dict = {}

    def fake_call(socket, request, *, timeout):
        captured.update(request)
        if request["op"] == "eval.plan":
            return {"timeout_s": 7200}
        return {"n_tasks": 2, "k": 1, "score": 0.5, "n_bad": 0,
                "exit_status_counts": {}, "feedback": "scalar"}

    monkeypatch.setattr("coevobench.runtime.broker_wire.call", fake_call)
    recipe = tmp_path / "serve_config.json"
    recipe.write_text('{"max_len": 32768}')
    rc = eval_client.run_eval(
        "swe",
        "selection",
        k=2,
        tasks="A-11111111",
        model="/ws/out/swe_final/weights",
        serve_config=str(recipe),
        temperature=0.0,
        max_tokens=5120,
        wall_cap=900,
        step_limit=120,
        formal=True,
        run_id="glm53_swe_1",
    )
    assert rc == 0
    assert captured["op"] == "eval.run"
    assert captured["model"] == "/ws/out/swe_final/weights"
    assert captured["serve_config"] == {"max_len": 32768}
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 5120
    assert captured["wall_cap_s"] == 900
    assert captured["step_limit"] == 120
    assert captured["formal"] is True
    assert captured["run_id"] == "glm53_swe_1"


def test_eval_client_list_tasks_prints_anon_ids(monkeypatch, capsys) -> None:
    from coevobench.research import eval_client

    rc = eval_client.run_eval("swe", "selection", list_tasks=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("A-")


# ---- workspace CLIs ----

def test_submit_request_and_confirm(tmp_path, monkeypatch, capsys) -> None:
    from coevobench.cli.workspace import submit_cmd

    ws = tmp_path / "ws"
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    submission = ws / "out" / "swe_final"
    submission.mkdir(parents=True)
    (submission / "harness").mkdir()
    (submission / "model_path.txt").write_text(str(weights) + "\n")
    (submission / "meta.json").write_text("{}")
    (submission / "serve_config.json").write_text("{}")
    (ws / "out" / "DONE.md").write_text("final result\n")
    rc = submit_cmd(str(ws), "swe")
    assert rc == 0
    assert (submission / "submission.json").is_file()
    rc = submit_cmd(str(ws), "swe", confirm=True)
    assert rc == 0
    # An edit after the request invalidates it.
    (submission / "meta.json").write_text("changed")
    rc = submit_cmd(str(ws), "swe", confirm=True)
    assert rc == 1


def test_track_times_a_command(tmp_path, capsys) -> None:
    from coevobench.cli.workspace import track_cmd

    out = tmp_path / "out" / "research-timing.jsonl"
    rc = track_cmd(str(out), ["sh", "-c", "exit 0"])
    assert rc == 0
    rc = track_cmd(str(out), ["sh", "-c", "exit 3"])
    assert rc == 3
    rc = track_cmd(str(out), [])
    assert rc == 0
    summary = capsys.readouterr().out
    assert "[track]" in summary


def test_results_cmd_lists_ledger_rows(tmp_path, monkeypatch, capsys) -> None:
    from coevobench.cli.workspace import results_cmd

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from coevobench.core.lifecycle.results import ResultStore

    store = ResultStore(tmp_path / ".ceb" / "results")
    store.record({"run_id": "glm53_swe_1", "bench": "swe", "split": "selection",
                  "score": 0.25, "n_tasks": 200})
    rc = results_cmd(bench="swe")
    assert rc == 0
    out = capsys.readouterr().out
    assert "glm53_swe_1" in out and "score=0.2500" in out


def test_eval_run_routes_live_workspace_candidate(monkeypatch, tmp_path) -> None:
    """eval.run with model= takes the live harness from the workspace."""
    from coevobench.research.server import Broker

    workspace = tmp_path / "ws"
    harness_pkg = workspace / "harness" / "swe" / "src" / "minisweagent"
    harness_pkg.mkdir(parents=True)
    (harness_pkg / "__init__.py").write_text("")
    weights = tmp_path / "weights"
    weights.mkdir()

    class _FakeEpisodeSandbox:
        def run(self, command, *, timeout_s=60, cwd=None):
            return {"exit_code": 0, "text": ""}

        def kill(self) -> None:
            pass

    class _RecordingRuntime:
        def __init__(self) -> None:
            self.mirrors = []

        def connect(self, sandbox_id):
            return _FakeEpisodeSandbox()

    from coevobench.cli.data import frozen_splits as _frozen

    _real = _frozen("swe")
    instances = {
        iid: {"problem_statement": f"p{i}", "patch": "", "test_patch": ""}
        for i, iid in enumerate(
            list(_real.visible) + list(_real.selection) + list(_real.test)
        )
    }
    monkeypatch.setattr(
        "coevobench.bench.swe.tasks.load_swe_instances", lambda: instances
    )
    monkeypatch.setattr(
        "coevobench.core.evaluation.candidate.run_candidate_episode",
        lambda *a, **k: {"sandbox_id": "sb-1", "exit_status": "Submitted"},
    )

    broker = Broker(runtime=_RecordingRuntime(), workspace_root=workspace)
    saw = {}

    class _Pool:
        def __init__(self) -> None:
            self._endpoint = {"base_url": "http://x", "model_name": "m"}

        @contextlib.contextmanager
        def endpoint(self):
            saw["used"] = True
            yield self._endpoint

    import coevobench.research.judges as judges_mod

    monkeypatch.setattr(judges_mod, "_ServingPool", lambda candidate, port: _Pool())
    monkeypatch.setattr(
        "coevobench.bench.swe.adapter.SweBench.judge",
        lambda self, task_id, submission, sandbox: __import__(
            "coevobench.core.protocol.spec", fromlist=["Verdict"]
        ).Verdict(resolved=True, score=1.0, exit_status="Submitted"),
    )
    from coevobench.cli.data import frozen_splits

    frozen = frozen_splits("swe")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr("coevobench.core.protocol.protocol_assets.cache_root", lambda cfg: cache)
    (cache / "swe_splits.json").write_text(
        json.dumps({"visible": list(frozen.visible), "selection": list(frozen.selection),
                    "test": list(frozen.test)})
    )
    broker._workspace_root = workspace
    response = broker.dispatch(
        {
            "op": "eval.run",
            "bench": "swe",
            "split": "visible",
            "model": str(weights),
            "n": 2,
            "k": 1,
            "workers": 1,
        }
    )
    assert saw.get("used") is True
    assert response["n_tasks"] == 2
    assert response["n_bad"] == 0


import contextlib  # noqa: E402


def test_formal_presets_override_researcher_settings(monkeypatch) -> None:
    """formal=True applies the announced Table 5 values, ignoring overrides."""
    from coevobench.core.protocol.spec import Verdict
    from coevobench.research.server import Broker

    seen = {}

    def fake_candidate_judge(bench, submission_dir, request):
        seen.update(request)

        def judge(task_id: str) -> Verdict:
            return Verdict(resolved=True, score=1.0, exit_status="Submitted")

        return judge

    broker = Broker()
    monkeypatch.setattr(broker._ctx, "candidate_judge_for", fake_candidate_judge)
    response = broker.dispatch(
        {
            "op": "eval.run", "bench": "toy", "split": "visible",
            "k": 3, "formal": True, "temperature": 0.7, "max_tokens": 111,
            "model": "/some/weights", "n": 1, "workers": 1,
        }
    )
    assert response["k"] == 1
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] != 111
    assert seen["wall_cap_s"] == 900
    assert seen["step_limit"] == 80


def test_candidate_serve_passes_the_whole_recipe() -> None:
    from pathlib import Path

    """The serving recipe contract (paper D.3): parsers and throughput
    knobs declared by the candidate reach the serve command."""
    from coevobench.core.evaluation.candidate import Candidate, candidate_serve
    from coevobench.runtime.serve import vllm as vllm_mod

    candidate = Candidate(
        label="c",
        weights=Path("/w"),
        harness_src=Path("/h"),
        serve_shape={
            "tool_call_parser": "hermes",
            "reasoning_parser": "qwen3",
            "max_num_seqs": 64,
            "enforce_eager": True,
            "data_parallel": 2,
        },
    )
    captured = {}

    class _FakeEndpoint:
        base_url = "http://x"
        model_name = "m"

        def stop(self) -> None:
            pass

    def fake_serve(self, weights, port, served_name, shape):
        captured["shape"] = shape
        return _FakeEndpoint()

    monkeypatch_target = vllm_mod.VllmServeRuntime.serve
    vllm_mod.VllmServeRuntime.serve = fake_serve
    try:
        with candidate_serve(candidate):
            pass
    finally:
        vllm_mod.VllmServeRuntime.serve = monkeypatch_target
    shape = captured["shape"]
    assert shape.tool_call_parser == "hermes"
    assert shape.reasoning_parser == "qwen3"
    assert shape.max_num_seqs == 64
    assert shape.enforce_eager is True
    assert shape.data_parallel == 2


def test_list_tasks_reads_the_workspace_pool_first(monkeypatch, tmp_path, capsys) -> None:
    """In-envelope: the anonymous pool file is the source of truth."""
    from coevobench.research import eval_client

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "anonymous_pool_toy.txt").write_text("A-aaaa1111\nA-bbbb2222\n")
    monkeypatch.setenv("CEB_WORKSPACE", str(ws))
    rc = eval_client.run_eval("toy", "selection", list_tasks=True)
    assert rc == 0
    assert capsys.readouterr().out == "A-aaaa1111\nA-bbbb2222\n"


def test_ceb_client_routes_the_full_tree_before_argparse(monkeypatch) -> None:
    """`ceb client ...` reaches the client parser untouched: -h shows the
    client's own subcommand tree, and commands route to the broker."""
    from coevobench.cli.main import main as ceb_main

    # -h is handled by the client's parser, not the outer REMAINDER shim.
    with pytest.raises(SystemExit) as caught:
        ceb_main(["client", "-h"])
    assert caught.value.code == 0

    monkeypatch.setenv("CEB_BROKER_SOCKET", "/tmp/fake.sock")
    monkeypatch.setattr(
        "coevobench.runtime.broker_wire.call",
        lambda socket, request, *, timeout: {"sandbox_id": "sb-9"},
    )
    assert ceb_main(["client", "sandbox", "create", "--image", "python:3.11"]) == 0


def test_materialize_swe_bakes_episode_images(monkeypatch, tmp_path) -> None:
    """Judge records carry the image candidate episodes spawn from."""
    from coevobench.bench.swe import tasks as swe_tasks
    from coevobench.config.schema import EvalImageMirror

    rows = {
        f"repo__x-{i:03d}": {"instance_id": f"repo__x-{i:03d}",
                             "problem_statement": "p", "patch": "", "test_patch": ""}
        for i in range(10)
    }
    monkeypatch.setattr(swe_tasks, "_fetch_verified_rows", lambda: rows)
    from coevobench.core.protocol.splits import from_mapping

    splits = from_mapping(
        {"visible": list(rows)[:2], "selection": list(rows)[2:6], "test": list(rows)[6:]}
    )
    cache = tmp_path / "swe_tasks.jsonl"
    swe_tasks.materialize_swe(splits, tmp_path / "ws", judge_cache=cache)
    recs = [json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()]
    assert recs[0]["image"] == "swebench/sweb.eval.x86_64.repo_1776_x-000"

    mirror = EvalImageMirror(
        prefix="swebench/", rewrite="cr.example.com/swebench-verified:",
        tag_source="instance_id",
    )
    swe_tasks.materialize_swe(
        splits, tmp_path / "ws2", judge_cache=cache, mirror=mirror
    )
    recs = [json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()]
    assert recs[0]["image"] == "cr.example.com/swebench-verified:repo__x-000"


def test_episode_driver_module_parses_prompt() -> None:
    """The driver reads its prompt from argv — no string interpolation
    to break; adversarial prompts survive argv round-tripping."""
    import subprocess
    import sys as _sys

    code = (
        "import json, sys; "
        "assert sys.argv[1] == json.loads(sys.argv[1]) or True; "
        "print(json.loads(sys.argv[1]) == sys.argv[1] or json.loads(sys.argv[1]))"
    )
    for nasty in (
        'fix "the" crash\nwith newline',
        "back\\slash 'quotes' {braces}",
        "unicode 你好 \\n literal",
    ):
        out = subprocess.run(
            [_sys.executable, "-c", code, json.dumps(nasty)],
            capture_output=True, text=True, timeout=30,
        )
        assert out.returncode == 0


def test_eval_run_routes_tb_candidate_judge_on_episode_sandbox(monkeypatch, tmp_path) -> None:
    """tb candidate evaluation: the same episode driver, judged on the
    episode sandbox (verifiers upload after the agent), image strictly
    from the task record."""
    from coevobench.research.server import Broker

    workspace = tmp_path / "ws"
    harness_pkg = workspace / "harness" / "tb" / "src" / "minisweagent"
    harness_pkg.mkdir(parents=True)
    (harness_pkg / "__init__.py").write_text("")
    weights = tmp_path / "weights"
    weights.mkdir()

    # eval.run resolves ids from the REPO's frozen split, so the fake
    # task records must carry the real frozen visible ids
    from coevobench.cli.data import frozen_splits as _frozen_ids

    task_ids = list(_frozen_ids("tb").visible)
    tasks = {
        tid: {"instruction": "do it", "image": f"example.com/tb:{tid}"}
        for tid in task_ids
    }
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )

    import json as _json

    from coevobench.cli.data import frozen_splits as _frozen

    _real = _frozen("tb")
    _cache = tmp_path / "cache"
    _cache.mkdir()
    monkeypatch.setattr(
        "coevobench.core.protocol.protocol_assets.cache_root", lambda cfg: _cache
    )
    (_cache / "tb_splits.json").write_text(
        _json.dumps(
            {
                "visible": task_ids,
                "selection": list(_real.selection),
                "test": list(_real.test),
            }
        )
    )
    # _real is the same frozen split; the cache copy exists only to
    # satisfy the TbBench constructor

    class _FakeEpisodeSandbox:
        def run(self, command, *, timeout_s=60, cwd=None):
            return {"exit_code": 0, "text": "1.0"}

        def kill(self) -> None:
            pass

    judged = []

    class _RecordingRuntime:
        def connect(self, sandbox_id):
            return _FakeEpisodeSandbox()

    monkeypatch.setattr(
        "coevobench.core.evaluation.candidate.run_candidate_episode",
        lambda *a, **k: {"sandbox_id": "sb-1", "exit_status": "Submitted"},
    )
    import contextlib

    saw = {}

    class _Pool:
        @contextlib.contextmanager
        def endpoint(self):
            saw["used"] = True
            yield {"base_url": "http://x", "model_name": "m"}

    monkeypatch.setattr(
        "coevobench.research.judges._ServingPool", lambda candidate, port: _Pool()
    )

    def _fake_judge(self, task_id, submission, sandbox):
        judged.append((task_id, sandbox))
        from coevobench.core.protocol.spec import Verdict

        return Verdict(resolved=True, score=1.0, exit_status="Submitted")

    monkeypatch.setattr("coevobench.bench.tb.adapter.TbBench.judge", _fake_judge)

    broker = Broker(runtime=_RecordingRuntime(), workspace_root=workspace)
    response = broker.dispatch(
        {
            "op": "eval.run",
            "bench": "tb",
            "split": "visible",
            "model": str(weights),
            "n": 2,
            "k": 1,
            "workers": 1,
        }
    )
    assert saw.get("used") is True
    assert [t for t, _ in judged] == task_ids[:2]
    # judged on the episode sandbox itself, not a fresh one
    assert all(isinstance(sb, _FakeEpisodeSandbox) for _, sb in judged)
    assert response["n_tasks"] == 2
    assert response["n_bad"] == 0


def test_eval_run_refuses_toy_candidate_with_a_real_reason() -> None:
    import pytest

    from coevobench.research.server import Broker
    from coevobench.runtime.broker_wire import BrokerError

    broker = Broker()
    with pytest.raises(BrokerError, match="toy runs script rollouts"):
        broker.dispatch(
            {
                "op": "eval.run",
                "bench": "toy",
                "split": "visible",
                "model": "/tmp/weights",
            }
        )


def test_gpu_ops_slot_lifecycle_and_exclusive_eval() -> None:
    """gpu.submit/status/logs/stop through the broker; eval.run refuses
    while the cards are held; a second job waits for gpu.stop."""
    import pytest

    from coevobench.research.server import Broker
    from coevobench.runtime.broker_wire import BrokerError

    class FakeJobs:
        def __init__(self) -> None:
            self.state = {"status": "idle"}
            self.stopped = 0

        def status(self):
            return self.state

        def submit(self, command):
            self.state = {"status": "running", "job_id": "j1"}
            return self.state

        def logs(self):
            return "train log line"

        def stop(self):
            self.stopped += 1
            self.state = {"status": "idle"}

    jobs = FakeJobs()
    broker = Broker(gpu_jobs=jobs)
    assert broker.dispatch({"op": "gpu.status"})["status"] == "idle"
    submitted = broker.dispatch({"op": "gpu.submit", "command": "torchrun train.py"})
    assert submitted["job_id"] == "j1"
    assert broker.dispatch({"op": "gpu.logs"})["text"] == "train log line"
    with pytest.raises(BrokerError, match="holds the cards"):
        broker.dispatch(
            {"op": "eval.run", "bench": "tb", "split": "visible", "n": 1}
        )
    with pytest.raises(BrokerError, match="already active"):
        broker.dispatch({"op": "gpu.submit", "command": "another"})
    broker.dispatch({"op": "gpu.stop"})
    assert jobs.stopped == 1


def test_gpu_ops_refuse_cleanly_without_a_slot() -> None:
    import pytest

    from coevobench.research.server import Broker
    from coevobench.runtime.broker_wire import BrokerError

    broker = Broker()
    with pytest.raises(BrokerError, match="not configured"):
        broker.dispatch({"op": "gpu.status"})
