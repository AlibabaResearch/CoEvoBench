"""AppWorld protocol, private data, candidate binding, and aggregation contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from coevobench.bench.appworld.replay import AppWorldJudge, scene_metrics
from coevobench.bench.appworld.runtime import run_task
from coevobench.core.evaluation.candidate import Candidate
from coevobench.core.grading.freeze import freeze, validate, verify_unchanged
from coevobench.core.protocol.spec import Verdict

PROJECT = Path(__file__).resolve().parents[2]


def test_official_evaluator_is_authoritative():
    class Tracker:
        success = False

    class World:
        def execute(self, code):
            return "I claim this task passed"

        def evaluate(self):
            return Tracker()

    assert run_task(World(), "anything")["tgc"] is False


def test_appworld_candidate_freeze_and_mutation(tmp_path):
    shutil.copytree(PROJECT / "templates/appworld/harness", tmp_path / "harness/appworld")
    (tmp_path / "model_path.txt").write_text(str(tmp_path / "weights"))
    (tmp_path / "meta.json").write_text(json.dumps({"bench": "appworld"}))
    candidate = Candidate.from_dir(tmp_path, "appworld")
    assert candidate.harness_src.name == "src"
    assert validate(tmp_path) == ()
    assert freeze(tmp_path, "test").ok
    assert verify_unchanged(tmp_path)
    path = candidate.harness_src / "appworldharness/agent.py"
    path.chmod(0o644)
    path.write_text(path.read_text() + "\n# modification\n")
    assert not verify_unchanged(tmp_path)


def test_scene_metrics_excludes_incomplete_scenes_and_keeps_repeats():
    ids = ["a_1", "a_2", "a_3", "b_1"]
    values = [True, True, True, False, True, False, True, True]
    verdicts = [Verdict(value, float(value), "Submitted") for value in values]
    result = scene_metrics(ids, verdicts, 2)
    assert result["task_goal_completion"] == 75
    assert result["scenario_goal_completion"] == 50
    assert result["n_complete_scene_trials"] == 2


def test_formal_options_override_candidate_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_steps": 1, "temperature": 1.0}))
    candidate = Candidate("test", tmp_path, tmp_path / "src", {}, "appworld")
    judge = AppWorldJudge(candidate, {"formal": True}, runtime=object())
    assert judge._options()["max_steps"] == 50
    assert judge._options()["seed"] == 100
    assert judge._options()["temperature"] == 0


def test_symlink_policy_refused_at_freeze(tmp_path):
    shutil.copytree(PROJECT / "templates/appworld/harness", tmp_path / "harness/appworld")
    (tmp_path / "model_path.txt").write_text("/base")
    (tmp_path / "meta.json").write_text('{"bench":"appworld"}')
    package = tmp_path / "harness/appworld/src/appworldharness"
    (package / "outside").symlink_to("/etc/passwd")
    assert any("symlinks" in error for error in validate(tmp_path))


def test_appworld_broker_selection_is_scalar_and_test_is_refused(tmp_path, monkeypatch):
    import pytest

    from coevobench.research.server import Broker, BrokerError

    class Judge:
        visible_results = [{"private_detail": "only visible"}]
        metrics = {"scenario_goal_completion": 100}

        def evaluate_batch(self, ids, k, workers):
            return [Verdict(True, 1.0, "Submitted") for _ in range(len(ids) * k)]

        def __call__(self, task_id):
            return self.evaluate_batch([task_id], 1, 1)[0]

    broker = Broker(runtime=object(), workspace_root=tmp_path, bench="appworld")
    monkeypatch.setattr(
        "coevobench.research.judges.candidate_judge_for",
        lambda *a, **k: Judge(),
    )
    sel = {"op": "eval.run", "bench": "appworld", "split": "selection",
           "model": "/base", "n": 3, "k": 1, "workers": 1}
    result = broker.dispatch(sel)
    assert set(result) >= {"score", "n_tasks", "k"}
    assert result["score"] == 1
    with pytest.raises(BrokerError):
        broker.dispatch({**sel, "split": "test"})


def test_final_bundle_resolves_remote_workspace_model_and_preserves_metrics(tmp_path):
    from coevobench.core.lifecycle.final_eval import FinalReport, prepare

    workspace = tmp_path / "workspace"
    submission = workspace / "out/appworld_final"
    shutil.copytree(PROJECT / "templates/appworld/harness", submission / "harness")
    weights = workspace / "out/weights"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text("{}")
    (weights / "model.safetensors").write_bytes(b"test only")
    (submission / "model_path.txt").write_text("/workspace/out/weights")
    (submission / "meta.json").write_text('{"bench":"appworld"}')
    (submission / "serve_config.json").write_text("{}")
    bundle = prepare(submission, "appworld", tmp_path / "final")
    assert (bundle / "weights/model.safetensors").read_bytes() == b"test only"
    report = FinalReport(
        "appworld", 3, 3, 1.0, "score_mean", str(weights), metrics={"scenario_goal_completion": 100}
    )
    assert json.loads(report.to_json())["metrics"]["scenario_goal_completion"] == 100


def test_final_replay_pins_context_and_weight_reversion(tmp_path):
    from coevobench.bench.appworld.replay import replay_judge

    submission = tmp_path / "bundle/submission"
    shutil.copytree(PROJECT / "templates/appworld/harness", submission / "harness")
    (submission / "model_path.txt").write_text(str(tmp_path / "evolved"))
    (submission / "serve_config.json").write_text('{"max_len":1024}')
    judge = replay_judge(tmp_path / "bundle", "appworld", tmp_path / "base")
    assert judge.candidate.weights == tmp_path / "base"
    assert judge.candidate.serve_shape["max_len"] == 65536
