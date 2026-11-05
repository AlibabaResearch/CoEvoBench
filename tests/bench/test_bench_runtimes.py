"""Environment bindings for the three in-process benches.

None of these import their heavy dependencies at module load — ensure()
or the runtime call does it lazily — so the adapter layer (and every
test of it) runs without tau2 / appworld / the BFCL engine installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from coevobench.bench.appworld import runtime as appworld_rt
from coevobench.bench.bfcl import runtime as bfcl_rt
from coevobench.bench.tau import runtime as tau_rt


def test_tau_settings_freeze_user_simulator_protocol() -> None:
    settings = tau_rt.load_settings()
    # D.2 constants: user simulator temperature 0, 600-token output.
    assert settings["user_args"] == {"temperature": 0.0, "max_tokens": 600}
    # Table 5: agent loop 200 steps, 1024 output tokens, 1200s wall.
    assert settings["max_steps"] == 200
    assert settings["max_tokens"] == 1024
    assert settings["wall_cap_s"] == 1200


def test_tau_load_tasks_rejects_unknown_ids() -> None:
    class FakeRunner:
        class task:  # noqa: N801
            id = "known-1"

        @staticmethod
        def get_tasks(domain):
            assert domain == "telecom"
            return [FakeRunner.task]

    monkeypatched = sys.modules
    original = monkeypatched.get("tau2.runner")
    monkeypatched["tau2.runner"] = FakeRunner
    try:
        tasks = tau_rt.load_tasks(["telecom:known-1"])
        assert tasks == [FakeRunner.task]
        with pytest.raises(ValueError, match="unknown telecom tasks"):
            tau_rt.load_tasks(["telecom:nope"])
    finally:
        if original is not None:
            monkeypatched["tau2.runner"] = original
        else:
            monkeypatched.pop("tau2.runner", None)


def test_appworld_settings_validation() -> None:
    with pytest.raises(ValueError, match="invalid AppWorld"):
        appworld_rt.settings(max_steps=0)
    with pytest.raises(ValueError, match="invalid AppWorld"):
        appworld_rt.settings(request_timeout_s=-1)
    settings = appworld_rt.settings()
    assert settings["max_steps"] == 50
    assert settings["max_tokens"] == 3000


def test_appworld_run_task_reads_the_evaluator() -> None:
    class FakeTracker:
        success = True

    class FakeWorld:
        executed = False
        evaluated = False
        tracker = FakeTracker()

        def execute(self, code):
            self.executed = code == "print(1)"

        def evaluate(self):
            self.evaluated = True
            return self.tracker

    world = FakeWorld()
    payload = appworld_rt.run_task(world, "print(1)")
    assert world.executed and world.evaluated
    assert payload["tgc"] is True


def test_bfcl_settings_require_tokenizer() -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        bfcl_rt.settings()
    settings = bfcl_rt.settings(tokenizer="/models/qwen")
    assert settings["max_steps"] == 20
    assert settings["context_limit"] == 65536
    assert settings["temperature"] == pytest.approx(0.001)


def test_bfcl_settings_validation() -> None:
    with pytest.raises(ValueError, match="invalid BFCL"):
        bfcl_rt.settings(tokenizer="/m", max_tokens=8192)
    with pytest.raises(ValueError, match="invalid BFCL"):
        bfcl_rt.settings(tokenizer="/m", max_steps=0)


def test_bfcl_grading_env_wires_the_official_engine(tmp_path) -> None:
    """grading_env carries the pinned engine root and no credentials."""
    project = tmp_path / "proj"
    project.mkdir()
    env = bfcl_rt.grading_env(project)
    assert env["BFCL_PROJECT_ROOT"] == str(project)
    assert "API_KEY" not in " ".join(env.values()).upper()


def test_bfcl_run_repeat_orchestrates_two_stages(tmp_path, monkeypatch) -> None:
    calls = []

    calls = []

    def fake_container(command, name, owner, log, timeout):
        verb = "generate" if "generate" in command else "evaluate"
        calls.append(verb)
        # the engine container writes verdicts through the mounted
        # project dir; find it from the command's -v flag
        for i, arg in enumerate(command):
            if arg == "-v" and ":" in command[i + 1]:
                host, _, _ = command[i + 1].partition(":")
                project = Path(host)
                (project / "verdicts.jsonl").write_text(
                    '{"id": "t1", "correct": true, "exit_status": "Submitted"}\n'
                )
        return 0

    import pytest

    pytest.importorskip("bfcl_eval")
    monkeypatch.setattr(bfcl_rt, "image_identity", lambda image=None: {}, raising=False)
    monkeypatch.setattr(bfcl_rt, "verify_upstream", lambda: {}, raising=False)
    monkeypatch.setattr(bfcl_rt, "source_records", lambda: {"t1": {}}, raising=False)
    monkeypatch.setattr(
        bfcl_rt, "generation_plan", lambda ids, records: [{"id": "t1"}], raising=False
    )
    import coevobench.runtime.run_resources as rr

    monkeypatch.setattr(rr, "run_container", fake_container, raising=False)
    (tmp_path / "harness").mkdir(exist_ok=True)
    (tmp_path / "harness" / "default_prompts.py").write_text("# policy\n")
    (tmp_path / "harness" / "weak_policy.py").write_text("# policy\n")
    (tmp_path / "tok").mkdir(exist_ok=True)
    options = bfcl_rt.settings(tokenizer=str(tmp_path / "tok"))
    project = bfcl_rt.run_repeat(
        ["t1"], 0, options,
        harness_dir=tmp_path / "harness",
        private_dir=tmp_path / "private",
    )
    assert sorted(set(calls)) == ["evaluate", "generate"]
    rows = bfcl_rt.read_results(project)
    assert rows[0]["correct"] is True


def test_bfcl_read_results_validates_completeness(tmp_path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "grading-input.json").write_text(
        '{"records": {"t1": {}, "t2": {}}}'
    )
    (project / "verdicts.jsonl").write_text(
        '{"id": "t1", "correct": true, "exit_status": "Submitted"}\n'
        '{"id": "t2", "correct": false, "exit_status": "RecordIncorrect"}\n'
    )
    rows = bfcl_rt.read_results(project)
    assert {r["id"]: r["correct"] for r in rows} == {"t1": True, "t2": False}
    # incomplete verdicts are refused, never silently scored
    (project / "verdicts.jsonl").write_text(
        '{"id": "t1", "correct": true, "exit_status": "Submitted"}\n'
    )
    with pytest.raises(ValueError, match="incomplete"):
        bfcl_rt.read_results(project)
