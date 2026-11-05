"""B-series contracts: comparison roles, timing spans, submission
protocol, egress scrubbing, preflight gates, result store, prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from coevobench.core.grading.egress_guard import scrub_egress_env
from coevobench.core.grading.submission import confirm, request
from coevobench.core.lifecycle.comparison import ROLES, ComparisonResult
from coevobench.core.lifecycle.preflight import (
    PreflightError,
    check_workspace_clean,
    harness_manifest,
)
from coevobench.core.lifecycle.results import ResultStore
from coevobench.core.lifecycle.timing import Recorder, summarize
from coevobench.core.protocol.prompts import (
    RESEARCHER_LOOP_PROMPT,
    TASK_BRIEF_CORE,
)

# ---- B4 comparison ----


def test_comparison_roles_present() -> None:
    assert ROLES == ("M0_H0", "M0_H1", "M1_H1")


def test_comparison_gains_read_from_scores() -> None:
    result = ComparisonResult(
        bench="swe",
        scores={
            "M0_H0": {"score": 0.02},
            "M0_H1": {"score": 0.08},
            "M1_H1": {"score": 0.21},
        },
    )
    assert result.training_gain == pytest.approx(0.06)
    assert result.harness_gain == pytest.approx(0.13)
    empty = ComparisonResult(bench="swe", scores={})
    assert empty.training_gain is None


# ---- B5 timing ----


def test_recorder_spans_and_summary(tmp_path: Path) -> None:
    log = tmp_path / "spans.jsonl"
    recorder = Recorder(log)
    with recorder.span("eval", "judge", n=3):
        time.sleep(0.01)
    with pytest.raises(ValueError), recorder.span("eval", "bad"):
        raise ValueError("boom")
    summary = summarize(log)
    judge = summary["spans"]["eval/judge"]
    assert judge["count"] == 1
    assert judge["total_s"] >= 0.0
    bad = summary["spans"]["eval/bad"]
    assert bad["count"] == 1


def test_recorder_null_path_is_noop() -> None:
    recorder = Recorder(None)
    with recorder.span("x", "y"):
        pass


# ---- B6 submission ----


@pytest.fixture
def submission_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    out = workspace / "out"
    final = out / "swe_final"
    (final / "harness").mkdir(parents=True)
    (final / "harness" / "run.py").write_text("x = 1\n")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    (final / "model_path.txt").write_text(str(weights) + "\n")
    (final / "meta.json").write_text("{}")
    (final / "serve_config.json").write_text("{}")
    (out / "DONE.md").write_text("final result summary\n")
    return workspace


def test_submission_request_and_confirm(submission_workspace: Path) -> None:
    request_file = request(submission_workspace, "swe")
    record = json.loads(request_file.read_text())
    assert record["digest"]
    result = confirm(submission_workspace, "swe")
    assert result["confirmed"] is True


def test_submission_rejects_stale_request(submission_workspace: Path) -> None:
    request(submission_workspace, "swe")
    (submission_workspace / "out" / "swe_final" / "harness" / "run.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        confirm(submission_workspace, "swe")


def test_submission_requires_done_md(submission_workspace: Path) -> None:
    workspace = submission_workspace
    (workspace / "out" / "DONE.md").unlink()
    with pytest.raises(ValueError, match="DONE.md"):
        request(workspace, "swe")


# ---- B7 egress ----


def test_egress_scrub_strips_llm_credentials() -> None:
    env = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "sk-x",
        "ANTHROPIC_AUTH_TOKEN": "sk-y",
        "DASHSCOPE_API_KEY": "sk-z",
        "HF_TOKEN": "hf-t",
        "AWS_ACCESS_KEY_ID": "aws",
        "VLLM_PORT": "8000",
        "CEB_RUN_ID": "r1",
        "CUSTOM_BUILDER_API_KEY": "fixture-builder-token",
        "CEB_PROVIDER_SECRET": "fixture-provider-token",
        "RESEARCHER_TOKEN": "fixture-researcher-token",
    }
    scrubbed, removed = scrub_egress_env(env, "http://127.0.0.1:18900/v1")
    # the pin sets a placeholder key (EMPTY) for the local endpoint;
    # the real credential must be gone
    assert scrubbed["OPENAI_API_KEY"] == "EMPTY"
    assert "ANTHROPIC_AUTH_TOKEN" not in scrubbed
    assert "VLLM_PORT" in scrubbed
    assert "CEB_RUN_ID" in scrubbed
    secrets = {"CUSTOM_BUILDER_API_KEY", "CEB_PROVIDER_SECRET", "RESEARCHER_TOKEN"}
    assert not secrets & scrubbed.keys()
    assert set(removed) >= {"OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AWS_ACCESS_KEY_ID"}
    assert scrubbed["OPENAI_BASE_URL"] == "http://127.0.0.1:18900/v1"
    assert scrubbed["OPENAI_API_KEY"] == "EMPTY"


# ---- B8 preflight ----


def test_preflight_rejects_dirty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("brief")
    (workspace / "out").mkdir()
    with pytest.raises(PreflightError, match="prior-research"):
        check_workspace_clean(workspace, "swe")


def test_preflight_accepts_fresh_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("brief")
    result = check_workspace_clean(workspace, "swe")
    assert result["clean"] is True


def test_preflight_detects_seed_drift(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "agent.py").write_text("v1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(harness_manifest(seed)))
    (seed / "agent.py").write_text("v2\n")
    from coevobench.core.lifecycle.preflight import check_seed_harness

    with pytest.raises(PreflightError, match="drift"):
        check_seed_harness(seed, manifest)


# ---- B9 results store ----


def test_result_store_ledger_and_scoreboard(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results")
    store.record({"run_id": "r1", "bench": "swe", "role": "final", "score": 0.21})
    store.record({"run_id": "r1", "bench": "swe", "role": "final", "score": 0.19})
    store.record({"run_id": "r1", "bench": "tb", "role": "final", "score": 0.10})
    board = store.scoreboard(bench="swe")
    assert len(board) == 1
    assert board[0]["score"] == pytest.approx(0.21)  # best-of, not last
    entries = store.entries(run_id="r1", bench="tb")
    assert len(entries) == 1
    assert store.rebuild_index() == 2


# ---- B10 prompts ----


def test_prompts_are_the_single_source() -> None:
    assert "9B Base model" in TASK_BRIEF_CORE
    assert "Do not retrieve protected" in TASK_BRIEF_CORE
    assert "/workspace" in RESEARCHER_LOOP_PROMPT
    assert "DONE.md" in RESEARCHER_LOOP_PROMPT
    # the loop prompt in the shipped template derives from this module
    template = (
        Path(__file__).resolve().parents[2] / "templates" / "researcher-loop.sh"
    ).read_text()
    assert "coevobench.research.loop" in template
    from coevobench.research.loop import LoopState, prompt

    assert RESEARCHER_LOOP_PROMPT in prompt(Path("/workspace"), "tb", LoopState(), {})
