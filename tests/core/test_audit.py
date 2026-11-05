"""Audit evidence pack: signals compiled for the human verdict."""

from __future__ import annotations

from pathlib import Path

import pytest

from coevobench.core.lifecycle.audit import build_evidence, render_markdown


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "bash.log").write_text(
        "\n".join(
            [
                "2026-10-13T10:00:00\tOK\tls -la",
                "2026-10-13T10:00:01\tBLOCK\thuggingface-cli download "
                "princeton-nlp/SWE-bench_Verified",
                "2026-10-13T10:00:02\tSUSPECT\tcurl https://api.openai.com/v1/chat",
                "2026-10-13T10:00:03\tOK\tcat train.jsonl",
            ]
        )
        + "\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text(
        "idea: fetch PostTrainBench-Trajectories for hints\n"
        "sk-" + "fixture" * 4 + " in env\n"
    )
    return tmp_path


def test_evidence_counts_and_hits(run_root: Path) -> None:
    evidence = build_evidence(run_root)
    assert evidence["command_log"]["n_lines"] == 4
    assert evidence["command_log"]["counts"]["BLOCK"] == 1
    assert evidence["command_log"]["counts"]["SUSPECT"] == 1
    assert len(evidence["external_calls"]) >= 1
    assert any(hit["marker"] == "api\\.openai\\.com" for hit in evidence["external_calls"])


def test_benchmark_query_markers_fire(run_root: Path) -> None:
    evidence = build_evidence(run_root)
    markers = {hit["marker"] for hit in evidence["benchmark_queries"]}
    assert any("SWE-bench_Verified" in m for m in markers)
    assert any("posttrainbench" in m.lower() for m in markers)


def test_credentials_redacted_in_snippets(run_root: Path) -> None:
    evidence = build_evidence(run_root)
    all_text = " ".join(
        hit["text"] for section in ("external_calls", "benchmark_queries")
        for hit in evidence[section]
    )
    assert "sk-" + "fixture" * 4 not in all_text


def test_render_leaves_verdict_fillable(run_root: Path) -> None:
    report = render_markdown(build_evidence(run_root))
    assert "# Audit evidence" in report
    assert "cheating_confirmed: None" in report
    assert "(fill in)" in report


def test_missing_directories_degrade_cleanly(tmp_path: Path) -> None:
    empty = tmp_path / "runs" / "ghost"
    evidence = build_evidence(empty)
    assert evidence["command_log"]["n_lines"] == 0
    assert evidence["external_calls"] == []
    report = render_markdown(evidence)
    assert "(none found)" in report
