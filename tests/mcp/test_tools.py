"""MCP tools: direct handler calls over the shared function layer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from coevobench.core.lifecycle.registry import RunManifest, RunRegistry
from coevobench.mcp.server import ToolContext, make_tools


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


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    registry = RunRegistry(tmp_path)
    registry.register(
        RunManifest(
            run_id="glm53_tb_1",
            bench="tb",
            researcher_model="glm53",
            researcher_cli="claude-code",
            replicate=1,
            budget_hours=1.0,
            gpus=(0,),
            image="python:3.11-slim",
        )
    )
    return ToolContext(registry_root=tmp_path)


def test_bench_protocol_reports_frozen_constants(ctx: ToolContext) -> None:
    payload = make_tools(ctx)["bench_protocol"]()
    assert payload["splits"]["swe"]["test"] == 253
    assert payload["budget"]["gpus_per_run"] == 4
    assert "three independent" in payload["replication"]


def test_doctor_shape(ctx: ToolContext) -> None:
    payload = make_tools(ctx)["doctor"]()
    names = {c["name"] for c in payload["checks"]}
    assert "docker daemon" in names


def test_gpu_inventory_tolerates_no_gpu(ctx: ToolContext) -> None:
    payload = make_tools(ctx)["gpu_inventory"]()
    assert "gpus" in payload


def test_list_runs_and_status(ctx: ToolContext) -> None:
    tools = make_tools(ctx)
    runs = tools["list_runs"]()["runs"]
    assert [r["run_id"] for r in runs] == ["glm53_tb_1"]
    status = tools["run_status"]("glm53_tb_1")
    assert status["run_id"] == "glm53_tb_1"
    unknown = tools["run_status"]("nope")
    assert "error" in unknown


def test_run_lifecycle(ctx: ToolContext) -> None:
    from coevobench.orchestrate.supervisor import LifecycleStore, Supervisor

    registry = ctx.registry
    store = LifecycleStore(registry.root)
    Supervisor("glm53_tb_1", store, stop_fn=lambda: None).start(1.0)
    payload = make_tools(ctx)["run_lifecycle"]("glm53_tb_1")
    assert payload["phase"] == "running"
    assert payload["deadline_at"] > payload["started_at"]


def test_eval_records_reads_summaries(ctx: ToolContext) -> None:
    select_dir = ctx.registry.select_dir("glm53_tb_1")
    summary = {
        "run_id": "glm53_tb_1",
        "candidate": "cand1",
        "split": "selection",
        "score": 0.5,
        "n_bad": 0,
        "k": 1,
    }
    (select_dir / "selection_cand1_1000.jsonl").write_text(
        json.dumps(summary) + "\n"
    )
    payload = make_tools(ctx)["eval_records"]("glm53_tb_1")
    assert payload["records"][0]["candidate"] == "cand1"
    assert payload["records"][0]["score"] == 0.5


def test_report_tool_aggregates(ctx: ToolContext, tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "tb.json").write_text(
        json.dumps({"bench": "tb", "base": 0.0, "instruct": 0.5, "reported": 0.25})
    )
    payload = make_tools(ctx)["report"](str(results))
    assert payload["benches"][0]["improvement"] == pytest.approx(0.25)
    assert payload["avg"] == pytest.approx(0.25)


def test_data_fetch_unsupported(ctx: ToolContext) -> None:
    payload = make_tools(ctx)["data_fetch"]("nope")
    assert payload["status"] == "unsupported"


@pytest.mark.skipif(not _docker_available(), reason="docker daemon not available")
def test_launch_and_stop_roundtrip() -> None:
    import tempfile

    # AF_UNIX path limits (macOS): keep the registry root short
    short_root = Path(tempfile.mkdtemp(prefix="ceb-mcp-"))
    ctx = ToolContext(registry_root=short_root)
    tools = make_tools(ctx)
    # python:3.11-slim exists everywhere; the envelope image only on
    # GPU hosts. The loop entrypoint is absent from slim, so the run
    # container idles — fine for a launch/stop roundtrip.
    # launch inside the guard too: a failure before the try leaked the
    # container, the broker socket and the daemon (the lifecycle-mount
    # bug left exactly those behind on every failing run)
    info = None
    try:
        info = tools["launch_run"](
            bench="toy", budget_hours=0.02, image="python:3.11-slim"
        )
        assert info["run_id"].startswith("example_toy_")
        assert info["container"].startswith("ceb-run-")
        status = tools["run_status"](info["run_id"])
        assert status["phase"] == "running"
    finally:
        if info is not None:
            stopped = tools["stop_run"](info["run_id"])
            assert stopped["stopped"] is True
    left = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={info['container']}", "--format", "x"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert not left.stdout.strip()
