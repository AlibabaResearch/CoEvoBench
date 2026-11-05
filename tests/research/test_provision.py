"""Workspace provisioning stamps the Appendix A contract."""

from __future__ import annotations

from pathlib import Path

from coevobench.orchestrate.provision import provision_workspace


def test_toy_workspace_gets_task_timer_eval(tmp_path: Path) -> None:
    landed = provision_workspace(tmp_path, "toy")
    assert (tmp_path / "TASK.md").is_file()
    assert (tmp_path / "timer.sh").is_file()
    assert (tmp_path / "eval").is_file()
    assert landed["TASK.md"] == 1
    # frozen entries are read-only for the researcher
    import stat

    mode = stat.S_IMODE((tmp_path / "eval").stat().st_mode)
    assert not (mode & stat.S_IWUSR)


def test_swe_workspace_gets_harness_and_pool(tmp_path: Path) -> None:
    landed = provision_workspace(tmp_path, "swe")
    harness = tmp_path / "harness"
    assert harness.is_dir() and any(harness.rglob("_transport.py"))
    assert (tmp_path / "TASK.md").is_file()
    assert landed["harness"] > 10


def test_provision_is_idempotent(tmp_path: Path) -> None:
    provision_workspace(tmp_path, "toy", serve_config={"tensor_parallel": 2})
    (tmp_path / "TASK.md").write_text("researcher edits")
    (tmp_path / "serve_config.json").write_text('{"max_len":8192}')
    provision_workspace(tmp_path, "toy", serve_config={"max_len":32768})
    # second provisioning must not clobber researcher work
    assert (tmp_path / "TASK.md").read_text() == "researcher edits"
    assert (tmp_path / "serve_config.json").read_text() == '{"max_len":8192}'


def test_unknown_bench_lands_only_the_scaffold(tmp_path: Path) -> None:
    provision_workspace(tmp_path, "nope")
    # No template exists; only the bench-independent scaffold lands.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "tasks_visible"]


def test_toy_workspace_gets_materialized_public_views(tmp_path: Path) -> None:
    import json

    provision_workspace(tmp_path, "toy")
    views = sorted((tmp_path / "tasks_visible" / "toy").glob("*.json"))
    assert [v.name for v in views] == ["toy-sum-1.json", "toy-sum-2.json"]
    payload = json.loads(views[0].read_text())
    assert payload["task_id"] == "toy-sum-1"
    assert "sum of squares" in payload["prompt"]
    assert "answer" in payload["prompt"]


def test_lifecycle_store_makes_state_world_readable(tmp_path: Path) -> None:
    import os
    import time

    from coevobench.orchestrate.supervisor import LifecycleState, LifecycleStore

    store = LifecycleStore(tmp_path)
    state = LifecycleState(run_id="r1", started_at=time.time(), deadline_at=time.time() + 3600)
    store.save(state)
    mode = os.stat(tmp_path / "r1" / "lifecycle.json").st_mode
    assert mode & 0o004, "in-envelope timer (uid 1000) must be able to read it"
