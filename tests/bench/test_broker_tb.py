"""Broker TB judge: task loading and the task_id mirror fast path."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.config.schema import Config, EvalImageMirror
from coevobench.research.server import Broker


def _config_with_tb_mirror() -> Config:
    cfg = Config()
    cfg.eval_images.mirrors = [
        EvalImageMirror(
            prefix="swebench/",
            rewrite="cr.example.com/swebench-verified:",
            tag_source="instance_id",
        ),
        EvalImageMirror(
            prefix="tb/",
            rewrite="cr.example.com/tb-lite:",
            tag_source="task_id",
        ),
    ]
    return cfg


class _FakeSandbox:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.killed = False

    def write_files(self, files) -> None:
        self.written.extend(f.path for f in files)

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None):
        return {"exit_code": 0, "text": "verify_rc=0\n1.0\n"}

    def kill(self) -> None:
        self.killed = True


class _RecordingRuntime:
    def __init__(self) -> None:
        self.specs = []
        self.sandbox = _FakeSandbox()

    def create(self, spec, ready_timeout_s: int | None = None):
        self.specs.append(spec)
        return self.sandbox


def _task_row(task_id: str, image: str | None) -> dict:
    return {
        "task_id": task_id,
        "instruction": "do it",
        "test_sh": "#!/bin/bash\ntrue\n",
        "test_files": [{"path": "tests/test.sh", "data": "#!/bin/bash\ntrue\n"}],
        "solution_sh": "true\n",
        "image": image,
    }


def _workspace_with_tb_splits(tmp_path: Path) -> Path:
    from coevobench.cli.data import frozen_splits

    frozen = frozen_splits("tb")
    root = tmp_path / "ws"
    (root / "data" / "tb").mkdir(parents=True)
    (root / "data" / "tb" / "tb_splits.json").write_text(
        json.dumps(
            {
                "visible": list(frozen.visible),
                "selection": list(frozen.selection),
                "test": list(frozen.test),
            }
        )
    )
    return root


def test_tb_judge_resolves_image_through_task_id_mirror(tmp_path, monkeypatch) -> None:
    tasks = {f"tb-{i:03d}": _task_row(f"tb-{i:03d}", None) for i in range(100)}
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )
    broker = Broker(
        runtime=_RecordingRuntime(),
        config=_config_with_tb_mirror(),
        workspace_root=_workspace_with_tb_splits(tmp_path),
    )
    judge = broker._judge_for("tb", {})
    verdict = judge("tb-000")
    assert verdict.resolved
    spec = broker.runtime.specs[0]
    assert spec.image == "cr.example.com/tb-lite:tb-000"


def test_tb_judge_prefers_materialized_image_over_mirror(tmp_path, monkeypatch) -> None:
    tasks = {f"tb-{i:03d}": _task_row(f"tb-{i:03d}", f"my-reg:tb-{i:03d}") for i in range(100)}
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )
    broker = Broker(
        runtime=_RecordingRuntime(),
        config=_config_with_tb_mirror(),
        workspace_root=_workspace_with_tb_splits(tmp_path),
    )
    judge = broker._judge_for("tb", {})
    judge("tb-001")
    spec = broker.runtime.specs[0]
    assert spec.image == "my-reg:tb-001"


def test_tb_judge_without_task_data_fails(tmp_path, monkeypatch) -> None:
    tasks = {f"tb-{i:03d}": _task_row(f"tb-{i:03d}", None) for i in range(100)}
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )
    broker = Broker(
        runtime=_RecordingRuntime(),
        config=Config(),
        workspace_root=_workspace_with_tb_splits(tmp_path),
    )
    judge = broker._judge_for("tb", {})
    verdict = judge("tb-000")
    assert verdict.exit_status == "EvalImageMissing"
    assert broker.runtime.specs == []


def test_tb_judge_missing_task_id_reports_task_data_missing(
    tmp_path, monkeypatch
) -> None:
    tasks = {f"tb-{i:03d}": _task_row(f"tb-{i:03d}", "reg:x") for i in range(100)}
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )
    broker = Broker(
        runtime=_RecordingRuntime(),
        config=Config(),
        workspace_root=_workspace_with_tb_splits(tmp_path),
    )
    judge = broker._judge_for("tb", {})
    verdict = judge("not-a-task")
    assert verdict.exit_status == "TaskDataMissing"


def test_tb_judge_uploads_whole_tests_dir(tmp_path, monkeypatch) -> None:
    task = _task_row("tb-000", "reg:x")
    task["test_files"] = [
        {"path": "tests/test.sh", "data": "#!/bin/bash\ntrue\n"},
        {"path": "tests/test_outputs.py", "data": "def test_ok():\n    assert True\n"},
    ]
    tasks = {
        f"tb-{i:03d}": (task if i == 0 else _task_row(f"tb-{i:03d}", "reg:x"))
        for i in range(100)
    }
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )
    broker = Broker(
        runtime=_RecordingRuntime(),
        config=Config(),
        workspace_root=_workspace_with_tb_splits(tmp_path),
    )
    judge = broker._judge_for("tb", {})
    judge("tb-000")
    assert broker.runtime.sandbox.written == [
        "/tests/test.sh",
        "/tests/test_outputs.py",
    ]


def test_tb_judge_sandbox_carries_network(tmp_path, monkeypatch) -> None:
    """Terminal-bench graders install their venv at runtime."""
    tasks = {f"tb-{i:03d}": _task_row(f"tb-{i:03d}", "reg:x") for i in range(100)}
    monkeypatch.setattr(
        "coevobench.bench.tb.tasks.load_tb_tasks", lambda: tasks
    )
    broker = Broker(
        runtime=_RecordingRuntime(),
        config=Config(),
        workspace_root=_workspace_with_tb_splits(tmp_path),
    )
    broker._judge_for("tb", {})("tb-000")
    assert broker.runtime.specs[0].network is True
