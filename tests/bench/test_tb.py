"""TB adapter: verifier-after-rollout ordering and reward parsing."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.bench.tb.adapter import TbBench

TASK = {
    "instruction": "Install and run the cron daemon; leave it running.",
    "image": "registry.example/tb-lite:cron-01",
    "test_sh": "#!/bin/bash\npython3 /tests/check.py\n",
}


def _tb(tmp_path: Path) -> TbBench:
    ids = [f"tb-task-{i:03d}" for i in range(100)]
    (tmp_path / "tb_splits.json").write_text(
        json.dumps({"visible": ids[:20], "selection": ids[20:60], "test": ids[60:100]})
    )
    return TbBench(tmp_path, tasks={tid: TASK for tid in ids})


class VerifierSandbox:
    def __init__(self, reward: str) -> None:
        self._reward = reward
        self.uploaded: list[str] = []

    def write_files(self, files) -> None:
        self.uploaded.extend(f.path for f in files)

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> dict:
        return {"exit_code": 0, "text": f"verify_rc=0\n{self._reward}"}

    def read_file(self, path: str) -> str:
        return ""


def test_public_view_has_task_image(tmp_path: Path) -> None:
    view = _tb(tmp_path).public_view("tb-task-000")
    assert view.image == "registry.example/tb-lite:cron-01"
    assert "cron" in view.prompt


def test_tests_uploaded_only_at_judge_time(tmp_path: Path) -> None:
    bench = _tb(tmp_path)
    sandbox = VerifierSandbox("0.25")
    bench.judge("tb-task-000", "", sandbox)
    # The verifier script arrives with the judge, never with the task view:
    assert "/tests/test.sh" in sandbox.uploaded


def test_partial_credit_reward(tmp_path: Path) -> None:
    verdict = _tb(tmp_path).judge("tb-task-000", "", VerifierSandbox("0.25"))
    assert not verdict.resolved
    assert verdict.score == 0.25


def test_full_reward_resolves(tmp_path: Path) -> None:
    verdict = _tb(tmp_path).judge("tb-task-000", "", VerifierSandbox("1.0"))
    assert verdict.resolved
    assert verdict.score == 1.0


def test_missing_reward_is_verifier_failure(tmp_path: Path) -> None:
    verdict = _tb(tmp_path).judge("tb-task-000", "", VerifierSandbox("no-reward"))
    assert verdict.exit_status == "VerifierFailed"
    assert verdict.score == 0.0


def test_verifier_files_upload_whole_tests_dir(tmp_path: Path) -> None:
    task = {
        "instruction": "Rank the anomalies.",
        "image": "registry.example/tb-lite:anomaly",
        "test_sh": "#!/bin/bash\npytest /tests/test_outputs.py\n",
        "test_files": [
            {"path": "tests/test.sh", "data": "#!/bin/bash\npytest /tests/test_outputs.py\n"},
            {"path": "tests/test_outputs.py", "data": "def test_ok():\n    assert True\n"},
            {"path": "tests/labels.json", "data": "{}\n"},
        ],
        "solution_sh": "true",
    }
    ids = [f"tb-task-{i:03d}" for i in range(100)]
    (tmp_path / "tb_splits.json").write_text(
        json.dumps({"visible": ids[:20], "selection": ids[20:60], "test": ids[60:100]})
    )
    bench = TbBench(tmp_path, tasks={tid: task for tid in ids})
    files = bench.verifier_files(ids[0])
    paths = [f.path for f in files]
    assert paths == ["/tests/test.sh", "/tests/test_outputs.py", "/tests/labels.json"]
    assert files[0].data.startswith("#!/bin/bash")
