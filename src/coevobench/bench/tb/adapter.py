"""TB-Lite adapter (design document Section 4.3).

Terminal-bench grading: the verifier runs only after the agent finishes —
tests are uploaded post-rollout, so the agent never sees the assertions
before verification. The reward is the verifier's own
number from reward.txt, including partial credit; judging never reads
agent self-reports.
"""

from __future__ import annotations

from pathlib import Path

from coevobench.bench.adapter import FileBench
from coevobench.core.protocol.spec import TaskView, Verdict
from coevobench.core.protocol.splits import validate_frozen

WALL_LIMIT_S = 600
STEP_LIMIT = 120
OUTPUT_TOKENS = 5120
CONTEXT_LENGTH = 32768

VERIFY_SCRIPT = """set +e
mkdir -p /logs/verifier /tests
bash /tests/test.sh >/tmp/verify.log 2>&1
echo "verify_rc=$?"
cat /logs/verifier/reward.txt 2>/dev/null || echo "no-reward"
"""


class TbBench(FileBench):
    name = "tb"

    def __init__(self, data_root: Path, tasks: dict[str, dict] | None = None) -> None:
        super().__init__(data_root)
        self._tasks = tasks or {}
        validate_frozen(self.name, self.splits)

    def public_view(self, task_id: str) -> TaskView:
        task = self._tasks[task_id]
        return TaskView(
            task_id=task_id,
            prompt=task["instruction"],
            image=task.get("image"),
        )

    def verifier_files(self, task_id: str):
        """Files the platform uploads only after the agent finishes.

        The whole tests directory, not just test.sh: terminal-bench
        graders read label files and helper modules from next to
        test.sh, and test_outputs.py imports them off /tests. Binary
        entries (encoding=base64 in the task record) decode to bytes.
        """
        import base64

        from coevobench.runtime.sandbox.base import FileWrite

        task = self._tasks[task_id]
        files = []
        for entry in task.get("test_files") or []:
            rel = entry["path"].removeprefix("tests/")
            data = entry["data"]
            if entry.get("encoding") == "base64":
                data = base64.b64decode(data)
            files.append(
                FileWrite(path=f"/tests/{rel}", data=data, mode=0o755)
            )
        if not files and task.get("test_sh"):
            files = [FileWrite(path="/tests/test.sh", data=task["test_sh"], mode=0o755)]
        return files

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict:
        # Grader files land under /tests; the task image does not carry
        # the directory (tests are platform-uploaded), so create it
        # before the copy.
        sandbox.run("mkdir -p /tests /logs/verifier", timeout_s=30)
        for entry in self.verifier_files(task_id):
            sandbox.write_files([entry])
        run = sandbox.run(VERIFY_SCRIPT, timeout_s=600)
        text = run["text"]
        if "no-reward" in text:
            return Verdict(resolved=False, score=0.0, exit_status="VerifierFailed")
        try:
            reward_line = [ln for ln in text.splitlines() if ln.strip()][-1]
            reward = float(reward_line)
        except ValueError:
            return Verdict(resolved=False, score=0.0, exit_status="VerifierFailed")
        return Verdict(
            resolved=reward >= 1.0,
            score=reward,
            exit_status="Submitted",
        )


__all__ = ["TbBench"]
