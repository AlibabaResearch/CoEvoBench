"""Continue one Codex researcher session until submission or the persisted deadline."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run_loop(command: list[str], workspace: Path, lifecycle: Path, *, pause_s=20):
    if command[:2] != ["codex", "exec"]:
        raise ValueError("the researcher loop requires a codex exec command")
    resume = False
    while True:
        state = json.loads(lifecycle.read_text())
        remaining = state["deadline_at"] - time.time()
        if remaining <= 0 or state.get("phase", "running") != "running":
            return
        if (workspace / "out/DONE.md").is_file() and (
            workspace / "out/appworld_final/submission.json"
        ).is_file():
            return
        argv = command[:2] + (["resume", "--last"] if resume else []) + command[2:]
        proc = subprocess.Popen(
            argv, cwd=workspace, stdin=subprocess.DEVNULL, start_new_session=True
        )
        try:
            code = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return
        if code == 0:
            resume = True
        time.sleep(min(pause_s, max(0, state["deadline_at"] - time.time())))


if __name__ == "__main__":
    run_loop(
        sys.argv[1:],
        Path(os.environ.get("CEB_WORKSPACE", "/workspace")),
        Path(os.environ["CEB_LIFECYCLE"]),
    )
