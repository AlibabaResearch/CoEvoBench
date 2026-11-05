"""Budgeted Claude Code continuation with bounded recovery and explicit submission."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from coevobench.core.grading.submission import inspect_submission
from coevobench.core.protocol.prompts import RESEARCHER_LOOP_PROMPT
from coevobench.research.progress import Progress


@dataclass
class LoopState:
    session: str = field(default_factory=lambda: str(uuid.uuid4()))
    resume: bool = False
    iteration: int = 0
    failures: int = 0
    recoveries: int = 0
    mode: str = "research"
    final_rounds: int = 0
    seen: list[str] = field(default_factory=list)
    phase: str = "running"
    reason: str = ""

    def advance(self, progress: Progress, rc: int, submission: dict) -> None:
        self.resume = True
        self.seen = progress.seen
        self.failures = 0 if progress.completed and rc == 0 else self.failures + 1
        if self.mode == "finalize":
            self.final_rounds += 1
        if submission["status"] == "confirmed":
            self.phase, self.reason = "submitted", "submission verified"
            return
        if self.final_rounds >= 3:
            self.phase, self.reason = "incomplete", "finalization incomplete"
            return
        if progress.context_error or progress.malformed or self.failures >= 3:
            self.session, self.resume = str(uuid.uuid4()), False
            self.recoveries += 1
            self.failures = 0
            self.reason = "context-error" if progress.context_error else "no-progress"
            if self.recoveries >= 3:
                if self.mode == "finalize":
                    self.phase, self.reason = "incomplete", "recovery exhausted"
                else:
                    self.mode = "finalize"


def prompt(workspace: Path, bench: str, state: LoopState, submission: dict) -> str:
    common = RESEARCHER_LOOP_PROMPT.replace("/workspace", str(workspace)) + "\n"
    if not state.resume:
        common += "This is a fresh session; earlier sessions remain archived. "
    if state.mode == "finalize":
        common += (
            f"Finalize now. Complete out/{bench}_final/ with harness/, model_path.txt, "
            "meta.json, serve_config.json and the training recipe. Describe limitations honestly. "
            "Write out/DONE.md, then run ./submit --bench " + bench + ". "
            f"The current submission check is: {json.dumps(submission)}. "
            "Do not begin new training or long evaluations."
        )
    else:
        common += "Before finishing, write out/DONE.md AND run ./submit --bench " + bench + "."
    return common


def invoke(argv: list[str], workspace: Path, deadline: float, progress: Progress,
           *, idle_seconds: float = 900, stop: threading.Event | None = None) -> int:
    stop = stop or threading.Event()
    proc = subprocess.Popen(argv, cwd=workspace, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    last_progress = time.monotonic()
    buffer = b""
    cancelled = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                if stop.is_set() or time.time() >= deadline:
                    cancelled = True
                    return 124
                if not progress.pending and time.monotonic() - last_progress >= idle_seconds:
                    cancelled = True
                    return 125
                if not selector.select(timeout=0.2):
                    if proc.poll() is not None:
                        break
                    continue
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(event, dict) and progress.observe(event):
                        last_progress = time.monotonic()
                if len(buffer) > 4 * 1024 * 1024:
                    buffer = b""
        return proc.wait(timeout=5)
    finally:
        # Background research jobs survive normal rounds. Cancellation and
        # deadlines terminate the invocation's process group.
        if cancelled or proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        proc.stdout.close()


def run_loop(workspace: Path, lifecycle: Path, bench: str, model: str,
             *, executable: str = "claude", pause_s: float = 20) -> int:
    workspace = workspace.resolve()
    output = workspace / "out"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "researcher-state.json"
    state = LoopState(**json.loads(path.read_text())) if path.is_file() else LoopState()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    def record(submission):
        data = {**asdict(state), "submission": submission, "time": time.time()}
        saved = path.with_suffix(".tmp")
        saved.write_text(json.dumps(asdict(state), indent=2))
        saved.replace(path)
        with (output / "researcher-loop.jsonl").open("a") as log:
            log.write(json.dumps(data) + "\n")
        print(f"[researcher-loop] iteration {state.iteration} {state.phase} "
              f"mode={state.mode} reason={state.reason}", file=sys.stderr, flush=True)

    while state.phase == "running" and not stop.is_set():
        budget = json.loads(lifecycle.read_text())
        deadline = budget["deadline_at"]
        submission = inspect_submission(workspace, bench)
        if submission["status"] == "confirmed":
            state.phase, state.reason = "submitted", "submission verified"
            record(submission)
            break
        if budget["phase"] != "running" or time.time() >= deadline:
            state.phase, state.reason = "incomplete", "deadline"
            record(submission)
            break
        if deadline - time.time() <= 1800 or (output / "DONE.md").exists():
            state.mode = "finalize"
        state.iteration += 1
        argv = [executable, "--print", "--verbose", "--output-format", "stream-json",
                "--model", model, "--dangerously-skip-permissions"]
        argv += ["--resume" if state.resume else "--session-id", state.session]
        argv.append(prompt(workspace, bench, state, submission))
        progress = Progress(seen=state.seen)
        record(submission)
        try:
            rc = invoke(argv, workspace, deadline, progress, stop=stop)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[researcher-loop] invocation failed: {exc}", file=sys.stderr, flush=True)
            rc = 1
        submission = inspect_submission(workspace, bench)
        state.advance(progress, rc, submission)
        record(submission)
        if state.phase == "running":
            stop.wait(min(pause_s * 2 ** min(state.failures, 3), 120,
                          max(0, deadline - time.time())))
    if stop.is_set() and state.phase == "running":
        state.phase, state.reason = "incomplete", "interrupted"
        record(inspect_submission(workspace, bench))
    return 0 if state.phase == "submitted" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        default=os.environ.get("CEB_WORKSPACE", "/workspace"))
    parser.add_argument("--lifecycle", type=Path, default=os.environ.get("CEB_LIFECYCLE"))
    parser.add_argument("--bench", default=os.environ.get("CEB_BENCH"))
    args = parser.parse_args()
    if not args.lifecycle or not args.bench:
        parser.error("the run must declare CEB_LIFECYCLE and CEB_BENCH")
    return run_loop(args.workspace, args.lifecycle, args.bench, os.environ["RESEARCHER_MODEL"])


if __name__ == "__main__":
    raise SystemExit(main())
