"""Run lifecycle supervisor (design document Section 5.7).

The supervisor is a long-lived process that owns `lifecycle.json` for each
run: deadline, phase, and reconciliation events. Budget is accounted from
the persisted started_at timestamp, never from container uptime, so a host
reboot (or a docker restart) cannot gift compute to a run.

Scenarios covered:
- deadline: graceful stop, default nomination capture, ordered teardown
- early stop: researcher exits or writes DONE.md
- crash: reconcile from persisted state; containers without lifecycle state
  are killed (orphan prevention)
- hung researcher: budget keeps burning (paper: the launcher sets the deadline)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PHASE_RUNNING = "running"
PHASE_STOPPING = "stopping"
PHASE_DONE = "done"


@dataclass
class LifecycleState:
    run_id: str
    started_at: float
    deadline_at: float
    phase: str = PHASE_RUNNING
    events: list[dict] = field(default_factory=list)

    def budget_left_hours(self) -> float:
        return max(0.0, (self.deadline_at - time.time()) / 3600.0)

    def record(self, kind: str, detail: str = "") -> None:
        self.events.append({"time": time.time(), "kind": kind, "detail": detail})


class LifecycleStore:
    """Atomic JSON persistence for LifecycleState under runs/<id>/."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, run_id: str) -> Path:
        return self.root / run_id / "lifecycle.json"

    def save(self, state: LifecycleState) -> None:
        target = self.path_for(state.run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2))
        tmp.replace(target)
        # The in-envelope timer (uid 1000) reads this file read-only
        # through CEB_LIFECYCLE; world-read keeps ./timer.sh working
        # without exposing anything (the document is deadline/phase only).
        target.chmod(0o644)
        # A directory bind sees atomic replacements; binding lifecycle.json
        # itself would leave a container reading the original inode forever.
        control = target.parent / "control"
        control.mkdir(exist_ok=True)
        public = control / "lifecycle.tmp"
        public.write_text(json.dumps({key: getattr(state, key) for key in (
            "run_id", "started_at", "deadline_at", "phase",
        )}))
        public.chmod(0o644)
        public.replace(control / "lifecycle.json")

    def load(self, run_id: str) -> LifecycleState | None:
        path = self.path_for(run_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return LifecycleState(**data)


class Supervisor:
    """Owns one run's lifecycle. Survives operator shells (daemonize caller)."""

    def __init__(self, run_id: str, store: LifecycleStore, stop_fn) -> None:
        self.run_id = run_id
        self.store = store
        self._stop_fn = stop_fn
        self.state = store.load(run_id)

    def rebind(self, stop_fn) -> None:
        """Attach the real stop function once the run handle exists.

        The lifecycle starts before the container does (the envelope
        mounts the record); the stop hook binds after launch.
        """
        self._stop_fn = stop_fn

    def start(self, budget_hours: float) -> LifecycleState:
        if self.state is not None:
            return self.state
        now = time.time()
        self.state = LifecycleState(
            run_id=self.run_id,
            started_at=now,
            deadline_at=now + budget_hours * 3600.0,
        )
        self.state.record("start")
        self.store.save(self.state)
        return self.state

    def reconcile(self, container_running: bool) -> str:
        """Crash recovery: the deadline follows persisted state, not uptime."""
        state = self.state
        if state is None:
            return "no-state"
        if state.phase != PHASE_RUNNING:
            return state.phase
        if time.time() >= state.deadline_at:
            self.shutdown("deadline-reached")
            return PHASE_DONE
        if not container_running:
            state.record("container-gone", "budget keeps burning until deadline")
            self.store.save(state)
        return PHASE_RUNNING

    def shutdown(self, reason: str) -> None:
        state = self.state
        if state is None or state.phase == PHASE_DONE:
            return
        state.phase = PHASE_STOPPING
        state.record("shutdown", reason)
        self.store.save(state)
        try:
            self._stop_fn()
        finally:
            state.phase = PHASE_DONE
            state.record("stopped")
            self.store.save(state)


def kill_orphan_containers(prefix: str = "ceb-run-") -> list[str]:
    """Kill run containers without lifecycle state (orphan prevention).

    Intended for operator-directed cleanup of failed launches.
    """
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    killed: list[str] = []
    for name in result.stdout.split():
        if not name.startswith(prefix):
            continue
        subprocess.run(
            ["docker", "rm", "-f", name], capture_output=True, timeout=60
        )
        killed.append(name)
    return killed


def daemonize() -> None:
    """Detach from the calling shell: new session, stdio to /dev/null.

    The containerized answer to the old `setsid nohup ... < /dev/null` rule.
    """
    if os.getpgrp() == os.getpid():
        os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    if devnull > 2:
        os.close(devnull)


__all__ = [
    "LifecycleState",
    "LifecycleStore",
    "PHASE_DONE",
    "PHASE_RUNNING",
    "PHASE_STOPPING",
    "Supervisor",
    "daemonize",
    "kill_orphan_containers",
]
