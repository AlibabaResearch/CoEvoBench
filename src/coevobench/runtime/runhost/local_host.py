"""Local process RunHost: the researcher runs directly on the operator
machine, no container.

The researcher uses a separate HOME and the run's original deadline.
Training and serving use local processes; sandbox operations use the broker.

Isolation is cooperative, not structural: workspace-scoped hooks and
a scrubbed environment instead of a container boundary. That is the
honest trade — this host exists for operators who want the paper
protocol on a single machine without docker-in-docker nesting, or
whose command sandboxes run on another provider.

What launch() does:
- provision the workspace into a run directory (~/.ceb/runs/<id>)
- start the broker for the run (unix socket next to the workspace)
- write the lifecycle state (the budget the loop checks)
- spawn the researcher loop as a detached local process group,
  logging to the run directory
- return a handle backed by the process pid and the run directory
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

from coevobench.runtime.runhost.base import (
    Entrypoint,
    HostCapabilities,
    RunEnvelope,
    RunHandle,
)

LOOP_SCRIPT = "/opt/researcher-loop.sh"


class LocalProcessHandle:
    def __init__(self, proc: subprocess.Popen, run_dir: Path) -> None:
        self._proc = proc
        self._run_dir = run_dir

    @property
    def id(self) -> str:
        return str(self._proc.pid)

    def status(self) -> str:
        """poll() reports exit even before the zombie is reaped."""
        return "running" if self._proc.poll() is None else "exited"

    def logs(self, *, tail: int = 200) -> str:
        log = self._run_dir / "loop.log"
        if not log.is_file():
            return ""
        lines = log.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    def exec(self, command: str, *, timeout_s: int = 60) -> dict:
        proc = subprocess.run(
            ["sh", "-c", command],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(self._run_dir / "workspace"),
            start_new_session=True,
        )
        return {"exit_code": proc.returncode, "text": proc.stdout}

    def fetch(self, remote: str, local_path: str) -> None:
        src = self._run_dir / "workspace" / remote.lstrip("/")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            Path(local_path).write_bytes(src.read_bytes())

    def stop(self) -> None:
        """Terminate the process group (the loop spawns children)."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            time.sleep(1)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)


class LocalRunHost:
    """Runs the researcher as a local process on the operator machine.

    The workspace is a plain directory, sandboxes (however provided) are
    driven through the run's broker. Cooperative isolation only.
    """

    name = "local"
    capabilities = HostCapabilities(
        persistent_workspace=True,
        gpu_passthrough=True,   # local GPUs, directly
        network_endpoints=True,
        kernel_isolated=False,  # same kernel as the operator — stated honestly
    )

    def launch(
        self,
        envelope: RunEnvelope,
        entrypoint: Entrypoint,
        workspace_dir: str,
        *,
        base_dir: str | None = None,
        broker_socket_host_path: str | None = None,
    ) -> RunHandle:
        run_dir = Path(workspace_dir).parent
        run_dir.mkdir(parents=True, exist_ok=True)

        merged_env = {**envelope.env, **entrypoint.env}
        env = dict(os.environ)
        env.update({k: os.path.expandvars(v) for k, v in merged_env.items()})
        env.update(CEB_WORKSPACE=str(Path(workspace_dir).resolve()),
                   CEB_LIFECYCLE=str((run_dir / "control/lifecycle.json").resolve()))
        home = run_dir / "researcher-home"
        home.mkdir(exist_ok=True)
        env["HOME"] = str(home.resolve())
        if base_dir:
            with contextlib.suppress(FileExistsError):
                (run_dir / "workspace" / "base").symlink_to(
                    Path(base_dir).resolve(), target_is_directory=True
                )
        if broker_socket_host_path:
            env.setdefault("CEB_BROKER_SOCKET", broker_socket_host_path)

        log_path = run_dir / "loop.log"
        argv = [str(a) for a in entrypoint.argv] or ["bash", LOOP_SCRIPT]
        if LOOP_SCRIPT in argv:
            import sys

            argv = [sys.executable, "-m", "coevobench.research.loop"]
        with log_path.open("ab") as log:
            proc = subprocess.Popen(
                argv,
                env=env,
                cwd=str(run_dir / "workspace"),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return LocalProcessHandle(proc, run_dir)


__all__ = ["LocalProcessHandle", "LocalRunHost"]
