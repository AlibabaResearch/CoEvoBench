"""Reattach lifecycle operations to persisted run handles."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

from coevobench.runtime.docker_cleanup import remove_container
from coevobench.runtime.run_resources import process_start


def status(provider: str, identity: str, start: str | None = None) -> str:
    if provider == "docker":
        result = subprocess.run(["docker", "inspect", identity, "--format", "{{.State.Status}}"],
                                capture_output=True, text=True, timeout=30)
        if result.returncode and "No such" not in result.stderr and "no such" not in result.stderr:
            raise RuntimeError(result.stderr[-300:])
        return result.stdout.strip() if result.returncode == 0 else "missing"
    if provider == "local":
        current = process_start(int(identity))
        return "running" if current and (start is None or current == start) else "exited"
    if provider == "modal":
        from coevobench.runtime.modal_util import require_modal

        return "running" if require_modal().Sandbox.from_id(identity).poll() is None else "exited"
    if provider == "daytona":
        from coevobench.runtime.sandbox.daytona import client

        box = client().get(identity)
        return "running" if str(box.state).lower().endswith("started") else "exited"
    raise ValueError(f"unknown runhost: {provider}")


def stop(provider: str, identity: str, start: str | None = None) -> None:
    if provider == "docker":
        if status(provider, identity) not in ("missing", "exited", "dead"):
            subprocess.run(["docker", "stop", "--timeout", "15", identity],
                           check=True, capture_output=True, timeout=40)
    elif provider == "local":
        if status(provider, identity, start) == "running":
            pid = int(identity)
            original = process_start(pid)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
            time.sleep(0.2)
            current = process_start(pid)
            if not current or current == original:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
    elif provider == "modal":
        from coevobench.runtime.modal_util import require_modal

        require_modal().Sandbox.from_id(identity).terminate(wait=True)
    elif provider == "daytona":
        from coevobench.runtime.sandbox.daytona import client

        api = client()
        api.stop(api.get(identity))
    else:
        raise ValueError(f"unknown runhost: {provider}")


def archive(provider: str, identity: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if provider == "docker" and status(provider, identity) != "missing":
        log = subprocess.run(["docker", "logs", identity], capture_output=True, timeout=120)
        (destination / "container.log").write_bytes(log.stdout + log.stderr)
        for path, name in ((".claude/projects", "claude-projects"),
                           (".codex/sessions", "codex-sessions")):
            subprocess.run(
                ["docker", "cp", f"{identity}:/home/researcher/{path}", str(destination / name)],
                capture_output=True, timeout=120,
            )


def remove(provider: str, identity: str) -> None:
    if provider == "docker":
        remove_container(identity)
    elif provider == "daytona":
        from coevobench.runtime.sandbox.daytona import client

        api = client()
        api.delete(api.get(identity), wait=True)
