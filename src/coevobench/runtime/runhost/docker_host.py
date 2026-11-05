"""Docker-backed RunHost: one long-lived container per development run.

Design decisions (Section 2.1.1 of the design document):
- No docker socket is mounted into the run container. Every docker need
  inside the run (judge sandboxes, rollouts, image builds) goes through the
  broker socket, which is a protocol-level boundary.
- Workspace is a bind mount (operator-side, readable by ceb web/audit).
- Base weights mount read-only; the 0824-class "task said read-only but the
  directory was writable" incident becomes structurally impossible.
- CPU/memory are cgroup-limited per run (replaces the old hand-rolled CPU
  partitioning); the container outlives any operator shell (the setsid/nohup
  lesson, containerized).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from coevobench.runtime.runhost.base import (
    Entrypoint,
    HostCapabilities,
    RunEnvelope,
    RunHandle,
)
from coevobench.runtime.sandbox.docker import _run_cli


class DockerRunHandle:
    def __init__(self, container_id: str) -> None:
        self.id = container_id

    def status(self) -> str:
        result = _run_cli(
            ["inspect", "-f", "{{.State.Status}}", self.id], check=False
        )
        return result.stdout.strip() or "unknown"

    def logs(self, *, tail: int = 200) -> str:
        result = _run_cli(
            ["logs", "--tail", str(tail), self.id], check=False
        )
        return result.stdout

    def exec(self, command: str, *, timeout_s: int = 60) -> dict:
        result = _run_cli(
            ["exec", self.id, "sh", "-lc", command], timeout=timeout_s, check=False
        )
        return {"exit_code": result.returncode, "text": result.stdout}

    def fetch(self, remote: str, local_path: str) -> None:
        _run_cli(["cp", f"{self.id}:{remote}", local_path], timeout=300)

    def stop(self) -> None:
        _run_cli(["rm", "-f", self.id], timeout=120, check=False)


class DockerRunHost:
    name = "docker"
    capabilities = HostCapabilities(
        persistent_workspace=True,
        gpu_passthrough=True,
        network_endpoints=True,
        kernel_isolated=False,
    )

    def __init__(self, *, gpus_all: tuple[int, ...] = ()) -> None:
        self._gpus_all = gpus_all

    @staticmethod
    def _expand(value: str) -> str:
        """Resolve $VAR references from the host environment.

        Adapters declare secret references ("$RESEARCHER_TOKEN") so
        tokens never appear in adapter code or logs; docker -e values
        are literal, so the host resolves them here.
        """
        import os
        import re

        def resolve(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), "")

        return re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", resolve, value)

    def launch(
        self,
        envelope: RunEnvelope,
        entrypoint: Entrypoint,
        workspace_dir: str,
        *,
        base_dir: str | None = None,
        broker_socket_host_path: str | None = None,
    ) -> RunHandle:
        container = f"ceb-run-{uuid.uuid4().hex[:12]}"
        bind_root = Path(workspace_dir)
        import os

        for path in [bind_root, *sorted(bind_root.rglob("*"))]:
            try:
                os.chown(path, 1000, 1000, follow_symlinks=False)
            except PermissionError:
                # unprivileged operators: fall back to world-writable
                if not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o200)
        argv: list[str] = [
            "run",
            "-d",
            "--init",
            "--name",
            container,
            "--cpus",
            "8",
            "--memory",
            "64g",
            "--shm-size",
            "1g",
        ]
        if envelope.gpus.device_ids:
            argv += ["--gpus", f'"device={",".join(map(str, envelope.gpus.device_ids))}"']
        elif envelope.gpus.count and envelope.gpus.profile:
            argv += ["--gpus", f'"count={envelope.gpus.count}"']
        for key, value in {**envelope.env, **entrypoint.env}.items():
            expanded = self._expand(value)
            argv += ["-e", f"{key}={expanded}"]
        argv += ["-v", f"{workspace_dir}:/workspace"]
        # the lifecycle record rides the workspace mount: readable by the
        # researcher (budget awareness is their contract), at a path that
        # exists inside the container
        control = Path(workspace_dir).parent / "control"
        control.mkdir(exist_ok=True)
        argv += [
            "-v", f"{control}:/run/ceb:ro",
            "-e", "CEB_LIFECYCLE=/run/ceb/lifecycle.json",
            "-e", "CEB_WORKSPACE=/workspace",
        ]
        if base_dir:
            argv += ["-v", f"{base_dir}:/base:ro"]
        if broker_socket_host_path:
            argv += ["-v", f"{broker_socket_host_path}:/broker.sock"]
        argv += [envelope.image, *entrypoint.argv]
        handle = DockerRunHandle(container)
        try:
            _run_cli(argv, timeout=300)
            ready = handle.exec("test -d /workspace", timeout_s=30)
            if ready["exit_code"] != 0:
                raise RuntimeError(f"run container {container} failed to start")
        except BaseException:
            handle.stop()
            raise
        return handle


def entrypoint_from_json(payload: str) -> Entrypoint:
    data = json.loads(payload)
    return Entrypoint(argv=data["argv"], env=data.get("env", {}))


__all__ = ["DockerRunHandle", "DockerRunHost", "entrypoint_from_json"]
