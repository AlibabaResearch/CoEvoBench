"""Broker-backed SandboxRuntime: the in-envelope client (design doc Section 3.2).

Same SPI as DockerSandboxRuntime, but every operation is forwarded to the
host-side broker over the mounted unix socket. This is what the dev-eval
engine and the workspace CLIs use inside the run container; it is also the
credential-isolation boundary: the container never touches docker itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from coevobench.runtime.sandbox.base import (
    ExecResult,
    FileWrite,
    Sandbox,
    SandboxSpec,
)

SOCKET_PATH = "/broker.sock"


class BrokerSandbox:
    def __init__(self, sandbox_id: str, socket_path: str) -> None:
        self.id = sandbox_id
        self._socket = socket_path

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> ExecResult:
        from coevobench.runtime.broker_wire import call

        full = command if cwd is None else f"cd {cwd} && {command}"
        return dict(
            call(
                self._socket,
                {
                    "op": "sandbox.exec",
                    "sandbox_id": self.id,
                    "command": full,
                    "timeout_s": timeout_s,
                },
                timeout=timeout_s + 60,
            )
        )

    def write_files(self, files: Iterable[FileWrite]) -> None:
        from coevobench.runtime.broker_wire import call

        payload = [
            {"path": f.path, "data": f.data, "mode": f.mode} for f in files
        ]
        call(self._socket, {"op": "sandbox.write", "sandbox_id": self.id, "files": payload})

    def read_file(self, path: str) -> str:
        from coevobench.runtime.broker_wire import call

        response = call(
            self._socket, {"op": "sandbox.read", "sandbox_id": self.id, "path": path}
        )
        return response["text"]

    def kill(self) -> None:
        from coevobench.runtime.broker_wire import call

        call(self._socket, {"op": "sandbox.kill", "sandbox_id": self.id})


class BrokerSandboxRuntime:
    name = "broker"

    def __init__(self, socket_path: str = SOCKET_PATH) -> None:
        self._socket = socket_path

    def create(self, spec: SandboxSpec, *, ready_timeout_s: int = 120) -> Sandbox:
        from coevobench.runtime.broker_wire import call

        response = call(
            self._socket,
            {
                "op": "sandbox.create",
                "image": spec.image,
                "cpu": spec.cpu,
                "mem_gib": spec.mem_gib,
                "ttl_s": spec.ttl_s,
            },
            timeout=ready_timeout_s + 60,
        )
        return BrokerSandbox(response["sandbox_id"], self._socket)

    def create_from_context(
        self, ctx_dir: Path, spec: SandboxSpec, *, ready_timeout_s: int = 120,
        build_timeout_s: int = 1800,
    ) -> Sandbox:
        from coevobench.runtime.broker_wire import call

        response = call(
            self._socket,
            {
                "op": "sandbox.create",
                "from_dir": str(ctx_dir),
                "cpu": spec.cpu,
                "mem_gib": spec.mem_gib,
                "ttl_s": spec.ttl_s,
                "ready_timeout_s": ready_timeout_s,
                "build_timeout_s": build_timeout_s,
            },
            timeout=build_timeout_s + ready_timeout_s + 120,
        )
        return BrokerSandbox(response["sandbox_id"], self._socket)

    def connect(self, sandbox_id: str) -> Sandbox:
        return BrokerSandbox(sandbox_id, self._socket)


__all__ = ["BrokerSandbox", "BrokerSandboxRuntime", "SOCKET_PATH"]
