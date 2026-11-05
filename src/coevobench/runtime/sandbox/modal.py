"""Modal-backed SandboxRuntime: sandboxes on the Modal cloud.

The same Sandbox SPI as the docker implementation, executed remotely.
Two deployments use it (design Section 3.8):

- the remote run profile: the run envelope's in-run sandboxes (judge
  sandboxes, rollouts) run co-located in Modal — env-ops do not cross
  back to the operator host;
- any operator-side use where cloud capacity beats local docker (the
  combo matrix: local serve + Modal judge sandboxes works because the
  judge path is one-directional — the sandbox never calls back).

Semantics follow the SPI contract exactly: `run` is a fresh shell per
call (cwd does not persist), the sandbox persists across calls,
timeouts terminate the whole command (`timeout` wrapper; its 124
expiry maps to the SPI's -9), `kill` is idempotent, and FileWrite
modes apply with real octal semantics.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from coevobench.runtime.modal_util import quote, read_stream, require_modal
from coevobench.runtime.sandbox.base import (
    ExecResult,
    FileWrite,
    Sandbox,
    SandboxSpec,
)

APP_NAME = "coevobench-sandboxes"
_KEEPALIVE = ("sh", "-c", "sleep infinity")


class ModalSandbox:
    """One Modal sandbox behind the Sandbox protocol."""

    def __init__(self, sandbox, *, owner_dir: Path | None = None) -> None:
        self.id = sandbox.object_id
        self._sandbox = sandbox
        self._owner_dir = owner_dir
        self._closed = False

    def endpoint(self, port: int) -> str:
        """The public http:// URL for a port declared in the spec.

        Only ports listed in SandboxSpec.ports are forwarded (Modal
        tunnels); the tunnel exists for the sandbox's lifetime.
        """
        tunnels = self._sandbox.tunnels()
        tunnel = tunnels.get(port)
        if tunnel is None:
            raise RuntimeError(
                f"no tunnel for port {port}; declare it in SandboxSpec.ports"
            )
        host, tunnel_port = tunnel.tcp_socket
        return f"http://{host}:{tunnel_port}"

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> ExecResult:
        # `timeout` reproduces the SPI's whole-tree termination; 124 is
        # its own expiry marker and maps to the docker impl's -9.
        wrapped = f"timeout {int(timeout_s)} sh -c {quote(command)}"
        if cwd:
            wrapped = f"cd {quote(cwd)} && {wrapped}"
        proc = self._sandbox.exec("sh", "-c", wrapped, timeout=int(timeout_s) + 30)
        stdout = read_stream(proc.stdout)
        proc.wait()
        rc = -9 if proc.returncode == 124 else proc.returncode
        return {"exit_code": rc, "text": stdout}

    def write_files(self, files: Iterable[FileWrite]) -> None:
        for f in files:
            data = f.data.encode() if isinstance(f.data, str) else f.data
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                local = Path(tmp.name)
            try:
                self._sandbox.filesystem.copy_from_local(str(local), f.path)
            finally:
                local.unlink(missing_ok=True)
            octal = oct(f.mode)[2:]
            proc = self._sandbox.exec("sh", "-c", f"chmod {octal} {quote(f.path)}", timeout=30)
            proc.wait()
            if proc.returncode:
                raise RuntimeError("Modal file permission update failed")

    def read_file(self, path: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "download"
            self._sandbox.filesystem.copy_to_local(path, str(dest))
            return dest.read_text(errors="replace")

    def write_bytes(self, path: str, data: bytes) -> None:
        self.write_files([FileWrite(path, data)])

    def read_bytes(self, path: str) -> bytes:
        return self._sandbox.filesystem.read_bytes(path)

    def kill(self) -> None:
        from coevobench.runtime.run_resources import release

        if not self._closed:
            try:
                self._sandbox.terminate(wait=True)
            except Exception as exc:
                if type(exc).__name__ != "NotFoundError":
                    raise
            release(self._owner_dir, "modal", self.id)
            self._closed = True


class ModalSandboxRuntime:
    name = "modal"

    def __init__(self, *, registry_secret: str | None = None,
                 owner_dir: Path | None = None, platform=None) -> None:
        self._registry_secret = registry_secret
        self._owner_dir = owner_dir
        self._platform = platform

    def create(self, spec: SandboxSpec, *, ready_timeout_s: int = 120) -> Sandbox:
        if self._platform is not None or spec.image.startswith("modal:"):
            if self._platform is None:
                from coevobench.runtime.modal_platform import ModalPlatform

                self._platform = ModalPlatform()
            image = self._platform.image(spec.image)
            return self._spawn(self._platform.sdk, image, spec, ready_timeout_s)
        modal = require_modal()
        secret = (
            modal.Secret.from_name(self._registry_secret)
            if self._registry_secret
            else None
        )
        image = modal.Image.from_registry(spec.image, secret=secret)
        return self._spawn(modal, image, spec, ready_timeout_s)

    def create_from_context(
        self, ctx_dir: Path, spec: SandboxSpec, *, ready_timeout_s: int = 120,
        build_timeout_s: int = 1800,
    ) -> Sandbox:
        """Build the context inside Modal and run the sandbox on it.

        from_dockerfile uploads the local context to Modal's builder;
        the image stays in Modal's image system for the sandboxes that
        run it — no registry, no push. Standard COPY semantics hold:
        files land at build time (later RUN steps read them),
        .dockerignore applies, and the context directory may sit apart
        from the Dockerfile.
        """
        from coevobench.runtime.sandbox.base import validate_context

        ctx = Path(ctx_dir)
        validate_context(ctx)
        modal = self._platform.sdk if self._platform else require_modal()
        image = modal.Image.from_dockerfile(
            str(ctx / "Dockerfile"), context_dir=str(ctx)
        )
        return self._spawn(modal, image, spec, ready_timeout_s)

    def _spawn(self, modal, image, spec: SandboxSpec, ready_timeout_s: int) -> Sandbox:
        from coevobench.runtime.run_resources import check_active, register

        check_active(self._owner_dir)
        app = (self._platform.app if self._platform
               else modal.App.lookup(APP_NAME, create_if_missing=True))
        kwargs: dict = {
            "image": image,
            "app": app,
            "timeout": max(spec.ttl_s, 60),
            "idle_timeout": spec.ttl_s,
            "cpu": spec.cpu,
            "memory": int(spec.mem_gib * 1024),
            "block_network": not spec.network,
        }
        if self._platform:
            kwargs.update(app=self._platform.app, client=self._platform.client)
        if spec.ports:
            # Forwarded through public tunnels for the sandbox's
            # lifetime; see ModalSandbox.endpoint.
            kwargs["unencrypted_ports"] = list(spec.ports)
        sandbox = modal.Sandbox.create(*_KEEPALIVE, **kwargs)
        try:
            register(self._owner_dir, "modal", sandbox.object_id, ttl_s=spec.ttl_s)
        except BaseException:
            sandbox.terminate(wait=False)
            raise
        box = ModalSandbox(sandbox, owner_dir=self._owner_dir)
        ready = box.run("echo READY", timeout_s=min(ready_timeout_s, 60))
        if "READY" not in ready["text"]:
            sandbox.terminate(wait=False)
            raise RuntimeError(f"modal sandbox not ready: {ready['text'][:200]}")
        return box

    def connect(self, sandbox_id: str) -> Sandbox:
        if self._platform:
            return ModalSandbox(self._platform.get(sandbox_id), owner_dir=self._owner_dir)
        modal = require_modal()
        return ModalSandbox(modal.Sandbox.from_id(sandbox_id), owner_dir=self._owner_dir)


__all__ = ["APP_NAME", "ModalSandbox", "ModalSandboxRuntime"]
