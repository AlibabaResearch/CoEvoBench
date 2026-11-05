"""Modal-backed RunHost: one remote sandbox per development run.

The remote profile (design Section 3.8): the run executes in a Modal
sandbox with GPU passthrough and kernel isolation from the operator
host (HostCapabilities tells the truth: True here, False for local
docker). The broker boundary differs from the local profile — no unix
socket can cross hosts, so in-run sandbox work is served by the
co-located ModalSandboxRuntime (env-ops do not cross the network), and
image builds push to an OCI registry the remote pulls from.

Usage follows the harbor project's proven Modal SDK recipe: a deployed
App (`App.lookup(create_if_missing=True)`), `Image.from_registry`,
a keepalive entrypoint (many task images have short-lived commands
that would terminate the sandbox before the first exec — the run's
real command starts via exec and outlives it), and the
read-streams-then-wait ordering for exec results.
"""

from __future__ import annotations

from pathlib import Path

from coevobench.runtime.modal_util import exec_sync, quote, read_stream, require_modal
from coevobench.runtime.runhost.base import (
    Entrypoint,
    HostCapabilities,
    RunEnvelope,
    RunHandle,
)

APP_NAME = "coevobench-runhost"
KEEPALIVE = ("sh", "-c", "sleep infinity")
# The port the run's served model listens on inside the sandbox.
# Judges and episodes reach it through the public tunnel (never
# localhost — a Modal sandbox is its own network namespace), and the
# run command re-points VLLM_BASE at the tunnel when the platform
# drives episodes from outside.
SERVE_PORT = 8000


class ModalRunHandle:
    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox

    @property
    def id(self) -> str:
        return getattr(self._sandbox, "object_id", "") or ""

    def status(self) -> str:
        rc = getattr(self._sandbox, "returncode", None)
        if rc is not None:
            return "exited"
        phase = str(getattr(self._sandbox, "_phase", "running")).lower()
        if "terminat" in phase or "exited" in phase or "done" in phase:
            return "exited"
        return "running"

    def logs(self, *, tail: int = 200) -> str:
        """Best-effort entrypoint stdout (the stream drains once)."""
        raw = read_stream(getattr(self._sandbox, "stdout", None))
        return "\n".join(raw.splitlines()[-tail:])

    def exec(self, command: str, *, timeout_s: int = 60) -> dict:
        return exec_sync(self._sandbox, command, timeout_s=timeout_s, shell="sh")

    def fetch(self, remote: str, local_path: str) -> None:
        """Stream a file out of the run sandbox to the operator host.

        Artifacts land operator-side for freeze; large checkpoints move
        through a Modal Volume when the deployment mounts one.
        """
        data = self._sandbox.filesystem.read_bytes(remote)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(data)

    def endpoint(self, port: int = SERVE_PORT) -> str:
        """The public URL of the run's served model.

        The researcher serves the trained weights inside the run
        sandbox (0.0.0.0:SERVE_PORT there — tunnels only forward what
        binds the container network, not loopback); every consumer
        outside the sandbox points at this tunnel URL.
        """
        tunnels = self._sandbox.tunnels()
        tunnel = tunnels.get(port)
        if tunnel is None:
            raise RuntimeError(f"no tunnel for port {port}")
        host, tunnel_port = tunnel.tcp_socket
        return f"http://{host}:{tunnel_port}"

    def stop(self) -> None:
        self._sandbox.terminate(wait=True)


class ModalRunHost:
    name = "modal"
    capabilities = HostCapabilities(
        persistent_workspace=True,
        gpu_passthrough=True,
        network_endpoints=True,
        kernel_isolated=True,
    )

    def __init__(self, *, profile: str = "H100", count: int = 4) -> None:
        self._profile = profile
        self._count = count

    def launch(
        self,
        envelope: RunEnvelope,
        entrypoint: Entrypoint,
        workspace_dir: str,
        *,
        base_dir: str | None = None,
        broker_socket_host_path: str | None = None,
    ) -> RunHandle:
        """Launch the run sandbox with the envelope image and GPUs.

        This primitive runs an explicit remote command. It does not transfer a
        local workspace/base mount or connect a local broker socket. Complete
        remote researcher deployment needs a separate transport composition.

        Two live-verified Modal facts shape the command construction:
        - entrypoints run as root regardless of the image's USER, and
          researcher CLIs refuse unsafe-permission flags under root:
          the real command setpriv-drops to uid 1000 with the merged
          env re-exported (setpriv wipes the environment).
        - the run's output goes to /workspace/.run.log because the
          entrypoint stdout stream drains once; the log file survives
          for later reads.
        """
        if broker_socket_host_path:
            raise ValueError("Modal RunHost needs an operator-provided remote broker transport")
        modal = require_modal()

        image = modal.Image.from_registry(envelope.image)
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        kwargs: dict = {
            "image": image,
            "app": app,
            "timeout": 60 * 60 * 24,
            # the served-model tunnel lives with the run
            "unencrypted_ports": [SERVE_PORT],
        }
        if self._profile:
            # ':N' requests N accelerators (parse_gpu_config's contract).
            kwargs["gpu"] = (
                f"{self._profile}:{self._count}" if self._count > 1 else self._profile
            )
        sandbox = modal.Sandbox.create(*KEEPALIVE, **kwargs)

        merged_env = {**envelope.env, **entrypoint.env}
        env_exports = " ".join(
            f"{key}={quote(value)}" for key, value in merged_env.items()
        )
        argv = " ".join(quote(part) for part in entrypoint.argv)
        # The workspace is the sandbox's own directory (a Volume mounts
        # here when the deployment has one — no bind mounts cross hosts).
        command = (
            "mkdir -p /workspace && cd /workspace && "
            "nohup setpriv --reuid=1000 --regid=1000 --clear-groups "
            f"env {env_exports} HOME=/home/researcher "
            f"{argv} > /workspace/.run.log 2>&1 & echo started"
        )
        exec_sync(sandbox, command, timeout_s=60)
        return ModalRunHandle(sandbox)

    def logs(self, sandbox_id: str) -> str:
        """Re-attach to a live sandbox by id and read its run log."""
        modal = require_modal()
        sandbox = modal.Sandbox.from_id(sandbox_id)
        out = exec_sync(sandbox, "cat /workspace/.run.log 2>/dev/null", timeout_s=30)
        return out["text"]


__all__ = ["APP_NAME", "ModalRunHandle", "ModalRunHost"]
