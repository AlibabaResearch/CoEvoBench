"""Sandbox-runtime selection: one function, every call site.

The precedence is deliberate:

1. `[sandbox] runtime` in the config, when set — the operator's
   explicit choice, wired through the broker, the operator-side
   CLIs and the replay path. "docker" and "modal" name the local
   and cloud families; "broker" names the in-envelope forwarder.
2. Otherwise an environment probe: docker when the docker CLI is
   reachable, Modal sandboxes otherwise. This is the right default
   on both sides of a run — the operator host has a daemon, and a
   run envelope has no docker socket by design, so a remote
   (Modal) run gets co-located Modal sandboxes and env-ops never
   cross back to the operator host.

Docker pulls honor the eval-image mirrors; Modal pulls honor
`[sandbox] registry_secret` (a Modal secret with registry
credentials, for private task-image registries).
"""

from __future__ import annotations

import shutil


def sandbox_runtime(config=None, *, owner_dir=None):
    """Resolve the sandbox runtime: explicit config, else the probe."""
    settings = getattr(config, "sandbox", None) if config is not None else None
    name = getattr(settings, "runtime", None)

    if name:
        if name == "docker":
            from coevobench.runtime.sandbox.docker import DockerSandboxRuntime

            mirrors = getattr(getattr(config, "eval_images", None), "mirrors", None)
            return DockerSandboxRuntime(mirrors=mirrors or [], owner_dir=owner_dir)
        if name == "modal":
            from coevobench.runtime.sandbox.modal import ModalSandboxRuntime

            secret = getattr(settings, "registry_secret", None)
            return ModalSandboxRuntime(registry_secret=secret, owner_dir=owner_dir)
        if name == "broker":
            from coevobench.runtime.sandbox.broker_client import BrokerSandboxRuntime

            return BrokerSandboxRuntime()
        if name == "daytona":
            from coevobench.runtime.sandbox.daytona import DaytonaSandboxRuntime

            return DaytonaSandboxRuntime(owner_dir=owner_dir)
        raise ValueError(f"unknown sandbox runtime: {name}")

    # The probe: docker CLI present -> the local daemon; otherwise the
    # Modal cloud (the envelope shape — no docker socket by design).
    if shutil.which("docker") is not None:
        from coevobench.runtime.sandbox.docker import DockerSandboxRuntime

        mirrors = getattr(getattr(config, "eval_images", None), "mirrors", None)
        return DockerSandboxRuntime(mirrors=mirrors or [], owner_dir=owner_dir)
    try:
        from coevobench.runtime.sandbox.modal import ModalSandboxRuntime

        return ModalSandboxRuntime(owner_dir=owner_dir)
    except Exception as exc:
        raise RuntimeError(
            f"no sandbox backend available (docker missing, modal failed: {exc})"
        ) from exc


__all__ = ["sandbox_runtime"]
