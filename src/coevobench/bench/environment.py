"""The welded exec seam (design document Section 4.2; integrity primitive 1).

SandboxEnvironment is the environment object a harness receives: one exit,
implemented as Sandbox.run. The harness can rewrite everything above this
seam, but it cannot swap the backend — swapping to a local environment
would mean reading gold and splits off the host. The platform constructs
this object; harnesses never do.

The contract aligns with the mini-swe-agent Environment protocol, so
ported harnesses work unmodified.
"""

from __future__ import annotations

from typing import Any, Protocol

from coevobench.runtime.sandbox.base import Sandbox


class Environment(Protocol):
    def execute(self, action: Any, cwd: str = "") -> dict: ...


class SandboxEnvironment:
    """Adapts a Sandbox to the harness-visible Environment protocol."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    def execute(self, action: Any, cwd: str = "", **kwargs: Any) -> dict:
        """Run one action; `timeout` (seconds) is honored when passed.

        Evolved harnesses call execute with keyword options the seed
        never used (timeout being the common one) — the seam accepts
        them rather than breaking the episode: the whole point of the
        co-evolution protocol is that the harness above this seam can
        grow. Unknown keywords are ignored, never fatal.
        """
        command = _command_of(action)
        timeout_s = kwargs.get("timeout")
        run_kwargs: dict[str, Any] = {"cwd": cwd or None}
        if isinstance(timeout_s, (int, float)):
            run_kwargs["timeout_s"] = int(timeout_s)
        result = self._sandbox.run(command, **run_kwargs)
        return {
            "output": result["text"],
            "returncode": result["exit_code"],
            "exception_info": None,
        }

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id


def _command_of(action: Any) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        command = action.get("command") or action.get("cmd") or ""
        if not isinstance(command, str):
            raise ValueError("action command must be a string")
        return command
    raise ValueError(f"unsupported action type: {type(action).__name__}")


__all__ = ["Environment", "SandboxEnvironment"]
