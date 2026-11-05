"""RunHost SPI: where a development run executes.

A RunHost is a composition-level profile: selecting one binds a whole set of
component implementations for that host (in-run sandbox runtime, broker
transport, image distribution path). Invariants for every implementation:

- The workspace contract (paper Appendix A) is identical on every host.
- Test data, splits, grading, and credentials never move into the run
  envelope; they stay in the operator domain.
- Selection records land operator-side.
- status()/logs() work for remote runs the same way as local ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from coevobench.runtime.gpu.profile import GPURequest
from coevobench.runtime.sandbox.base import ExecResult


@dataclass(frozen=True)
class RunEnvelope:
    """Deployment projection of a RunSpec: how the run executes."""

    image: str
    gpus: GPURequest
    env: Mapping[str, str] = field(default_factory=dict)
    workspace_gib: int = 50
    budget_hours: float = 24.0


@dataclass(frozen=True)
class HostCapabilities:
    persistent_workspace: bool
    gpu_passthrough: bool
    network_endpoints: bool
    kernel_isolated: bool


@dataclass(frozen=True)
class Entrypoint:
    argv: Sequence[str]
    env: Mapping[str, str]


class RunHandle(Protocol):
    id: str

    def status(self) -> str: ...

    def logs(self, *, tail: int = 200) -> str: ...

    def exec(self, command: str, *, timeout_s: int = 60) -> ExecResult: ...

    def fetch(self, remote: str, local_path: str) -> None: ...

    def stop(self) -> None: ...


class RunHost(Protocol):
    name: str
    capabilities: HostCapabilities

    def launch(
        self, envelope: RunEnvelope, entrypoint: Entrypoint, workspace_dir: str
    ) -> RunHandle: ...


__all__ = [
    "Entrypoint",
    "HostCapabilities",
    "RunEnvelope",
    "RunHandle",
    "RunHost",
]
