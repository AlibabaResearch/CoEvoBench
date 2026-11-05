"""ResearcherCLI SPI (design document Section 3.5).

Provision runs once at run-envelope init, idempotently: verify or install
the CLI, write its settings (permission deny lists), install the audit hook
(Section 5.6 evidence source), and configure the cleanroom HOME. Entrypoint
yields (argv, env) for the RunHost to place in the envelope; the CLI layer
never knows whether it runs in a local container or a remote sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from coevobench.runtime.runhost.base import Entrypoint


@dataclass(frozen=True)
class ResearcherConfig:
    """Paper: model-CLI pair plus reasoning effort (Section 4.1)."""

    model: str
    cli: str
    reasoning_effort: str | None = "high"
    cli_endpoint: str = ""


@dataclass(frozen=True)
class Endpoint:
    """Any OpenAI/Anthropic-compatible gateway; credentials via env only."""

    base_url: str
    token_env: str = "RESEARCHER_TOKEN"


@dataclass(frozen=True)
class ProvisionContext:
    """Paths inside the run envelope, provided by the launcher."""

    workspace: str = "/workspace"
    home: str = "/home/researcher"
    audit_stream: str = "/dev/stderr"


class ResearcherCLI(Protocol):
    name: str

    def provision(self, config: ResearcherConfig, ctx: ProvisionContext) -> None: ...

    def entrypoint(
        self, config: ResearcherConfig, endpoint: Endpoint, budget_hours: float
    ) -> Entrypoint: ...


__all__ = [
    "Endpoint",
    "ProvisionContext",
    "ResearcherCLI",
    "ResearcherConfig",
]
