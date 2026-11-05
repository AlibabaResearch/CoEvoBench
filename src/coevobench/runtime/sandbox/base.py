"""SandboxRuntime SPI.

Semantic invariants every implementation must satisfy:
- run() executes each command in a fresh shell; cwd/exports do not persist
  across calls.
- The sandbox persists across run() calls: files and background daemons
  survive.
- run(timeout_s) terminates the full process tree, not only the direct child.
- kill() is idempotent; provider TTL is the fallback cleanup guarantee.
- FileWrite.mode uses true octal semantics (0o755); adapters translate.
- create_from_context builds the image from a local context directory
  (Dockerfile at its root) and runs the sandbox on it in the same
  provider: the product never crosses providers, so no registry is
  involved. The build follows standard COPY semantics; .dockerignore
  in the context applies; build failures surface the log tail.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict


class ExecResult(TypedDict):
    exit_code: int
    text: str


def validate_context(ctx_dir: Path) -> None:
    """Guard a build-context directory before create_from_context.

    Dockerfile at the root; no symlink escaping the directory. The
    same rule both sandbox families apply — the context is about to
    be sent to a builder (local daemon or Modal's), so what it
    contains must be exactly what the caller chose.
    """
    ctx = Path(ctx_dir)
    if not (ctx / "Dockerfile").is_file():
        raise ValueError(f"context must contain Dockerfile at its root: {ctx}")
    resolved = ctx.resolve()
    for child in ctx.rglob("*"):
        if child.is_symlink() and not child.resolve().is_relative_to(resolved):
            raise ValueError(f"context symlink escapes the context: {child}")


@dataclass(frozen=True)
class FileWrite:
    path: str
    data: bytes | str
    mode: int = 0o644


@dataclass(frozen=True)
class SandboxSpec:
    # image: the image to run — a registry reference, or empty on the
    # create_from_context path, where the image comes from the build
    # context and never needs to be named.
    image: str = ""
    cpu: int = 2
    mem_gib: int = 8
    ttl_s: int = 1800
    network: bool = False
    # Ports the sandbox will serve on and that must be reachable from
    # outside the provider (e.g. a vLLM OpenAI endpoint inside a remote
    # run). Local docker maps nothing (localhost already works); remote
    # providers (Modal) forward these through public tunnels.
    ports: tuple[int, ...] = ()


class Sandbox(Protocol):
    id: str

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> ExecResult: ...

    def write_files(self, files: Iterable[FileWrite]) -> None: ...

    def read_file(self, path: str) -> str: ...

    def kill(self) -> None: ...


class SandboxRuntime(Protocol):
    name: str

    def create(self, spec: SandboxSpec, *, ready_timeout_s: int = 120) -> Sandbox: ...

    def create_from_context(
        self, ctx_dir: Path, spec: SandboxSpec, *, ready_timeout_s: int = 120,
        build_timeout_s: int = 1800,
    ) -> Sandbox: ...

    def connect(self, sandbox_id: str) -> Sandbox: ...


__all__ = [
    "ExecResult",
    "FileWrite",
    "Sandbox",
    "SandboxRuntime",
    "SandboxSpec",
    "validate_context",
]
