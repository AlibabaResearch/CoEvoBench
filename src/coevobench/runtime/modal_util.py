"""Shared helpers for the Modal frontends (runhost + sandbox runtime).

One place for the facts every Modal code path needs:
- dependency/auth gate with a single actionable message
- stream reading in the SDK's read-then-wait order
- per-word shell quoting (a naive ' '.join lets 'sh -c a; b' leak
  '; b' to the outer shell)

Live-verified SDK semantics these encode (modal 1.x):
- sandbox ids are object_id (sandbox_id does not exist)
- exec children die with the exec call: a long-lived service must be
  the sandbox entrypoint, never a background exec child
- tunnels forward at the container network layer: a service bound to
  127.0.0.1 never surfaces; bind 0.0.0.0
- ':N' requests N accelerators; '*N' is not a valid count
- the entrypoint runs as root regardless of the image's USER
- Image.from_dockerfile honors full COPY semantics: a COPY'd file is
  in the image at build time (a later RUN reads it), directories
  copy, .dockerignore in the context applies, and context_dir may
  sit apart from the Dockerfile's own directory
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover - trivial import
    import shlex

    _quote = shlex.quote
except ImportError:  # pragma: no cover
    _quote = lambda s: s  # noqa: E731


def require_modal():
    """Import modal or raise one actionable line (never a stack trace)."""
    try:
        import modal  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "modal support needs the extra: uv pip install 'coevobench[modal]'"
        ) from exc
    authenticated = os.environ.get("MODAL_TOKEN_ID") or (
        Path.home() / ".modal.toml"
    ).is_file()
    if not authenticated:
        raise RuntimeError(
            "modal is installed but not authenticated "
            "(modal token set, or MODAL_TOKEN_ID + MODAL_TOKEN_SECRET)"
        )
    return modal


def read_stream(stream) -> str:
    """Drain an exec's stdout stream; bytes decode, failures read empty.

    The SDK contract: read the stream BEFORE wait() — after wait the
    stream may already be detached.
    """
    try:
        raw = stream.read() if hasattr(stream, "read") else str(stream)
    except Exception:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode(errors="replace")
    return raw if isinstance(raw, str) else ""


def exec_sync(sandbox, command: str, *, timeout_s: int = 60, shell: str = "sh") -> dict:
    """Run one command to completion: read streams, then wait.

    Returns the SPI's ExecResult shape {exit_code, text}.
    """
    proc = sandbox.exec(shell, "-c", command, timeout=timeout_s)
    stdout = read_stream(proc.stdout)
    proc.wait()
    return {"exit_code": proc.returncode, "text": stdout}


def quote(part: str) -> str:
    """Shell-quote one word so embedded separators stay in-word."""
    return _quote(str(part))


__all__ = ["exec_sync", "quote", "read_stream", "require_modal"]
