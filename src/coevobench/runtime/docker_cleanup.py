"""Idempotent Docker removal, including concurrent auto-removal at TTL expiry."""

from __future__ import annotations

import subprocess
import time


def remove_container(identity: str, *, timeout_s: float = 30) -> None:
    """Wait for removal to finish; a busy removal is not proof of release."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = subprocess.run(
            ["docker", "rm", "-f", identity], capture_output=True, text=True,
            timeout=max(0.001, deadline - time.monotonic()),
        )
        detail = (result.stderr or result.stdout).strip()
        missing = any(s in detail.lower() for s in ("no such container", "no such object"))
        if result.returncode == 0 or missing:
            return
        remaining = deadline - time.monotonic()
        if "already in progress" not in detail.lower() or remaining <= 0:
            raise RuntimeError(f"container cleanup failed: {identity}: {detail[-2000:]}")
        time.sleep(min(0.2, remaining))
