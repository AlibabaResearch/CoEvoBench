"""Status collection: the single source of truth for run monitoring.

One collector feeds every frontend: the CLI table, `ceb run status
--json`, and the web console (design document Section 8.2.2). The JSON
schema is identical across frontends so consumers can diff them.

Data sources: the run registry (manifest, lifecycle), docker inspect
(container state), audit log sizes, and GPU usage — operational metadata
only. No evaluator secrets: no gold, no splits, no Selection
trajectories, no Test task ids. Same-UID processes can read this
without harm; that boundary is documented, not hidden.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from coevobench.core.lifecycle.registry import RunRegistry

_EXCLUDED_KEYS = ("token", "key", "secret", "credential")


def collect_run(registry: RunRegistry, run_id: str) -> dict[str, Any] | None:
    manifest = registry.load_manifest(run_id)
    if manifest is None:
        return None
    lifecycle_path = registry.root / run_id / "lifecycle.json"
    lifecycle: dict[str, Any] = {}
    if lifecycle_path.is_file():
        lifecycle = json.loads(lifecycle_path.read_text())
    doc: dict[str, Any] = {
        "run_id": run_id,
        "bench": manifest.bench,
        "researcher": manifest.researcher_model,
        "replicate": manifest.replicate,
        "phase": lifecycle.get("phase", manifest.phase),
        "started_at": manifest.started_at,
        "ended_at": manifest.ended_at,
        "budget_hours": manifest.budget_hours,
        "budget_left_hours": _budget_left(lifecycle),
        "gpus": list(manifest.gpus),
        "container": _container_state(manifest.container, run_id),
        "workspace_bytes": _tree_bytes(registry.workspace_dir(run_id)),
        "audit_lines": _log_lines(registry.audit_dir(run_id)),
        "guard_hits": _guard_hits(registry.audit_dir(run_id)),
        "timestamp": time.time(),
    }
    return doc


def collect_all(registry: RunRegistry) -> list[dict[str, Any]]:
    docs = []
    for run_id in registry.list_runs():
        doc = collect_run(registry, run_id)
        if doc is not None:
            docs.append(doc)
    return docs


def _budget_left(lifecycle: dict[str, Any]) -> float | None:
    deadline = lifecycle.get("deadline_at")
    if not deadline:
        return None
    return round(max(0.0, (deadline - time.time()) / 3600.0), 3)


def _container_state(container_id: str | None, run_id: str) -> str:
    if container_id:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
        return "not-found"
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=ceb-run-", "--format", "{{.Names}} {{.Status}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    for line in result.stdout.splitlines():
        name, _, status = line.partition(" ")
        if run_id in name:
            return status or "unknown"
    return "no-container"


def _tree_bytes(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _log_lines(audit_dir: Path) -> int:
    log = audit_dir / "bash.log"
    if not log.is_file():
        return 0
    return sum(1 for _ in log.open())


def _guard_hits(audit_dir: Path) -> dict[str, int]:
    counts = {"BLOCK": 0, "SUSPECT": 0}
    log = audit_dir / "bash.log"
    if not log.is_file():
        return counts
    for line in log.open(errors="replace"):
        for marker in counts:
            if marker in line:
                counts[marker] += 1
    return counts


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip credential-shaped values before anything leaves the collector."""
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if any(marker in key.lower() for marker in _EXCLUDED_KEYS):
            cleaned[key] = "[REDACTED]"
        elif isinstance(value, dict):
            cleaned[key] = redact(value)
        else:
            cleaned[key] = value
    return cleaned


__all__ = ["collect_all", "collect_run", "redact"]
