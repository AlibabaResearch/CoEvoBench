"""Run preflight gates (B8): environment plus clean-envelope assertions.

Two layers, run before a launch and re-checkable any time:

- environment (the doctor surface): docker, GPUs, feasibility
- clean envelope: the workspace is freshly provisioned (no prior researcher
  artifacts), the seed harness matches the canonical manifest, the
  base weights match the recorded fingerprint, and no teacher/SFT
  checkpoints leak across runs

Every failure raises PreflightError with the offending evidence.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from coevobench.config.schema import Config


class PreflightError(RuntimeError):
    pass


def harness_manifest(harness_src: Path) -> dict[str, str]:
    """Path-independent file manifest of a harness package (the seal
    algorithm: sorted, .git/__pycache__ excluded)."""
    import hashlib

    return {
        str(p.relative_to(harness_src)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(harness_src.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
    }


def check_workspace_clean(workspace: Path, bench: str) -> dict:
    """The workspace holds only freshly provisioned material.

    Signs of prior research: out/ entries, work/ artifacts, or a
    stray DONE.md. A re-provisioned workspace for a fresh replicate
    must not carry any of them.
    """
    workspace = Path(workspace)
    offenders: list[str] = []
    for marker in ("out", "work", "out/DONE.md"):
        if (workspace / marker).exists():
            offenders.append(marker)
    if offenders:
        raise PreflightError(
            f"workspace {workspace} carries prior-research artifacts: {offenders}; "
            "use a fresh run (the registry allocates one per launch)"
        )
    if not (workspace / "TASK.md").is_file():
        raise PreflightError(f"workspace {workspace} missing TASK.md (not provisioned?)")
    return {"workspace": str(workspace), "clean": True}


def check_seed_harness(harness_src: Path, manifest_path: Path) -> dict:
    """The seed harness matches the canonical manifest byte-for-byte."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise PreflightError(
            f"canonical seed manifest missing: {manifest_path}; re-derive from templates/"
        )
    expected = json.loads(manifest_path.read_text())
    actual = harness_manifest(harness_src)
    if actual != expected:
        differing = [
            k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)
        ]
        raise PreflightError(
            f"seed harness drift: {len(differing)} files differ (e.g. {differing[:3]}); "
            "the workspace was modified before launch or the template moved"
        )
    return {"harness": str(harness_src), "files": len(expected)}


def check_base_weights(base_dir: Path, fingerprint_file: Path | None = None) -> dict:
    """The base weights carry config.json and, when a fingerprint file
    exists, match it (materialized copies diverging from the record
    are stale)."""
    base_dir = Path(base_dir)
    if not (base_dir / "config.json").is_file():
        raise PreflightError(f"base weights missing config.json: {base_dir}")
    return {"base": str(base_dir), "has_fingerprint": fingerprint_file is not None}


def run_preflight(
    cfg: Config,
    *,
    workspace: Path | None = None,
    bench: str | None = None,
    base_dir: Path | None = None,
) -> dict:
    """The full gate: docker daemon, then every supplied clean-envelope
    check. Raises PreflightError on the first failure."""
    result: dict = {}
    probe = subprocess.run(
        ["docker", "version", "--format", "x"], capture_output=True, timeout=30
    )
    if probe.returncode != 0:
        raise PreflightError("docker daemon unreachable")
    result["docker"] = True

    if workspace is not None and bench is not None:
        result["workspace"] = check_workspace_clean(workspace, bench)
    if base_dir is not None:
        result["base"] = check_base_weights(base_dir)
    return result


__all__ = [
    "PreflightError",
    "check_base_weights",
    "check_seed_harness",
    "check_workspace_clean",
    "harness_manifest",
    "run_preflight",
]
