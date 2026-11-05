"""Explicit submission protocol (B6).

DONE.md alone is a progress document; the researcher signal for "this
is my final answer" must be explicit and verifiable. The protocol:

1. request: fingerprint the current submission directory + DONE.md
   (any later edit changes the digest — a stale request is refused)
2. confirm: the operator (or the supervisor's deadline path) verifies
   the digest matches and records the submission in the run manifest

The supervisor's deadline capture checks for a confirmed submission
first, falling back to DONE.md-only (marked unconfirmed).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REQUIRED = ("harness", "model_path.txt", "meta.json", "serve_config.json")


def checkpoint_path(workspace: Path, declared: Path) -> Path:
    """Resolve the public workspace mount against its operator-side copy.

    The researcher writes model_path.txt in envelope terms — /base or
    /workspace/... — and the operator-side replay resolves those
    against the run's workspace root. Host-absolute paths pass.
    """
    workspace = Path(workspace)
    if declared.is_relative_to("/workspace/base"):
        declared = Path("/base") / declared.relative_to("/workspace/base")
    if declared == Path("/base") or declared.is_relative_to("/base"):
        if workspace == Path("/workspace"):
            return declared
        manifest = workspace.parent / "manifest.json"
        if manifest.is_file():
            base = json.loads(manifest.read_text()).get("base_dir")
            if base:
                return (Path(base) / declared.relative_to("/base")).resolve()
        return (workspace / "base" / declared.relative_to("/base")).resolve()
    if declared.is_relative_to("/workspace"):
        return (workspace / declared.relative_to("/workspace")).resolve()
    return declared.resolve()


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def fingerprint(workspace: Path, bench: str) -> dict:
    """Validate the submission's shape and compute its digest."""
    workspace = Path(workspace)
    submission = workspace / "out" / f"{bench}_final"
    for name in REQUIRED:
        if not (submission / name).exists():
            raise ValueError(f"final submission missing {name}: {submission}")
    for name in ("meta.json", "serve_config.json"):
        if not isinstance(json.loads((submission / name).read_text()), dict):
            raise ValueError(f"invalid {name}")
    model = Path((submission / "model_path.txt").read_text().strip())
    if not model.is_absolute():
        raise ValueError("model_path.txt must contain an absolute checkpoint path")
    if not (checkpoint_path(workspace, model) / "config.json").is_file():
        raise ValueError("final checkpoint is missing config.json")
    done = workspace / "out" / "DONE.md"
    if not done.is_file() or not done.read_text().strip():
        raise ValueError("Write out/DONE.md describing the final result first")

    files = {
        str(p.relative_to(submission)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(submission.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
        and p.name != "submission.json"
    }
    return {
        "digest": _digest({"submission": files, "done": done.read_text()}),
        "submission_dir": str(submission),
        "timestamp": time.time(),
    }


def request(workspace: Path, bench: str) -> Path:
    """Write the submission request (the researcher-side act)."""
    record = fingerprint(workspace, bench)
    target = Path(workspace) / "out" / f"{bench}_final" / "submission.json"
    target.write_text(json.dumps(record, indent=2))
    return target


def confirm(workspace: Path, bench: str) -> dict:
    """Operator-side verification: the request digest still matches."""
    submission = Path(workspace) / "out" / f"{bench}_final"
    request_file = submission / "submission.json"
    if not request_file.is_file():
        raise ValueError("no submission request found (researcher never submitted)")
    request_record = json.loads(request_file.read_text())
    current = fingerprint(Path(workspace), bench)
    if request_record["digest"] != current["digest"]:
        raise ValueError(
            "submission changed after the request (digest mismatch); "
            "the researcher must re-request"
        )
    return {
        "confirmed": True,
        "digest": current["digest"],
        "submission_dir": str(submission),
        "confirmed_at": time.time(),
    }


def inspect_submission(workspace: Path, bench: str) -> dict:
    """Describe nomination completeness without creating a researcher signal."""
    submission = Path(workspace) / "out" / f"{bench}_final"
    missing = [name for name in REQUIRED if not (submission / name).exists()]
    if not (Path(workspace) / "out/DONE.md").is_file():
        missing.append("out/DONE.md")
    if missing:
        return {"status": "incomplete", "confirmed": False, "missing": missing}
    try:
        fingerprint(Path(workspace), bench)
        if not (submission / "submission.json").is_file():
            return {"status": "unconfirmed", "confirmed": False,
                    "reason": "researcher has not run ./submit"}
        record = confirm(Path(workspace), bench)
        return {"status": "confirmed", **record}
    except (ValueError, OSError, KeyError) as exc:
        return {"status": "invalid", "confirmed": False, "reason": str(exc)}


__all__ = ["confirm", "fingerprint", "request", "inspect_submission", "REQUIRED"]
