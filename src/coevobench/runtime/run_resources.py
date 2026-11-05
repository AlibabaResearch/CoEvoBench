"""Operator receipts for evaluation processes and containers owned by a run."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from coevobench.runtime.docker_cleanup import remove_container


def _write_json(path: Path, data: dict) -> None:
    """Replace records atomically while holding the owner's resource lock."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def _cleanup_event(root: Path, action: str, path: Path, receipt: dict, **details) -> dict:
    event = {"time": time.time(), "action": action, "receipt": path.name,
             "kind": receipt.get("kind"), "identity": receipt.get("identity"), **details}
    with (root / "cleanup-events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    return event


def _release_receipt(root: Path, path: Path, receipt: dict) -> None:
    if not receipt.get("released_at"):
        receipt["released_at"] = time.time()
        receipt.pop("cleanup_retry_at", None)
        _write_json(path, receipt)
        _cleanup_event(root, "released", path, receipt)


def remaining_seconds(owner: Path | None, default: int = 86400) -> int:
    """Seconds until the run's deadline; raises when the run is over.

    Teacher relays and evaluation workers gate every upstream call on
    this: after the deadline, access closes and owned resources reap.
    """
    if owner is None or not (owner / "lifecycle.json").is_file():
        return default
    state = json.loads((owner / "lifecycle.json").read_text())
    remaining = math.floor(state["deadline_at"] - time.time())
    if state.get("phase", "running") != "running" or remaining < 1:
        raise ValueError("run deadline reached")
    return min(default, remaining)


def process_start(pid: int) -> str:
    """Stable process identity; a recycled PID must never be reaped."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return "" if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, IndexError, OSError):
        fields = subprocess.run(
            ["ps", "-o", "lstart=", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.split()
        return "" if not fields or fields[-1].startswith("Z") else " ".join(fields[:-1])


def check_active(owner: Path | None) -> None:
    if owner is None:
        return
    if (owner / "eval-resources/closed").exists():
        raise RuntimeError("run ended; resources are closed")
    remaining_seconds(owner)


def register(owner: Path | None, kind: str, identity, *, ttl_s: float | None = None) -> None:
    if owner is None:
        return
    root = owner / "eval-resources"
    root.mkdir(exist_ok=True, mode=0o700)
    if kind == "container":
        (root / "docker-owned").touch()
    receipt = {"kind": kind, "identity": identity}
    if ttl_s is not None:
        receipt["expires_at"] = time.time() + ttl_s
    if kind == "process":
        receipt["start"] = process_start(identity)
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lifecycle = owner / "lifecycle.json"
        state = json.loads(lifecycle.read_text()) if lifecycle.is_file() else {}
        if (
            (root / "closed").exists()
            or state.get("phase", "running") != "running"
            or time.time() >= state.get("deadline_at", float("inf"))
        ):
            secrets = owner / ".cloud-secrets.json"
            _stop(receipt, json.loads(secrets.read_text()) if secrets.is_file() else None)
            raise RuntimeError("run ended; evaluation resource refused")
        _write_json(root / f"{uuid.uuid4().hex}.json", receipt)


def release(owner: Path | None, kind: str, identity) -> None:
    """Retain the receipt as audit evidence after successful cleanup."""
    if owner is None:
        return
    root = owner / "eval-resources"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in root.glob("*.json"):
            receipt = json.loads(path.read_text())
            if receipt["kind"] == kind and receipt["identity"] == identity:
                _release_receipt(root, path, receipt)


def _stop(receipt: dict, settings=None) -> None:
    if receipt["kind"] == "modal":
        from coevobench.runtime.modal_platform import ModalPlatform

        ModalPlatform(settings).terminate(receipt["identity"])
        return
    if receipt["kind"] == "daytona-snapshot":
        from coevobench.runtime.sandbox.daytona import client

        try:
            client(settings).snapshot.delete(receipt["identity"])
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise
        return
    if receipt["kind"] == "ssh-serve":
        from coevobench.runtime.serve.ssh import stop_remote

        stop_remote(receipt["identity"], settings)
        return
    if receipt["kind"] == "daytona":
        from coevobench.runtime.sandbox.daytona import client

        api = client(settings)
        try:
            api.delete(api.get(receipt["identity"]), wait=True)
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise
        return
    if receipt["kind"] == "container":
        remove_container(receipt["identity"])
    elif receipt["kind"] == "process":
        pid = int(receipt["identity"])
        try:
            if (not receipt["start"] or process_start(pid) != receipt["start"]
                    or os.getpgid(pid) != pid):
                return
            os.killpg(pid, signal.SIGTERM)
            time.sleep(0.2)
            current = process_start(pid)
            if not current or current == receipt["start"]:
                os.killpg(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _containers(owner: Path) -> list[str]:
    """Labels close the create-before-receipt crash window for Docker."""
    if not (owner / "eval-resources/docker-owned").is_file():
        return []
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required to clean up this run's containers")
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=coevobench.run_dir={owner.resolve()}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect owned Docker resources: {result.stderr[-2000:]}")
    return result.stdout.split()


def stop_resources(owner: Path, *, expired_only: bool = False) -> None:
    path = owner / ".cloud-secrets.json"
    settings = json.loads(path.read_text()) if path.is_file() else None
    root = owner / "eval-resources"
    root.mkdir(exist_ok=True, mode=0o700)
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not expired_only:
            (root / "closed").touch()
        failed = []
        for path in root.glob("*.json"):
            receipt = {}
            try:
                receipt = json.loads(path.read_text())
                if not isinstance(receipt, dict):
                    receipt = {}
                    raise ValueError("resource receipt must be an object")
                if receipt.get("released_at"):
                    continue
                if expired_only and receipt.get("expires_at", float("inf")) > time.time():
                    continue
                if expired_only and receipt.get("cleanup_retry_at", 0) > time.time():
                    failed.append(receipt["last_cleanup_error"])
                    continue
                _stop(receipt, settings)
                _release_receipt(root, path, receipt)
            except Exception as exc:
                error = _cleanup_event(root, "failed", path, receipt,
                                       error=type(exc).__name__, message=str(exc))
                failed.append(error)
                if receipt:
                    attempts = receipt.get("cleanup_failures", 0) + 1
                    receipt.update(cleanup_failures=attempts, last_cleanup_error=error,
                                   cleanup_retry_at=time.time() + min(60, 2 ** min(attempts, 6)))
                    _write_json(path, receipt)
        if not expired_only:
            try:
                identities = _containers(owner)
            except Exception as exc:
                identities = []
                failed.append(_cleanup_event(root, "failed", Path("docker-label-discovery"), {},
                                             error=type(exc).__name__, message=str(exc)))
            for ident in identities:
                try:
                    remove_container(ident)
                except Exception as exc:
                    failed.append(_cleanup_event(
                        root, "failed", Path("docker-label-discovery"),
                        {"kind": "container", "identity": ident},
                        error=type(exc).__name__, message=str(exc),
                    ))
        _write_json(root / "cleanup-status", {"pending": failed})
        if failed:
            raise RuntimeError(f"{len(failed)} resources still need cleanup; retry run stop")


def run_container(command: list[str], name: str, owner: Path | None, log, timeout: int):
    """Register a created container before starting it; closing a run wins races."""
    check_active(owner)
    labels = []
    if owner:
        root = owner / "eval-resources"
        root.mkdir(exist_ok=True)
        (root / "docker-owned").touch()
        labels = ["--label", f"coevobench.run_dir={owner.resolve()}"]
    create = ["docker", "create", *labels, *command[2:]]
    try:
        subprocess.run(create, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        register(owner, "container", name)
        subprocess.run(
            ["docker", "start", "-a", name],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    finally:
        remove_container(name)
        release(owner, "container", name)
