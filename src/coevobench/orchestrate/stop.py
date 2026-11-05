"""Ordered, retryable run shutdown with submission and resource receipts."""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

from coevobench.core.grading.submission import inspect_submission
from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.orchestrate.broker_daemon import stop_broker_daemon
from coevobench.orchestrate.supervisor import LifecycleStore
from coevobench.runtime import run_resources
from coevobench.runtime.runhost import control


def stop_run(run_id: str, *, registry_root: Path | None = None,
             reason: str = "operator-stop") -> dict:
    registry = RunRegistry(registry_root)
    owner = registry.dir_for(run_id)
    if not owner.is_dir():
        return {"error": f"unknown run {run_id}"}
    with (owner / "stop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = registry.load_manifest(run_id)
        if manifest is None:
            return {"error": f"unknown run {run_id}"}
        store = LifecycleStore(registry.root)
        state = store.load(run_id)
        if state and state.phase != "done":
            state.phase = "stopping"
            state.record("shutdown", reason)
            store.save(state)
        # Close admission before terminating resources, including creations
        # whose SDK/daemon call is still in flight.
        resources = owner / "eval-resources"
        resources.mkdir(exist_ok=True)
        (resources / "closed").touch()
        errors = []
        ident = manifest.container
        if ident:
            for action in (
                lambda: control.stop(manifest.runhost, ident, manifest.process_start),
                lambda: control.archive(manifest.runhost, ident, owner / "audit/researcher"),
            ):
                try:
                    action()
                except Exception as exc:
                    errors.append(str(exc))
        try:
            stop_broker_daemon(owner)
        except Exception as exc:
            errors.append(str(exc))
        try:
            run_resources.stop_resources(owner)
        except Exception as exc:
            errors.append(str(exc))
        if ident:
            try:
                control.remove(manifest.runhost, ident)
            except Exception as exc:
                errors.append(str(exc))
        submission = inspect_submission(registry.workspace_dir(run_id), manifest.bench)
        audit = owner / "audit"
        audit.mkdir(exist_ok=True)
        (audit / "submission-status.json").write_text(json.dumps(submission, indent=2))
        result = {"run_id": run_id, "stopped": not errors, "container": ident,
                  "reason": reason, "submission": submission, "cleanup_errors": errors}
        (audit / "stop.json").write_text(json.dumps(result, indent=2))
        manifest.submission = submission
        manifest.phase = "stopping" if errors else "done"
        manifest.ended_at = time.time()
        registry.save_manifest(manifest)
        if state:
            state.phase = manifest.phase
            state.record("cleanup-pending" if errors else "stopped")
            store.save(state)
        if not errors:
            from coevobench.orchestrate.monitor import stop_monitor

            stop_monitor(owner)
        return result
