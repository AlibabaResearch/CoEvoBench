"""Detached run monitor: enforce the deadline and finish exited researchers."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.orchestrate.stop import stop_run
from coevobench.orchestrate.supervisor import LifecycleStore
from coevobench.runtime.run_resources import process_start, stop_resources
from coevobench.runtime.runhost.control import status


def reconcile(owner: Path) -> bool:
    registry = RunRegistry(owner.parent.parent)
    manifest = registry.load_manifest(owner.name)
    state = LifecycleStore(registry.root).load(owner.name)
    if manifest is None or state is None:
        raise RuntimeError("run monitor needs persisted manifest and lifecycle")
    reason = None
    if state.phase in ("stopping", "done"):
        reason = "reconcile-stop"
    elif time.time() >= state.deadline_at:
        reason = "deadline-reached"
    elif manifest.container and status(
        manifest.runhost, manifest.container, manifest.process_start
    ) not in ("running", "created", "restarting"):
        reason = "researcher-exited"
    if reason:
        result = stop_run(owner.name, registry_root=owner.parent.parent, reason=reason)
        return result["stopped"]
    identity_file = owner / "broker-process.json"
    if identity_file.is_file():
        identity = json.loads(identity_file.read_text())
        if process_start(identity["pid"]) != identity["start"]:
            return stop_run(owner.name, registry_root=owner.parent.parent,
                            reason="broker-exited")["stopped"]
    stop_resources(owner, expired_only=True)
    return False


def start_monitor(owner: Path) -> None:
    (owner / "audit").mkdir(exist_ok=True)
    with (owner / "audit/monitor.log").open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "coevobench.orchestrate.monitor", str(owner.resolve())],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    (owner / "monitor.pid").write_text(str(proc.pid))
    (owner / "monitor-process.json").write_text(json.dumps({
        "pid": proc.pid, "start": process_start(proc.pid),
    }))


def stop_monitor(owner: Path) -> None:
    identity = owner / "monitor-process.json"
    if not identity.is_file():
        return
    record = json.loads(identity.read_text())
    pid = record["pid"]
    if pid == os.getpid():
        return
    if record["start"] and process_start(pid) == record["start"]:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    (owner / "monitor.pid").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        while True:
            if not args.run_dir.is_dir():
                return
            try:
                if reconcile(args.run_dir):
                    return
            except Exception as exc:
                print(json.dumps({"time": time.time(), "error": str(exc)}), flush=True)
            time.sleep(2)
    finally:
        pid = args.run_dir / "monitor.pid"
        if pid.is_file() and pid.read_text().strip() == str(os.getpid()):
            pid.unlink()


if __name__ == "__main__":
    main()
