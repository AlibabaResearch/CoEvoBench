"""Standalone broker daemon: outlives the launching CLI.

The broker thread previously lived inside the `ceb run launch`
process; when that process printed its summary and exited, the
socket died with it and the researcher lost ./eval and ./sandbox
for the rest of the run. This module daemonizes the broker
for a run's lifetime: setsid, /dev/null stdio, a pidfile next to the
socket, and a stop entry for teardown.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def start_broker_daemon(run_dir: Path, *, bench: str | None = None) -> dict:
    """Daemonize the broker for one run. Idempotent: reuses a live pid."""
    run_dir = Path(run_dir).resolve()
    socket_path = run_dir / "broker.sock"
    pid_file = run_dir / "broker.pid"
    identity_file = run_dir / "broker-process.json"
    from coevobench.runtime.run_resources import process_start

    # AF_UNIX paths are capped (104 bytes on macOS). Long registry roots
    # would kill the daemon at bind time; bind through a short symlink
    # under /tmp instead and expose that path to the envelope.
    bind_target = socket_path
    if len(str(socket_path)) >= 104:
        link = Path(tempfile.gettempdir()) / f"ceb-broker-{uuid.uuid4().hex[:8]}.sock"
        bind_target = link

    if pid_file.is_file():
        pid = int(pid_file.read_text().strip())
        try:
            identity = json.loads(identity_file.read_text())
            if process_start(pid) == identity["start"]:
                return {"pid": pid, "socket": str(socket_path), "reused": True}
        except OSError:
            pid_file.unlink(missing_ok=True)

    # Prefer the repository source tree; fall back to the installed
    # package root (pytest runs from site-packages where parents[2]
    # is not the repo).
    src_root = Path(__file__).resolve().parents[2]
    if not (src_root / "coevobench").is_dir():
        src_root = Path(__file__).resolve().parents[1]
    argv = [sys.executable, "-m", "coevobench.research.daemon", str(run_dir), str(bind_target)]
    if bench:
        argv += ["--bench", bench]
    audit = run_dir / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(src_root)}
    with (audit / "broker.log").open("ab") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        if Path(bind_target).exists() or socket_path.exists():
            try:
                import socket as socket_mod

                probe = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
                probe.settimeout(2)
                probe.connect(str(bind_target))
                probe.close()
                break
            except OSError:
                pass
        if proc.poll() is not None:
            raise RuntimeError(
                f"broker daemon exited before becoming ready; see {audit / 'broker.log'}"
            )
        time.sleep(0.2)
    else:
        proc.kill()
        proc.wait()
        raise RuntimeError("broker daemon failed to signal readiness")

    pid_file.write_text(str(proc.pid) + "\n")
    identity_file.write_text(json.dumps({"pid": proc.pid, "start": process_start(proc.pid)}))
    if bind_target != socket_path:
        socket_path.unlink(missing_ok=True)
        socket_path.symlink_to(bind_target)
    socket_path.chmod(0o660)
    try:
        # The envelope runs as uid 1000 (researcher); align the socket
        # owner so the in-container client can connect. Group lookup is
        # macOS-oriented; Linux hosts simply chown uid 1000.
        gid = None
        try:
            import grp

            gid = grp.getgrnam("researcher").gr_gid
        except KeyError:
            gid = 1000
        os.chown(socket_path, 1000 if gid is None else -1, gid if gid is not None else -1)
    except OSError:
        pass
    return {"pid": proc.pid, "socket": str(socket_path), "reused": False}


def stop_broker_daemon(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    pid_file = run_dir / "broker.pid"
    if not pid_file.is_file():
        return {"stopped": False, "reason": "no pidfile"}
    pid = int(pid_file.read_text().strip())
    from coevobench.runtime.run_resources import process_start

    identity_file = run_dir / "broker-process.json"
    if not identity_file.is_file():
        return {"stopped": False, "reason": "broker process identity missing"}
    identity = json.loads(identity_file.read_text())
    if process_start(pid) != identity["start"]:
        pid_file.unlink(missing_ok=True)
        return {"stopped": False, "reason": "broker already exited or PID changed"}
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            try:
                if process_start(pid) != identity["start"]:
                    break
            except OSError:
                break
            time.sleep(0.2)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    pid_file.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}


__all__ = ["start_broker_daemon", "stop_broker_daemon"]
