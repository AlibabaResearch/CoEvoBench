"""Broker process with signal-driven cleanup and persistent run ownership."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from pathlib import Path

from coevobench.config.loader import load_config
from coevobench.research.server import Broker, BrokerServer
from coevobench.runtime.run_resources import stop_resources

logger = logging.getLogger(__name__)


def serve(owner: Path, socket: str, bench: str | None = None) -> None:
    manifest = owner / "manifest.json"
    if bench is None and manifest.is_file():
        bench = json.loads(manifest.read_text()).get("bench")
    teacher = None
    if all(os.environ.get(k) for k in ("TEACHER_MODEL", "TEACHER_BASE_URL", "TEACHER_TOKEN")):
        from coevobench.runtime.proxy.teacher import TeacherForwarder

        teacher = TeacherForwarder(owner_dir=owner)
    broker = Broker(config=load_config(), workspace_root=owner / "workspace",
                    bench=bench, teacher=teacher)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with BrokerServer(socket, broker) as server:
        server.serve_forever_bg()
        try:
            delay = 1
            while not stopped.wait(delay):
                try:
                    stop_resources(owner, expired_only=True)
                except Exception:
                    logger.exception(
                        "TTL cleanup failed for %s; retrying while broker stays up", owner
                    )
                    delay = min(60, max(2, delay * 2))
                else:
                    delay = 1
        finally:
            server.shutdown()
            broker.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("socket")
    parser.add_argument("--bench")
    args = parser.parse_args()
    serve(args.run_dir, args.socket, args.bench)


if __name__ == "__main__":
    main()
