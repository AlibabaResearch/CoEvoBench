"""Enforce persisted deadlines for registered runs on the operator host."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.orchestrate.monitor import reconcile


def supervise_once(root: Path | None = None) -> list[dict]:
    registry = RunRegistry(root)
    stopped = []
    for run_id in registry.list_runs():
        manifest = registry.load_manifest(run_id)
        if manifest is None or manifest.phase not in ("running", "stopping"):
            continue
        try:
            if reconcile(registry.dir_for(run_id)):
                stopped.append({"run_id": run_id, "stopped": True})
        except Exception as exc:
            stopped.append({"run_id": run_id, "stopped": False, "error": str(exc)})
    return stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        for record in supervise_once(args.registry_root):
            print(json.dumps(record), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
