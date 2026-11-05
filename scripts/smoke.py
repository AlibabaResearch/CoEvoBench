"""Exercise a real run envelope, broker, toy evaluation and deadline cleanup."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from supervise_runs import supervise_once

from coevobench.cli.run import launch_run, stop_run
from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.orchestrate.supervisor import LifecycleStore

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="coevobench-run:latest")
    args = parser.parse_args()
    root = PROJECT / ".local/smoke-registry"
    info = launch_run(
        bench="toy",
        researcher_model="setup-smoke",
        researcher_cli="example",
        budget_hours=0.05,
        gpus=(),
        image=args.image,
        registry_root=root,
    )
    results = []
    try:
        for command in [
            "id",
            "ceb --version",
            "test ! -S /var/run/docker.sock",
            "./timer.sh status",
            "./eval --split visible --n 1",
        ]:
            result = subprocess.run(
                ["docker", "exec", "-w", "/workspace", info["container"], "sh", "-lc", command],
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            if command == "./timer.sh status":
                assert "unknown" not in result.stdout, result.stdout
            if command.startswith("./eval"):
                assert "score=1.0" in result.stdout, result.stdout
            results.append({"command": command, "output": result.stdout.strip()})
        registry = RunRegistry(root)
        store = LifecycleStore(registry.root)
        state = store.load(info["run_id"])
        assert state is not None
        # Expire this disposable smoke run to verify the supervisor's stop path.
        state.deadline_at = 0
        store.save(state)
        stopped = supervise_once(root)
        assert any(item["run_id"] == info["run_id"] for item in stopped), stopped
        assert store.load(info["run_id"]).phase == "done"
        report = {"run": info, "checks": results, "deadline_cleanup": stopped}
        target = PROJECT / ".local/setup-logs/smoke.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        stop_run(info["run_id"], registry_root=root)


if __name__ == "__main__":
    main()
