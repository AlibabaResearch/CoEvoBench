"""Launch an experiment after validating its environment and configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from preflight import PROJECT, check_environment

from coevobench.cli.run import launch_run
from coevobench.config.loader import load_config


def main() -> int:
    from coevobench.config.cloud_env import load_env

    load_env(PROJECT / ".local/appworld-cloud.env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=["swe", "bfcl", "tb", "tau", "appworld"])
    parser.add_argument("--image")
    parser.add_argument("--base-dir", default=os.environ.get("BASE_MODEL_DIR", ""))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1,2,3"))
    args = parser.parse_args()
    os.chdir(PROJECT)
    failures = [
        check
        for check in check_environment(
            image=args.image or "coevobench-run:latest",
            bench=args.bench,
            base_dir=args.base_dir,
            gpus=args.gpus,
        )
        if not check["ok"]
    ]
    if failures:
        for check in failures:
            print(f"[MISSING] {check['name']}: {check['detail']}")
        return 1
    cfg = load_config(PROJECT)
    if not (os.environ.get("RESEARCHER_MODEL") or cfg.researcher.model):
        parser.error("set RESEARCHER_MODEL or researcher.model to an available model")
    os.environ["RESEARCHER_BASE_URL"] = (
        os.environ.get("RESEARCHER_BASE_URL") or cfg.researcher.base_url
    )
    result = launch_run(
        bench=args.bench,
        researcher_model=os.environ.get("RESEARCHER_MODEL") or cfg.researcher.model,
        researcher_cli=cfg.researcher.cli,
        budget_hours=cfg.budget.hours,
        gpus=tuple(int(item) for item in args.gpus.split(",")),
        image=args.image or "coevobench-run:latest",
        runhost=cfg.runhost.provider,
        base_dir=str(Path(args.base_dir).resolve()),
    )
    print(f"Started {result['run_id']}: {result['workspace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
