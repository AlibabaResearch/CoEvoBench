"""Check local AppWorld assets and selected cloud configuration without exposing secrets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from coevobench.bench.appworld.assets import PROJECT, python_path, verify_source
from coevobench.bench.appworld.runtime import WorldSession
from coevobench.bench.appworld.tasks import load_tasks
from coevobench.config.cloud_env import load_env
from coevobench.core.protocol.protocol_assets import frozen_splits
from coevobench.runtime.modal_assets import checkpoint_manifest
from coevobench.runtime.modal_platform import SDK_VERSION, credentials_configured


def checks(*, world_smoke=False, cloud=False):
    rows = []

    def check(name, callback):
        try:
            value = callback()
            if value is False:
                raise ValueError("not configured")
            rows.append({"name": name, "ok": True})
        except Exception as exc:
            rows.append({"name": name, "ok": False, "detail": str(exc)[:300]})

    check("fixed AppWorld source", verify_source)
    check("129 tasks / 27 Visible / 51 Selection / 51 Test", lambda: len(load_tasks()) == 129)
    check("AppWorld Python", lambda: python_path().is_file())
    check(
        "base weights and shard index",
        lambda: checkpoint_manifest(
            Path(os.environ.get("BASE_MODEL_DIR", Path.home() / "models/Qwen3.5-9B-Base"))
        ),
    )
    if world_smoke:

        def smoke():
            with tempfile.TemporaryDirectory(prefix="ceb-appworld-check-") as td:
                world = WorldSession(frozen_splits("appworld").visible[0], Path(td) / "world")
                try:
                    result = world.call("execute", code="print(1 + 1)")
                    assert result["observation"].strip() == "2"
                    assert world.call("evaluate")["tgc"] is False
                finally:
                    world.close()

        check("real world execution and official no-op grading", smoke)
    if cloud:
        check(
            "Modal SDK " + SDK_VERSION,
            lambda: (
                __import__("importlib.metadata", fromlist=["version"]).version("modal")
                == SDK_VERSION
            ),
        )
        check("Modal operator credentials", credentials_configured)
        for name in (
            "CEB_MODAL_RESEARCHER_IMAGE",
            "CEB_MODAL_GPU_IMAGE",
            "CEB_APPWORLD_POLICY_IMAGE",
        ):
            check(name, lambda name=name: os.environ.get(name, "").startswith("modal:im-"))
        for name in (
            "CEB_MODAL_BASE_VOLUME",
            "RESEARCHER_MODEL",
            "RESEARCHER_BASE_URL",
            "RESEARCHER_TOKEN",
            "TEACHER_MODEL",
            "TEACHER_BASE_URL",
            "TEACHER_TOKEN",
        ):
            check(name, lambda name=name: bool(os.environ.get(name)))
        for name in ("CEB_APPWORLD_SANDBOX", "CEB_APPWORLD_SERVE"):
            check(name, lambda name=name: os.environ.get(name) == "modal")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-smoke", action="store_true")
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()
    load_env(PROJECT / ".local/appworld-cloud.env")
    os.chdir(PROJECT)
    rows = checks(world_smoke=args.world_smoke, cloud=args.cloud)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
