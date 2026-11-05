"""Verify prepared images and base Volume using short CPU sandboxes; no GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import tempfile
import time
from pathlib import Path

from coevobench.config.cloud_env import load_env
from coevobench.runtime.modal_platform import ModalPlatform
from coevobench.runtime.run_resources import register, stop_resources
from coevobench.runtime.sandbox.modal import ModalSandbox

PROJECT = Path(__file__).resolve().parents[1]
BASE_CHECK = """
import hashlib,json,pathlib
root=pathlib.Path('/base')
manifest=json.loads((root/'ceb-base-manifest.json').read_text())
for row in manifest['shards']+manifest['metadata']:
    assert pathlib.Path(row['file']).name==row['file']
    path=root/row['file']
    assert path.stat().st_size==row['bytes'], row['file']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'], row['file']
print(json.dumps(manifest))
"""


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    load_env(PROJECT / ".local/appworld-cloud.env")
    assets = json.loads((PROJECT / ".local/modal-appworld-assets.json").read_text())
    parent = PROJECT / ".local/modal-asset-verification"
    parent.mkdir(parents=True, exist_ok=True)
    owner = Path(tempfile.mkdtemp(dir=parent))
    deadline = time.time() + 1200
    (owner / "lifecycle.json").write_text(json.dumps({"phase": "running", "deadline_at": deadline}))
    secrets = owner / ".cloud-secrets.json"
    with secrets.open("x") as stream:
        secrets.chmod(0o600)
        json.dump(
            {k: v for k, v in os.environ.items() if k.startswith(("MODAL_", "CEB_MODAL_"))},
            stream,
        )
    platform = ModalPlatform()
    report = {"ok": False, "gpu_allocated": False, "images": {}}
    commands = {
        "researcher": (
            'id researcher && codex --version '
            '&& python3 -c "from coevobench.research.client import Client; '
            "print('broker client import OK')\""
        ),
        "policy": 'python3 -c "import jinja2; print(jinja2.__version__)"',
        "gpu": (
            "id cebtrain && id cebserve && nvcc --version "
            "&& uv pip check --python /opt/venv-train/bin/python "
            "&& uv pip check --python /opt/venv-serve/bin/python "
            '&& /opt/venv-train/bin/python -c "import torch,transformers,peft,trl; '
            "print('training imports OK')\" "
            '&& /opt/venv-serve/bin/python -c "import importlib.metadata as m; '
            "print('vllm='+m.version('vllm'))\""
        ),
    }
    print("Asset verification receipts: " + str(owner), flush=True)
    try:
        for kind, command in commands.items():
            key = (
                "CEB_APPWORLD_POLICY_IMAGE"
                if kind == "policy"
                else f"CEB_MODAL_{kind.upper()}_IMAGE"
            )
            mounts = {}
            if kind == "gpu":
                mounts["/base"] = platform.volume(
                    assets["CEB_MODAL_BASE_VOLUME"]
                ).with_mount_options(read_only=True)
            raw = platform.create(
                assets[key],
                cpu=2,
                memory=8,
                ttl=max(1, int(deadline - time.time())),
                network=False,
                volumes=mounts,
            )
            box = ModalSandbox(raw, owner_dir=owner)
            try:
                register(owner, "modal", box.id)
                result = box.run(command, timeout_s=180)
                report["images"][kind] = {"image": assets[key], **result}
                if result["exit_code"]:
                    raise RuntimeError(kind + " image verification failed")
                print(kind + " image verified", flush=True)
                if kind == "gpu":
                    result = box.run("python3 -c " + shlex.quote(BASE_CHECK), timeout_s=600)
                    if result["exit_code"] or json.loads(result["text"]) != assets["base_manifest"]:
                        raise RuntimeError("remote base Volume differs from local manifest")
                    report["base_volume"] = {
                        "name": assets["CEB_MODAL_BASE_VOLUME"],
                        "sha256_verified": True,
                        "shards": len(assets["base_manifest"]["shards"]),
                    }
                    print("Base Volume SHA-256 verified", flush=True)
            finally:
                box.kill()
        report["ok"] = True
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        raise
    finally:
        (owner / "report.json").write_text(json.dumps(report, indent=2))
        stop_resources(owner)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
