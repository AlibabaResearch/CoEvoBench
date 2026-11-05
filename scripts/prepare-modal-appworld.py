"""Build AppWorld's three Modal images and upload a verified read-only base Volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from coevobench.config.cloud_env import load_env, update_env
from coevobench.runtime.modal_assets import build_context, checkpoint_manifest
from coevobench.runtime.modal_platform import ModalPlatform, credentials_configured

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".local/appworld-cloud.env")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="local validation only; no cloud resources"
    )
    args = parser.parse_args()
    load_env(args.env_file)
    base = args.base_dir or Path(os.environ.get(
        "BASE_MODEL_DIR", Path.home() / "models/Qwen3.5-9B-Base"
    ))
    manifest = checkpoint_manifest(base, hash_files=not args.check)
    with tempfile.TemporaryDirectory(prefix="ceb-modal-contexts-") as td:
        contexts = {
            kind: build_context(PROJECT, kind, Path(td) / kind)
            for kind in ("researcher", "policy", "gpu")
        }
        if args.check:
            print(
                json.dumps(
                    {
                        "contexts": list(contexts),
                        "base": manifest,
                        "modal_credentials_present": credentials_configured(),
                    },
                    indent=2,
                )
            )
            return 0
        if not credentials_configured():
            raise SystemExit(
                "Fill MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in the private env file first."
            )
        keys = {
            "researcher": "CEB_MODAL_RESEARCHER_IMAGE",
            "policy": "CEB_APPWORLD_POLICY_IMAGE",
            "gpu": "CEB_MODAL_GPU_IMAGE",
        }
        assets = PROJECT / ".local/modal-appworld-assets.json"
        record = json.loads(assets.read_text()) if assets.is_file() else {}
        platform = ModalPlatform()
        assets.parent.mkdir(parents=True, exist_ok=True)
        for kind, context in contexts.items():
            print(f"Building Modal {kind} image...", flush=True)
            image = platform.sdk.Image.from_dockerfile(context / "Dockerfile", context_dir=context)
            with platform.sdk.enable_output():
                built = image.build(platform.app)
            reference = "modal:" + built.object_id
            record[keys[kind]] = reference
            assets.write_text(json.dumps(record, indent=2) + "\n")
            update_env(args.env_file, {keys[kind]: reference})
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        name = "ceb-qwen35-9b-base-" + digest[:16]
        volume = platform.volume(name, create=True)
        print("Uploading verified base checkpoint to Modal Volume...", flush=True)
        # Only model files; never upload HF credentials/cache metadata or the operator checkout.
        with volume.batch_upload(force=True) as batch:
            for path in sorted(base.iterdir()):
                if path.is_file() and not path.name.startswith("."):
                    if path.is_symlink():
                        raise ValueError("materialize base symlinks before upload")
                    batch.put_file(str(path), "/" + path.name)
            manifest_file = Path(td) / "ceb-base-manifest.json"
            manifest_file.write_text(json.dumps(manifest, sort_keys=True))
            batch.put_file(str(manifest_file), "/ceb-base-manifest.json")
        record.update(CEB_MODAL_BASE_VOLUME=name, base_manifest=manifest)
        assets.write_text(json.dumps(record, indent=2) + "\n")
        update_env(args.env_file, {"CEB_MODAL_BASE_VOLUME": name})
        print("Prepared three image references and base Volume; private env file updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
