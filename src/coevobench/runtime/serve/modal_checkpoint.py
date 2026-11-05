"""Stage operator checkpoints in a Volume before allocating a final-evaluation GPU."""

from __future__ import annotations

import hashlib
import json
import shlex
import uuid
from pathlib import Path

from coevobench.runtime.sandbox.base import FileWrite


def file_manifest(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint must be a regular directory")
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint symlinks are not allowed")
        if path.is_file():
            with path.open("rb") as stream:
                manifest[str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                }
    if "config.json" not in manifest:
        raise ValueError("checkpoint is missing config.json")
    return manifest


def stage_checkpoint(owner: Path, model: Path, platform) -> dict:
    """Upload through the bulk Volume API, without starting any compute resources."""
    if (owner / "modal-gpu.json").exists():
        raise ValueError("stage a checkpoint before allocating GPUs")
    model = model.resolve()
    manifest = file_manifest(model)
    record_path = owner / "modal-candidate.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["model_path"] == str(model) and record["manifest"] == manifest:
            if record.get("file_mode") != 0o444:
                platform.upload_directory(
                    platform.volume(record["volume"]), model, "/", file_mode=0o444
                )
                if file_manifest(model) != manifest:
                    raise ValueError("checkpoint changed during permission normalization")
                record["file_mode"] = 0o444
                record_path.write_text(json.dumps(record, indent=2))
            return record
        raise ValueError("staged checkpoint changed; use a new evaluation owner")
    volume_name = "ceb-candidate-" + uuid.uuid4().hex
    volume = platform.volume(volume_name, create=True)
    # The serving user must be able to read a root-owned frozen checkpoint.
    # Normalize uploaded permissions while retaining the read-only mount.
    platform.upload_directory(volume, model, "/", file_mode=0o444)
    if file_manifest(model) != manifest:
        raise ValueError("checkpoint changed during upload")
    record = {
        "model_path": str(model),
        "volume": volume_name,
        "manifest": manifest,
        "file_mode": 0o444,
    }
    record_path.write_text(json.dumps(record, indent=2))
    return record


_VERIFY = """
import hashlib,json,pathlib,sys
root=pathlib.Path('/candidate')
expected=json.loads(pathlib.Path(sys.argv[1]).read_text())['manifest']
actual={}
for p in sorted(root.rglob('*')):
    assert not p.is_symlink(), 'checkpoint symlink'
    if p.is_file():
        with p.open('rb') as f:
            actual[str(p.relative_to(root))]={'bytes':p.stat().st_size,
                'sha256':hashlib.file_digest(f,'sha256').hexdigest()}
assert actual==expected, 'staged checkpoint differs from frozen files'
print(json.dumps({'files':len(actual),'bytes':sum(x['bytes'] for x in actual.values()),
                  'sha256_verified':True}))
"""


def staged_path(owner: Path, model: Path, box, directory: str) -> str | None:
    record_path = owner / "modal-candidate.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if record["model_path"] != str(model.resolve()):
        return None
    allocation = json.loads((owner / "modal-gpu.json").read_text())
    if allocation.get("candidate_volume") != record["volume"]:
        raise ValueError("GPU does not mount the prepared candidate Volume")
    if file_manifest(model) != record["manifest"]:
        raise ValueError("checkpoint changed after Volume preparation")
    path = directory + "/candidate-manifest.json"
    box.write_files([FileWrite(path, json.dumps(record), 0o600)])
    result = box.run("python3 -c " + shlex.quote(_VERIFY) + " " + path, timeout_s=600)
    if result["exit_code"]:
        raise ValueError("Modal candidate Volume checksum verification failed")
    (owner / "modal-candidate-verified.json").write_text(result["text"])
    return "/candidate"
