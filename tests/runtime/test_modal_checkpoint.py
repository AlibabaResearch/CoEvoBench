"""Frozen checkpoint staging must precede GPU allocation and reject changed bytes."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from coevobench.runtime.serve.modal_checkpoint import stage_checkpoint, staged_path


@pytest.fixture
def staged(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"checkpoint")
    owner = tmp_path / "owner"
    owner.mkdir()
    remote = tmp_path / "remote"
    uploads = []

    def upload(volume, local, target, *, file_mode=None):
        assert not (owner / "modal-gpu.json").exists()
        uploads.append(target)
        assert file_mode == 0o444
        shutil.copytree(local, remote, dirs_exist_ok=True)
        for path in remote.rglob("*"):
            if path.is_file():
                path.chmod(file_mode)

    platform = SimpleNamespace(volume=lambda *a, **kw: object(), upload_directory=upload)
    record = stage_checkpoint(owner, model, platform)
    return owner, model, remote, platform, record, uploads


def test_bulk_staging_precedes_gpu_and_reuses_unchanged_record(staged):
    owner, model, remote, platform, record, uploads = staged
    assert stage_checkpoint(owner, model, platform) == record
    assert uploads == ["/"]
    assert (remote / "weights.safetensors").read_bytes() == b"checkpoint"
    (model / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        stage_checkpoint(owner, model, platform)


def test_staging_refuses_allocated_gpu(staged):
    owner, model, _, platform, _, _ = staged
    (owner / "modal-gpu.json").write_text("{}")
    with pytest.raises(ValueError, match="before allocating"):
        stage_checkpoint(owner, model, platform)


@pytest.mark.skipif(os.geteuid() != 0, reason="stages read-only trees as root (Linux)")
def test_existing_upload_is_upgraded_without_changing_checkpoint_bytes(staged):
    owner, model, remote, platform, record, uploads = staged
    for path in model.iterdir():
        path.chmod(0o400)
    record.pop("file_mode")
    (owner / "modal-candidate.json").write_text(json.dumps(record))
    updated = stage_checkpoint(owner, model, platform)
    assert uploads == ["/", "/"]
    assert updated["manifest"] == record["manifest"]
    assert updated["volume"] == record["volume"]
    assert updated["file_mode"] == 0o444
    assert (remote / "config.json").stat().st_mode & 0o777 == 0o444
    assert (model / "config.json").stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize("tampered", [False, True])
@pytest.mark.skipif(os.geteuid() != 0, reason="stages read-only trees as root (Linux)")
def test_staged_mount_checks_remote_bytes(staged, tampered):
    owner, model, remote, _, record, _ = staged
    (owner / "modal-gpu.json").write_text(json.dumps({"candidate_volume": record["volume"]}))
    if tampered:
        (remote / "weights.safetensors").write_bytes(b"corrupt")

    class Box:
        def write_files(self, files):
            for entry in files:
                Path(entry.path).write_text(entry.data)

        def run(self, command, **kwargs):
            import shlex

            args = shlex.split(command)
            args[2] = args[2].replace("'/candidate'", repr(str(remote)))
            result = subprocess.run(args, capture_output=True, text=True)
            return {"exit_code": result.returncode, "text": result.stdout}

    if tampered:
        with pytest.raises(ValueError, match="checksum"):
            staged_path(owner, model, Box(), str(owner))
    else:
        assert staged_path(owner, model, Box(), str(owner)) == "/candidate"
        assert json.loads((owner / "modal-candidate-verified.json").read_text())["sha256_verified"]
