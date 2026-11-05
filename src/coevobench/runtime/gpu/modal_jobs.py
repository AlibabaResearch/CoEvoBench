"""One Modal GPU sandbox per run; training and serving share the same allocation."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shlex
import time
import uuid
from pathlib import Path

from coevobench.runtime.gpu.job_worker import WORKER
from coevobench.runtime.modal_platform import ModalPlatform, remaining_seconds
from coevobench.runtime.modal_transfer import download_tree
from coevobench.runtime.run_resources import register
from coevobench.runtime.sandbox.base import FileWrite
from coevobench.runtime.sandbox.modal import ModalSandbox
from coevobench.runtime.workspace_transfer import FROZEN, pack_workspace

_VERIFY_BASE = r"""
import hashlib,json,pathlib
root=pathlib.Path("/base")
manifest=json.loads((root/"ceb-base-manifest.json").read_text())
assert len(manifest["shards"])==4
for row in manifest["shards"] + manifest.get("metadata",[]):
    assert pathlib.Path(row["file"]).name==row["file"]
    path=root/row["file"]
    assert not path.is_symlink() and path.stat().st_size==row["bytes"]
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==row["sha256"]
print(json.dumps(manifest))
"""


class ModalGpuSession:
    def __init__(self, owner: Path, *, platform=None, settings=None):
        self.owner = Path(owner)
        self.settings = dict(os.environ if settings is None else settings)
        self._platform = platform
        self.record = self.owner / "modal-gpu.json"

    @property
    def platform(self):
        if self._platform is None:
            self._platform = ModalPlatform(self.settings)
        return self._platform

    @contextlib.contextmanager
    def exclusive(self):
        with (self.owner / "modal-gpu.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "GPU allocation is busy; training and evaluation are exclusive"
                ) from exc
            yield

    def create(self, *, gpu_count=4):
        if type(gpu_count) is not int or gpu_count not in (1, 2, 4):
            raise ValueError("Modal GPU allocation must use 1, 2 or 4 cards")
        with self.exclusive():
            if self.record.exists():
                if json.loads(self.record.read_text())["gpu_count"] != gpu_count:
                    raise ValueError("cannot resize an existing Modal GPU allocation")
                return self.box()
            ttl = remaining_seconds(self.owner)
            base_name = self.settings["CEB_MODAL_BASE_VOLUME"]
            base = self.platform.volume(base_name)
            image = self.settings["CEB_MODAL_GPU_IMAGE"]
            profile = self.settings.get("CEB_MODAL_GPU_TYPE", "H100!")
            if ":" in profile or not profile.rstrip("!").replace("-", "").isalnum():
                raise ValueError("CEB_MODAL_GPU_TYPE must be one GPU type without a count")
            run_volume = "ceb-run-" + uuid.uuid4().hex
            output = self.platform.volume(run_volume, create=True)
            volumes = {"/base": base.with_mount_options(read_only=True), "/persist": output}
            candidate_volume = None
            candidate_record = self.owner / "modal-candidate.json"
            if candidate_record.is_file():
                candidate_volume = json.loads(candidate_record.read_text())["volume"]
                volumes["/candidate"] = self.platform.volume(candidate_volume).with_mount_options(
                    read_only=True
                )
            raw = self.platform.create(
                image,
                cpu=16,
                memory=128,
                ttl=ttl,
                network=True,
                gpu=f"{profile}:{gpu_count}",
                volumes=volumes,
            )
            box = ModalSandbox(raw, owner_dir=self.owner)
            self.record.write_text(
                json.dumps(
                    {
                        "sandbox_id": box.id,
                        "base_volume": base_name,
                        "output_volume": run_volume,
                        "gpu_type": profile,
                        "gpu_count": gpu_count,
                        "candidate_volume": candidate_volume,
                    }
                )
            )
            try:
                register(self.owner, "modal", box.id)
                result = box.run(
                    "id cebtrain && id cebserve && test -x /opt/venv-train/bin/python "
                    "&& test -x /opt/venv-serve/bin/vllm && test -f /base/config.json "
                    "&& mkdir -p /persist/jobs /persist/operator && chmod 700 /persist/operator",
                    timeout_s=120,
                )
                if result["exit_code"]:
                    raise RuntimeError("Modal GPU image or base Volume is incomplete")
                result = box.run("python3 -c " + shlex.quote(_VERIFY_BASE), timeout_s=600)
                if result["exit_code"]:
                    raise RuntimeError("Modal base Volume checksum validation failed")
                remote_manifest = json.loads(result["text"])
                expected = self.owner / "base-manifest.json"
                if expected.is_file() and json.loads(expected.read_text()) != remote_manifest:
                    raise RuntimeError("Modal base Volume differs from the local base checkpoint")
                (self.owner / "modal-base-verified.json").write_text(json.dumps(remote_manifest))
                return box
            except BaseException:
                box.kill()
                raise

    def box(self):
        if (self.owner / "modal-gpu-unusable").exists():
            raise RuntimeError("GPU cleanup is uncertain; stop this run before allocating again")
        data = json.loads(self.record.read_text())
        box = ModalSandbox(self.platform.get(data["sandbox_id"]), owner_dir=self.owner)
        if box._sandbox.poll() is not None:
            raise RuntimeError("Modal GPU sandbox ended; recover artifacts before restarting a run")
        return box

    def stop(self):
        if self.record.is_file():
            data = json.loads(self.record.read_text())
            self.platform.terminate(data["sandbox_id"])


class ModalGpuJobs:
    def __init__(self, run_dir: Path, settings=None, *, platform=None):
        self.run_dir = Path(run_dir)
        self.session = ModalGpuSession(self.run_dir, platform=platform, settings=settings)
        self.record = self.run_dir / "gpu-job.json"

    def status(self):
        if not self.record.exists():
            return {"status": "idle"}
        job = json.loads(self.record.read_text())
        if job.get("fetched"):
            return {"status": "completed", "job_id": job["job_id"], "fetched": True}
        code = (
            "import pathlib,json; p=pathlib.Path("
            + repr(job["directory"] + "/status.json")
            + "); print(p.read_text() if p.exists() else json.dumps({'status':'starting'}))"
        )
        result = self.session.box().run("python3 -c " + shlex.quote(code), timeout_s=30)
        if result["exit_code"]:
            raise RuntimeError("GPU job status unavailable")
        return {**json.loads(result["text"]), "job_id": job["job_id"]}

    def submit(self, command: str):
        if not isinstance(command, str) or not command.strip() or len(command) > 65536:
            raise ValueError("GPU command must be nonempty text under 64 KiB")
        with self.session.exclusive():
            remaining_seconds(self.run_dir)
            status = self.status()
            if status["status"] in ("running", "starting"):
                raise ValueError("wait or stop the active GPU training job")
            if self.record.exists() and not status.get("fetched"):
                raise ValueError("fetch the previous GPU job before submitting another")
            box = self.session.box()
            job_id = uuid.uuid4().hex
            directory = "/persist/jobs/" + job_id
            if box.run("mkdir -p " + directory + "/workspace")["exit_code"]:
                raise RuntimeError("cannot prepare GPU job directory")
            box.write_bytes(
                directory + "/workspace.tar.gz",
                pack_workspace(self.run_dir / "workspace", initial=True),
            )
            # Extract only the operator-generated archive. Researcher uploads pass
            # through the strict transfer validator.
            prepare = (
                "import tarfile; tarfile.open("
                + repr(directory + "/workspace.tar.gz")
                + ").extractall("
                + repr(directory + "/workspace")
                + ",filter='data')"
            )
            result = box.run(
                "python3 -c "
                + shlex.quote(prepare)
                + " && chown -R cebtrain:cebtrain "
                + directory
                + "/workspace && ln -s /base "
                + directory
                + "/workspace/base"
            )
            if result["exit_code"]:
                raise RuntimeError("GPU workspace upload failed")
            # The bounded code archive excludes binary artifacts. Restore them at any
            # researcher-owned relative path for continued training.
            workspace = self.run_dir / "workspace"
            for path in workspace.rglob("*"):
                relative = path.relative_to(workspace)
                if relative.parts[0] in FROZEN or any(
                    part in {".git", "__pycache__", ".venv"} for part in relative.parts
                ):
                    continue
                if path.is_symlink():
                    raise ValueError("GPU inputs cannot contain symlinks")
                if path.is_file() and path.suffix in {
                    ".safetensors",
                    ".bin",
                    ".pt",
                    ".pth",
                    ".sqlite",
                    ".db",
                }:
                    box._sandbox.filesystem.copy_from_local(
                        path, directory + "/workspace/" + str(relative)
                    )
            if box.run("chown -R cebtrain:cebtrain " + directory + "/workspace")["exit_code"]:
                raise RuntimeError("GPU input ownership update failed")
            state = json.loads((self.run_dir / "lifecycle.json").read_text())
            config = {
                "command": command,
                "deadline": state["deadline_at"],
                "bin": "/opt/venv-train/bin",
                "gpus": "0,1,2,3",
            }
            teacher_endpoint = self.run_dir / ".teacher-endpoint.json"
            if teacher_endpoint.is_file():
                endpoint = json.loads(teacher_endpoint.read_text())
                config["teacher_env"] = {
                    key: endpoint[key]
                    for key in ("TEACHER_BASE_URL", "TEACHER_API_KEY", "TEACHER_MODEL")
                }
            box.write_files(
                [
                    FileWrite(directory + "/config.json", json.dumps(config), 0o600),
                    FileWrite(directory + "/worker.py", WORKER, 0o600),
                ]
            )
            self.record.write_text(
                json.dumps({"job_id": job_id, "directory": directory, "provider": "modal"})
            )
            result = box.run(
                "nohup python3 "
                + directory
                + "/worker.py "
                + directory
                + " >"
                + directory
                + "/controller.log 2>&1 </dev/null &",
                timeout_s=30,
            )
            if result["exit_code"]:
                raise RuntimeError("GPU worker failed to launch")
            return {"job_id": job_id, "status": "starting"}

    def stop(self):
        if not self.record.exists():
            return {"status": "idle"}
        with self.session.exclusive():
            job = json.loads(self.record.read_text())
            code = (
                "import pathlib,fcntl,subprocess,json; p=pathlib.Path("
                + repr(job["directory"])
                + "); f=(p/'lock').open('a'); fcntl.flock(f,fcntl.LOCK_EX); "
                "(p/'cancelled').touch(); subprocess.run(['pkill','-KILL','-u','cebtrain'])"
            )
            result = self.session.box().run("python3 -c " + shlex.quote(code), timeout_s=30)
            if result["exit_code"]:
                raise RuntimeError("GPU training cleanup failed")
            # The worker writes completed only after its final descendant sweep.
            # Do not expose an idle allocation while an old worker can still kill it.
            for _ in range(50):
                if self.status()["status"] == "completed":
                    return {"status": "completed"}
                time.sleep(0.2)
            raise RuntimeError("GPU worker did not acknowledge cancellation")

    def fetch(self):
        with self.session.exclusive():
            status = self.status()
            if status["status"] in ("running", "starting"):
                raise ValueError("wait for training completion before fetching a stable candidate")
            if not self.record.exists():
                raise ValueError("no GPU job to fetch")
            job = json.loads(self.record.read_text())
            box = self.session.box()
            if box.run("pkill -KILL -u cebtrain; test $? -le 1", timeout_s=30)["exit_code"]:
                raise RuntimeError("cannot stop GPU artifact writers")
            transferred = download_tree(
                box, job["directory"] + "/workspace", self.run_dir / "workspace", excluded=FROZEN
            )
            # Explicit operator-side upload makes the fetched checkpoint durable without
            # installing management credentials in the GPU sandbox.
            data = json.loads(self.session.record.read_text())
            output = self.run_dir / "workspace/out"
            if output.is_dir():
                self.session.platform.upload_directory(
                    self.session.platform.volume(data["output_volume"]),
                    output,
                    "/fetched/" + job["job_id"] + "/out",
                )
            job["fetched"] = True
            self.record.write_text(json.dumps(job))
            return {**status, **transferred, "fetched": True}

    def logs(self):
        if not self.record.exists():
            return {"logs": ""}
        job = json.loads(self.record.read_text())
        result = self.session.box().run("tail -c 16000 " + job["directory"] + "/job.log")
        return {"logs": result["text"]}
