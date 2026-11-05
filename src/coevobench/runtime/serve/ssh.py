"""Operator-controlled vLLM on a rented GPU host, reached through SSH forwarding."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path

from coevobench.runtime.serve.vllm import ServeEndpoint, ServeShape, _build_cmd


class SshHost:
    def __init__(self, settings=None):
        settings = os.environ if settings is None else settings
        self.host = settings.get("CEB_VAST_HOST", "")
        self.user = settings.get("CEB_VAST_USER", "root")
        self.port = int(settings.get("CEB_VAST_PORT", "22"))
        self.key = Path(settings.get("CEB_VAST_KEY", "/nonexistent")).expanduser()
        self.root = settings.get("CEB_VAST_ROOT", "/workspace/coevobench")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", self.host):
            raise ValueError("set CEB_VAST_HOST to the SSH hostname")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", self.user) or not self.key.is_file():
            raise ValueError("set a valid CEB_VAST_USER and CEB_VAST_KEY file")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", self.root) or ".." in Path(self.root).parts:
            raise ValueError("CEB_VAST_ROOT must be a simple absolute directory")
        if not 1 <= self.port <= 65535:
            raise ValueError("invalid CEB_VAST_PORT")

    @property
    def options(self):
        return [
            "ssh",
            "-i",
            str(self.key),
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]

    @property
    def target(self):
        return f"{self.user}@{self.host}"

    def run(self, argv, *, data=None, timeout=120):
        return subprocess.run(
            [*self.options, self.target, shlex.join(map(str, argv))],
            input=data,
            capture_output=True,
            timeout=timeout,
            check=True,
        )

    def sync(self, local: Path, remote: str, *, download=False, exclude=()):
        if not remote.startswith(self.root + "/"):
            raise ValueError("remote transfer escapes configured root")
        local.mkdir(parents=True, exist_ok=True)
        self.run(["mkdir", "-p", remote])
        source, target = str(local.resolve()) + "/", f"{self.target}:{remote}/"
        if download:
            source, target = target, source
        command = [
            "rsync",
            "-rt",
            "--no-links",
            "--no-specials",
            "--no-devices",
            "--protect-args",
            "-e",
            shlex.join(self.options),
        ]
        for pattern in exclude:
            command += ["--exclude", pattern]
        subprocess.run([*command, source, target], capture_output=True, check=True, timeout=7200)


_WATCHDOG = r"""
import os,signal,sys,time
pid=int(sys.argv[1]); expected=sys.argv[2]; deadline=float(sys.argv[3])
while True:
    try:
        row=open(f"/proc/{pid}/stat").read().rsplit(")",1)[1].split()
        if row[19]!=expected or row[0]=="Z": break
        if time.time()>=deadline:
            if os.getpgid(pid)==pid: os.killpg(pid,signal.SIGKILL)
            break
    except (FileNotFoundError,ProcessLookupError): break
    time.sleep(.5)
"""

_START = r"""
import json,os,subprocess,sys,time
config=json.load(sys.stdin)
os.makedirs(config["directory"],mode=0o700,exist_ok=True)
env={k:v for k,v in os.environ.items()
     if not any(s in k.upper() for s in ("TOKEN","SECRET","KEY","PASSWORD"))}
env["CUDA_VISIBLE_DEVICES"]=config["gpus"]
with open(config["directory"]+"/vllm.log","w") as log:
    proc=subprocess.Popen(config["command"],env=env,stdin=subprocess.DEVNULL,stdout=log,
                          stderr=subprocess.STDOUT,start_new_session=True)
start=open(f"/proc/{proc.pid}/stat").read().rsplit(")",1)[1].split()[19]
subprocess.Popen(["python3","-c",config["watchdog"],str(proc.pid),start,str(config["deadline"])],
                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                 start_new_session=True)
print(json.dumps({"pid":proc.pid,"start":start}),flush=True)
"""

_STOP = r"""
import json,os,signal,sys,time
record=json.load(sys.stdin)
try:
    pid=int(record["pid"])
    start=open(f"/proc/{pid}/stat").read().rsplit(")",1)[1].split()[19]
    if start==record["start"] and os.getpgid(pid)==pid:
        os.killpg(pid,signal.SIGTERM)
        time.sleep(1)
        try: os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError: pass
except FileNotFoundError: pass
"""


def stop_remote(receipt, settings=None):
    host = SshHost(settings)
    host.run(["python3", "-c", _STOP], data=json.dumps(receipt).encode())
    model_dir = receipt.get("model_dir", "")
    if re.fullmatch(re.escape(host.root) + r"/\.operator/models/[0-9a-f]{32}", model_dir):
        host.run(["rm", "-rf", "--", model_dir])


def acquire_lease(run_dir, settings=None):
    """Reserve the dedicated training UID across all runs on this GPU instance."""
    host = SshHost(settings)
    token = uuid.uuid4().hex
    (run_dir / "gpu-lease.json").write_text(json.dumps({"token": token}))
    script = (
        "import os,sys; fd=os.open('/var/lock/coevobench-gpu',"
        "os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.write(fd,sys.argv[1].encode()); os.close(fd)"
    )
    try:
        host.run(["python3", "-c", script, token])
    except Exception:
        raise RuntimeError("Vast instance lease unavailable; stop its existing run first") from None


def release_lease(run_dir, settings=None):
    path = run_dir / "gpu-lease.json"
    if not path.is_file():
        return
    token = json.loads(path.read_text())["token"]
    script = (
        "import pathlib,sys; p=pathlib.Path('/var/lock/coevobench-gpu'); "
        "p.unlink() if p.is_file() and p.read_text()==sys.argv[1] else None"
    )
    SshHost(settings).run(["python3", "-c", script, token])


class RemoteEndpoint(ServeEndpoint):
    def __init__(self, process, base_url, model_name, receipt):
        super().__init__(process, base_url, model_name)
        self.receipt = receipt

    def stop(self):
        try:
            stop_remote(self.receipt)
        finally:
            if self._proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._proc.pid, signal.SIGTERM)
                self._proc.wait(timeout=20)


class SshServeRuntime:
    name = "ssh"

    def __init__(self, **_kwargs):
        self.host = SshHost()

    def serve(
        self,
        model_path: Path,
        port: int,
        *,
        served_name="qwen3.5-9b",
        shape: ServeShape | None = None,
        gpu_ids=(),
        owner_dir=None,
    ):
        from coevobench.runtime.run_resources import register

        model_path = Path(model_path).resolve()
        remote = self.host.root + "/.operator/models/" + uuid.uuid4().hex
        self.host.run(["mkdir", "-p", self.host.root + "/.operator"])
        self.host.run(["chmod", "700", self.host.root + "/.operator"])
        self.host.sync(model_path, remote, exclude=("*.optim", "checkpoint-*", "__pycache__"))
        self.host.run(["chmod", "-R", "a-w", remote])
        remote_bin = os.environ.get("CEB_VAST_VLLM_BIN", self.host.root + "/.venv-serve/bin")
        command, _ = _build_cmd(
            remote, port, served_name=served_name, venv_bin=remote_bin, shape=shape
        )
        deadline = time.time() + 86400
        if owner_dir is not None and (owner_dir / "lifecycle.json").is_file():
            deadline = json.loads((owner_dir / "lifecycle.json").read_text())["deadline_at"]
        if time.time() >= deadline:
            raise ValueError("run ended before remote serving launch")
        response = self.host.run(
            ["python3", "-c", _START],
            data=json.dumps(
                {
                    "command": command,
                    "directory": remote + "-run",
                    "gpus": ",".join(map(str, gpu_ids)) or os.environ.get("GPU_IDS", "0,1,2,3"),
                    "deadline": deadline,
                    "watchdog": _WATCHDOG,
                }
            ).encode(),
        )
        receipt = json.loads(response.stdout)
        receipt["model_dir"] = remote
        tunnel = None
        try:
            register(owner_dir, "ssh-serve", receipt)
            tunnel = subprocess.Popen(
                [
                    *self.host.options,
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-N",
                    "-L",
                    f"127.0.0.1:{port}:127.0.0.1:{port}",
                    self.host.target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            register(owner_dir, "process", tunnel.pid)
            endpoint = RemoteEndpoint(tunnel, f"http://127.0.0.1:{port}/v1", served_name, receipt)
            try:
                endpoint.wait_ready(timeout_s=600)
            except BaseException:
                endpoint.stop()
                raise
            return endpoint
        except BaseException:
            try:
                stop_remote(receipt)
            finally:
                if tunnel is not None and tunnel.poll() is None:
                    tunnel.terminate()
                    tunnel.wait(timeout=20)
            raise
