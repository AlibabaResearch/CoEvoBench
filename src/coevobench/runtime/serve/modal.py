"""vLLM in the run's Modal GPU sandbox, behind an authenticated operator bridge."""

from __future__ import annotations

import contextlib
import json
import shlex
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from coevobench.runtime.gpu.modal_jobs import ModalGpuJobs, ModalGpuSession
from coevobench.runtime.run_resources import remaining_seconds
from coevobench.runtime.sandbox.base import FileWrite
from coevobench.runtime.serve.modal_checkpoint import stage_checkpoint, staged_path
from coevobench.runtime.serve.vllm import ServeShape, _build_cmd

_WORKER = r"""
import fcntl,json,os,subprocess,sys,time
root=sys.argv[1]; data=json.load(open(root+"/serve.json"))
env={k:v for k,v in os.environ.items()
     if not any(x in k.upper() for x in ("KEY","TOKEN","SECRET","PASSWORD"))}
env.update(data["env"],CUDA_VISIBLE_DEVICES=data.get("gpus","0,1,2,3"),HOME="/home/cebserve")
with open(root+"/lock","a") as lock, open(root+"/vllm.log","w") as log:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if os.path.exists(root+"/cancelled"): sys.exit(0)
    p=subprocess.Popen(["runuser","-u","cebserve","--preserve-environment","--",*data["command"]],
                       env=env,stdout=log,stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL,start_new_session=True)
    start=open(f"/proc/{p.pid}/stat").read().rsplit(")",1)[1].split()[19]
    open(root+"/pid.json","w").write(json.dumps({"pid":p.pid,"start":start}))
    fcntl.flock(lock,fcntl.LOCK_UN)
    try: p.wait(timeout=max(1,data["deadline"]-time.time()))
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill","-KILL","-u","cebserve"])
        p.wait(timeout=15)
    # Holding the launch lock makes cancellation wait for the final sweep.
    fcntl.flock(lock,fcntl.LOCK_EX)
    subprocess.run(["pkill","-KILL","-u","cebserve"])
    open(root+"/finished","w").close()
"""

_STOP = r"""
import fcntl,json,os,signal,subprocess,sys,time
root=sys.argv[1]
with open(root+"/lock","a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 open(root+"/cancelled","w").close()
 subprocess.run(["pkill","-KILL","-u","cebserve"])
 # If the worker has not started, cancellation prevents its later launch.
 launched=os.path.isfile(root+"/pid.json")
if launched:
 for _ in range(100):
  if os.path.isfile(root+"/finished"): break
  time.sleep(.1)
 else: raise RuntimeError("vLLM worker did not finish its cleanup")
"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ModalServeEndpoint:
    def __init__(self, session, box, directory, model_name, port, lock, *, owns_gpu=False):
        self.session, self.box, self.directory = session, box, directory
        self.model_name = model_name
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._lock, self._owns_gpu = lock, owns_gpu
        self._closed = False
        self._server = None

    def start_bridge(self, port):
        creds = self.box._sandbox.create_connect_token(port=8000)
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def send_error(self, *args, **kwargs):
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    super().send_error(*args, **kwargs)

            def do_GET(self):  # noqa: N802
                self.forward()

            def do_POST(self):  # noqa: N802
                self.forward()

            def forward(self):
                if self.path not in ("/v1/models", "/v1/chat/completions", "/v1/completions"):
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(400)
                    return
                if not 0 <= length <= 16 * 1024 * 1024:
                    self.send_error(413)
                    return
                request = urllib.request.Request(
                    creds.url.rstrip("/") + self.path,
                    data=self.rfile.read(length) if self.command == "POST" else None,
                    headers={
                        "Authorization": "Bearer " + creds.token,
                        "Content-Type": "application/json",
                    },
                    method=self.command,
                )
                try:
                    response = urllib.request.build_opener(_NoRedirect).open(
                        request, timeout=5 if self.command == "GET" else 600
                    )
                except urllib.error.HTTPError as exc:
                    response = exc
                except Exception:
                    self.send_error(502, "Modal model connection unavailable")
                    return
                with response:
                    body = response.read(16 * 1024 * 1024 + 1)
                    if len(body) > 16 * 1024 * 1024:
                        self.send_error(502, "model response exceeds limit")
                        return
                    self.send_response(response.status)
                    self.send_header(
                        "Content-Type", response.headers.get("Content-Type", "application/json")
                    )
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        deadline = time.monotonic() + min(600, remaining_seconds(self.session.owner))
        next_worker_check = 0.0
        while time.monotonic() < deadline:
            if endpoint.box._sandbox.poll() is not None:
                raise RuntimeError("Modal GPU sandbox stopped during model startup")
            if time.monotonic() >= next_worker_check:
                finished = self.box.run("test -f " + self.directory + "/finished", timeout_s=10)
                if finished["exit_code"] == 0:
                    raise RuntimeError(
                        "Modal vLLM worker exited during startup; inspect startup log"
                    )
                next_worker_check = time.monotonic() + 15
            try:
                with urllib.request.urlopen(self.base_url + "/models", timeout=10) as response:
                    if any(
                        item["id"] == self.model_name
                        for item in json.load(response).get("data", [])
                    ):
                        return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("Modal vLLM did not become ready within the startup deadline")

    def stop(self):
        if self._closed:
            return
        try:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
            result = self.box.run(
                "python3 -c " + shlex.quote(_STOP) + " " + self.directory, timeout_s=30
            )
            if result["exit_code"]:
                raise RuntimeError("Modal model process cleanup failed")
        except BaseException:
            # Never release the allocation with uncertain old GPU processes.
            (self.session.owner / "modal-gpu-unusable").touch()
            self.session.stop()
            raise
        finally:
            try:
                if self._owns_gpu:
                    self.session.stop()
            finally:
                self._lock.__exit__(None, None, None)
                self._closed = True


class ModalServeRuntime:
    name = "modal"

    def __init__(self, *, platform=None, template_roots=(), **_kwargs):
        self.platform = platform
        self.template_roots = tuple(Path(p).resolve() for p in template_roots)

    def serve(
        self, model_path, port, *, served_name="qwen3.5-9b", shape=None, gpu_ids=(), owner_dir=None
    ):
        owns_gpu = owner_dir is None
        shape = shape or ServeShape()
        if any(
            type(value) is not int or value < 1
            for value in (shape.tensor_parallel, shape.data_parallel)
        ):
            raise ValueError("serving parallelism must use positive integers")
        required_gpus = shape.tensor_parallel * shape.data_parallel
        gpu_count = (len(gpu_ids) if gpu_ids else required_gpus) if owns_gpu else 4
        if gpu_count not in (1, 2, 4) or required_gpus > gpu_count:
            raise ValueError("serving shape must fit the allocated 1, 2 or 4 GPUs")
        if gpu_ids and tuple(gpu_ids) != tuple(range(gpu_count)):
            raise ValueError("GPU IDs must match the allocation starting at zero")
        if owner_dir is None:
            parent = Path.home() / ".ceb/modal-evaluations"
            parent.mkdir(parents=True, exist_ok=True)
            owner_dir = Path(tempfile.mkdtemp(dir=parent))
        session = ModalGpuSession(Path(owner_dir), platform=self.platform)
        if owns_gpu:
            stage_checkpoint(session.owner, Path(model_path), session.platform)
            (owner_dir / "lifecycle.json").write_text(
                json.dumps(
                    {
                        "phase": "running",
                        "deadline_at": time.time() + 86400,
                    }
                )
            )
            session.create(gpu_count=gpu_count)
        lock = session.exclusive()
        lock.__enter__()
        endpoint = None
        try:
            remaining_seconds(session.owner)
            if ModalGpuJobs(session.owner, platform=session.platform).status()["status"] in (
                "running",
                "starting",
            ):
                raise ValueError("stop or finish training before evaluation")
            box = session.box()
            allocation = json.loads(session.record.read_text())
            if allocation["gpu_count"] != gpu_count:
                raise ValueError("serving allocation differs from the requested GPU count")
            directory = "/persist/operator/models/" + uuid.uuid4().hex
            result = box.run("mkdir -p " + directory)
            if result["exit_code"]:
                raise RuntimeError("cannot create private serving directory")
            model_path = Path(model_path).resolve()
            base_record = session.owner / "base-model.json"
            base_path = (
                Path(json.loads(base_record.read_text())["path"]).resolve()
                if base_record.is_file()
                else None
            )
            remote_weights = directory + "/weights"
            if model_path == base_path:
                remote_weights = "/base"
            elif staged := staged_path(session.owner, model_path, box, directory):
                remote_weights = staged
            else:
                for path in sorted(model_path.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("candidate checkpoint must not contain symlinks")
                    if path.is_file():
                        box._sandbox.filesystem.copy_from_local(
                            path, directory + "/weights/" + str(path.relative_to(model_path))
                        )
                if box.run(
                    "chown -R cebserve:cebserve "
                    + remote_weights
                    + " && chmod -R a-w "
                    + remote_weights
                )["exit_code"]:
                    raise RuntimeError("failed to protect serving checkpoint")
            if box.run(
                "chgrp cebserve /persist/operator /persist/operator/models "
                "&& chmod 710 /persist/operator /persist/operator/models"
            )["exit_code"]:
                raise RuntimeError("cannot grant serving access to the private model directory")
            if shape and shape.chat_template:
                template = Path(shape.chat_template)
                if template.is_file():
                    template = template.resolve()
                    roots = (model_path, *self.template_roots)
                    if not any(template.is_relative_to(root) for root in roots):
                        raise ValueError(
                            "chat template must be inside the candidate model or harness"
                        )
                    box._sandbox.filesystem.copy_from_local(
                        template, directory + "/chat_template.jinja"
                    )
                    shape = replace(shape, chat_template=directory + "/chat_template.jinja")
            command, env = _build_cmd(
                remote_weights,
                8000,
                served_name=served_name,
                venv_bin="/opt/venv-serve/bin",
                shape=shape,
            )
            # Modal Connect reaches the sandbox interface, not its loopback device.
            command[command.index("--host") + 1] = "0.0.0.0"
            config = {
                "command": command,
                "env": env,
                "deadline": time.time() + remaining_seconds(session.owner),
                "gpus": ",".join(map(str, range(gpu_count))),
            }
            (session.owner / "modal-serve.json").write_text(
                json.dumps(
                    {
                        "gpu_count": gpu_count,
                        "tensor_parallel": shape.tensor_parallel,
                        "data_parallel": shape.data_parallel,
                        "max_model_len": shape.max_model_len,
                        "command": command,
                    },
                    indent=2,
                )
            )
            box.write_files(
                [
                    FileWrite(directory + "/serve.json", json.dumps(config), 0o600),
                    FileWrite(directory + "/worker.py", _WORKER, 0o600),
                ]
            )
            endpoint = ModalServeEndpoint(
                session, box, directory, served_name, port, lock, owns_gpu=owns_gpu
            )
            result = box.run(
                "nohup python3 "
                + directory
                + "/worker.py "
                + directory
                + " >/dev/null 2>&1 </dev/null &"
            )
            if result["exit_code"]:
                raise RuntimeError("Modal vLLM worker failed to start")
            endpoint.start_bridge(port)
            return endpoint
        except BaseException:
            if endpoint is not None:
                with contextlib.suppress(Exception):
                    log = box.read_file(directory + "/vllm.log")
                    (session.owner / ("vllm-startup-" + Path(directory).name + ".log")).write_text(
                        log
                    )
                endpoint.stop()
            else:
                lock.__exit__(None, None, None)
                if owns_gpu:
                    with contextlib.suppress(Exception):
                        session.stop()
            raise
