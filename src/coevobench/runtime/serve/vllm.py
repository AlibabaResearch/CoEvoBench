"""vLLM-backed ServeRuntime.

One command builder (`_build_cmd`) is the single source of serving
truth: development evaluation and final evaluation both start servers
through it, keeping the declared serving parameters consistent.

The runtime execs into `vllm serve` (process replacement semantics via
subprocess), waits for readiness on /v1/models, and resolves the real
weights directory by reading the live process's cmdline — the served
alias is never proof of identity.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ServeShape:
    """Serving configuration contract: the decode/throughput knobs a
    candidate may declare (whitelisted). Everything here travels with
    the candidate and is replayed by final evaluation."""

    tensor_parallel: int = 1
    data_parallel: int = 1
    gpu_mem_util: float = 0.85
    max_model_len: int = 32768
    max_num_seqs: int = 0
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    chat_template: str = ""
    # Pass-through vLLM flags the candidate declares (whitelisted by
    # the serve contract; text-only recipes disable multimodal inputs:
    # ["--limit-mm-per-prompt", '{"image":0,"video":0}'] — a pure-text
    # bench must not pay the encoder-cache budget).
    extra_args: tuple[str, ...] = ()


class ServeEndpoint:
    def __init__(
        self, process: subprocess.Popen, base_url: str, model_name: str,
        *, log_file: Path | None = None, shape: ServeShape | None = None,
        owner_dir: Path | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self._proc = process
        self._log_file = log_file
        self._shape = shape
        self._owner_dir = owner_dir

    def _failure(self, message: str) -> RuntimeError:
        """Expose an actionable startup diagnosis; keep raw logs operator-side."""
        tail = ""
        if self._log_file is not None:
            try:
                with self._log_file.open("rb") as stream:
                    stream.seek(max(0, self._log_file.stat().st_size - 32768))
                    tail = stream.read().decode(errors="replace")
            except OSError:
                pass
        if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
            message += "; CUDA out of memory during startup"
            message += "; reduce context/concurrency, disable multimodal inputs, or use more GPUs"
        elif "Free memory on device" in tail:
            message += "; insufficient free GPU memory; stop competing GPU workloads"
        elif "No available memory for the cache blocks" in tail or "KV cache is needed" in tail:
            message += "; insufficient KV cache memory during startup"
            text_only = json.dumps(["--limit-mm-per-prompt", '{"image":0,"video":0}'])
            message += (f"; reduce max_len/max_num_seqs, use extra_args={text_only} for text-only "
                        "serving, or increase gpu_mem_util within available memory")
            estimate = re.search(r"estimated maximum model length is (\d+)", tail)
            if estimate:
                message += f"; estimated max_len for this recipe: {estimate[1]}"
        elif "torch/_dynamo" in tail and "'NoneType' object has no attribute 'size'" in tail:
            message += "; torch.compile failed during startup; try enforce_eager=true"
        elif "C compiler" in tail or "gcc: not found" in tail:
            message += "; C compiler missing in the serving environment"
        elif "unrecognized arguments" in tail:
            message += "; unsupported vLLM arguments; check the serving recipe"
        elif "Address already in use" in tail or "address already in use" in tail:
            message += "; serving port is already occupied"
        elif "HFValidationError" in tail or "FileNotFoundError" in tail:
            message += "; model or serving file not found; check declared paths"
        if self._shape is not None:
            s = self._shape
            message += (
                f"; effective recipe: tp={s.tensor_parallel}, dp={s.data_parallel}, "
                f"max_len={s.max_model_len}, gpu_mem_util={s.gpu_mem_util}, "
                f"max_num_seqs={s.max_num_seqs}, eager={s.enforce_eager}"
            )
        if self._log_file is not None:
            message += f"; operator log: {self._log_file.name}"
        return RuntimeError(message)

    def wait_ready(self, timeout_s: int = 600) -> None:
        deadline = time.time() + timeout_s
        url = f"{self.base_url}/models"
        last_error = ""
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise self._failure(f"vllm serve exited early (rc={self._proc.returncode})")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read())
                    names = [m["id"] for m in payload.get("data", [])]
                    if self.model_name in names or not names:
                        return
                    last_error = f"served models {names} lack {self.model_name}"
            except Exception as exc:  # noqa: BLE001 - readiness probe loop
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
        raise self._failure(f"serve not ready after {timeout_s}s: {last_error}")

    def model_dir(self) -> Path:
        """Resolve the real weights directory from the live cmdline.

        `--served-model-name` is an alias; the positional model path is
        the truth. Returns the directory the server was started from.
        """
        result = subprocess.run(
            ["pgrep", "-f", "vllm serve"], capture_output=True, text=True
        )
        for pid in result.stdout.split():
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            except OSError:
                continue
            if self.model_name in cmdline and "serve" in cmdline:
                match = re.search(r"vllm\s+serve\s+(\S+)", " ".join(cmdline))
                if match:
                    return Path(match.group(1))
        raise RuntimeError("could not resolve the served model directory")

    def stop(self) -> None:
        # start_new_session=True makes pid the process-group id. Workers
        # can outlive an exited API process, so reap the group in either case.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._proc.pid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait(timeout=10)
        from coevobench.runtime.run_resources import release

        release(self._owner_dir, "process", self._proc.pid)

    @property
    def pid(self) -> int:
        return self._proc.pid


def _default_venv_bin() -> str:
    """The venv that owns the running interpreter (absolute).

    No .resolve(): venv pythons are symlinks to the base interpreter,
    and resolving jumps out of the venv straight into /usr/bin.
    """
    import sys

    return str(Path(sys.executable).absolute().parent)


def _build_cmd(
    model_dir: str,
    port: int,
    *,
    served_name: str,
    venv_bin: str = "",
    shape: ServeShape | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The one command builder. Both dev-eval and final-eval go through
    this function; do not rebuild the command anywhere else."""
    shape = shape or ServeShape()
    venv_bin = venv_bin or _default_venv_bin()
    cmd = [
        f"{venv_bin}/vllm",
        "serve",
        str(model_dir),
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(shape.gpu_mem_util),
        "--max-model-len",
        str(shape.max_model_len),
    ]
    if shape.tensor_parallel > 1:
        cmd += ["--tensor-parallel-size", str(shape.tensor_parallel)]
    if shape.data_parallel > 1:
        cmd += ["--data-parallel-size", str(shape.data_parallel)]
    if shape.max_num_seqs > 0:
        cmd += ["--max-num-seqs", str(shape.max_num_seqs)]
    if shape.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    else:
        cmd += ["--no-enable-prefix-caching"]
    if shape.enforce_eager:
        cmd += ["--enforce-eager"]
    cmd.extend(shape.extra_args)
    if shape.tool_call_parser:
        cmd += ["--tool-call-parser", shape.tool_call_parser]
    if shape.reasoning_parser:
        cmd += ["--reasoning-parser", shape.reasoning_parser]
    if shape.chat_template:
        cmd += ["--chat-template", shape.chat_template]
    env: dict[str, str] = {}
    return cmd, env


class VllmServeRuntime:
    name = "vllm"

    def __init__(self, venv_bin: str = "") -> None:
        self._venv_bin = venv_bin or _default_venv_bin()

    def serve(
        self,
        model_path: Path,
        port: int,
        *,
        served_name: str = "qwen3.5-9b",
        shape: ServeShape | None = None,
        gpu_ids: tuple[int, ...] = (),
        log_file: Path | None = None,
        owner_dir: Path | None = None,
    ) -> ServeEndpoint:
        from coevobench.runtime.run_resources import check_active, register

        check_active(owner_dir)
        cmd, extra_env = _build_cmd(
            str(model_path),
            port,
            served_name=served_name,
            venv_bin=self._venv_bin,
            shape=shape,
        )
        env = dict(os.environ)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        env.update(extra_env)
        if log_file is None:
            import time as _time

            # one file per serve, never truncated: a crash's last words
            # must survive the next start (the fixed port name kept
            # overwriting them — two runs lost their crash history)
            log_file = (owner_dir / "audit/serve" if owner_dir else Path()) / (
                f"coevobench-vllm-{port}-{int(_time.time())}-{uuid.uuid4().hex[:8]}.log"
            )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("w")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                cwd=str(Path(self._venv_bin).resolve().parent.parent),
            )
        finally:
            handle.close()
        endpoint = ServeEndpoint(
            proc, f"http://127.0.0.1:{port}/v1", served_name,
            log_file=log_file, shape=shape or ServeShape(), owner_dir=owner_dir,
        )
        try:
            register(owner_dir, "process", proc.pid)
            endpoint.wait_ready()
        except BaseException:
            endpoint.stop()
            raise
        return endpoint


class ServeRuntime(Protocol):
    name: str

    def serve(
        self, model_path: Path, port: int, *, served_name: str, shape: ServeShape | None
    ) -> ServeEndpoint: ...


__all__ = [
    "ServeEndpoint",
    "ServeRuntime",
    "ServeShape",
    "VllmServeRuntime",
    "_build_cmd",
]
