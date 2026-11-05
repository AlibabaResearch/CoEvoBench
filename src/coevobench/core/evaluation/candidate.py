"""Candidate evaluation session: serve candidate weights, drive the
candidate's own harness through the benchmark (A1+A2 of the gap audit).

The researcher trains weights and evolves a harness; evaluating the
PAIR is the benchmark's core loop (paper Section 3.3). This module
runs it operator-side:

- resolve the candidate (harness dir + weights path + serve shape)
- serve the weights through the single vLLM command builder
- drive episodes: the candidate harness's agent loop against the
  SandboxEnvironment seam, on the requested split's tasks
- collect submissions (SWE: the sandbox's real git diff; TB: judged
  in-sandbox by the verifier-after-rollout ordering)
- judge each submission fresh-sandbox and aggregate through the
  shared evaluate loop (n_bad / dominance rules apply)

The harness runs with its own PYTHONPATH (workspace first, platform
source behind it) and receives the welded SandboxEnvironment; the
frozen _transport.py contract keeps the model endpoint pinned to the
served candidate.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from coevobench.runtime.sandbox.base import SandboxSpec

# The sandboxed driver is bench-agnostic: it connects to the episode
# sandbox, wraps the SandboxEnvironment seam, and runs the harness's
# agent. SWE and TB share it (same minisweagent harness shape); the
# in-process benches (appworld/bfcl/tau) never spawn a sandbox driver —
# their episodes run inside the judge process.
_DRIVER_MODULES = {
    "swe": "coevobench.bench.swe.episode_driver",
    "tb": "coevobench.bench.swe.episode_driver",
}


def _episode_driver_module(bench: str) -> str:
    try:
        return _DRIVER_MODULES[bench]
    except KeyError:
        raise ValueError(
            f"sandboxed candidate episodes exist for {sorted(_DRIVER_MODULES)}; "
            f"got {bench!r} (in-process benches run episodes in the judge)"
        ) from None


@dataclass(frozen=True)
class Candidate:
    """A researcher submission: weights + harness + serving declaration."""

    label: str
    weights: Path
    harness_src: Path          # the harness package root (contains minisweagent/ or bfclharness/)
    serve_shape: dict = field(default_factory=dict)
    bench: str = "swe"
    owner_dir: Path | None = None

    @classmethod
    def from_dir(cls, submission_dir: Path, bench: str) -> Candidate:
        submission_dir = Path(submission_dir).resolve()
        weights = Path((submission_dir / "model_path.txt").read_text().strip())
        harness_src = submission_dir / "harness"
        candidates: list[Path] = []
        for pkg in (
            "minisweagent",
            "bfclharness",
            "appworldharness",
            "tauharness",
        ):
            candidates += list(harness_src.rglob(f"{pkg}/__init__.py"))
        if not candidates:
            raise ValueError(
                f"no harness package under {harness_src} "
                "(minisweagent/ bfclharness/ appworldharness/ tauharness/)"
            )
        package_root = candidates[0].parent.parent
        serve_shape = {}
        serve_file = submission_dir / "serve_config.json"
        if serve_file.is_file():
            serve_shape = json.loads(serve_file.read_text())
        return cls(
            label=submission_dir.name,
            weights=weights,
            harness_src=package_root,
            serve_shape=serve_shape,
            bench=bench,
        )


@contextmanager
def candidate_serve(
    candidate: Candidate,
    port: int = 18900,
    *,
    serve_sandbox=None,
    owner_dir: Path | None = None,
    gpu_ids: tuple[int, ...] = (),
) -> Iterator[dict]:
    """Serve the candidate's weights; yields the endpoint description.

    Two modes:
    - local (default): the operator host runs vLLM; episodes reach it
      at localhost — the docker and in-envelope profile.
    - remote (serve_sandbox): the weights already live in a remote run
      sandbox (the Modal profile); vLLM starts INSIDE that sandbox on
      its declared serve port and the yielded base_url is the sandbox's
      public tunnel URL — judge sandboxes cannot reach another
      sandbox's localhost, so the endpoint is the tunnel, never 127.0.0.1.
    """
    if serve_sandbox is not None:

        with _remote_serve(serve_sandbox, candidate) as served:
            yield served
        return

    from coevobench.runtime.serve.vllm import ServeShape, VllmServeRuntime

    shape = ServeShape(
        tensor_parallel=int(candidate.serve_shape.get("tensor_parallel", 1)),
        data_parallel=int(candidate.serve_shape.get("data_parallel", 1)),
        gpu_mem_util=float(candidate.serve_shape.get("gpu_mem_util", 0.85)),
        max_model_len=int(candidate.serve_shape.get("max_len", 32768)),
        max_num_seqs=int(candidate.serve_shape.get("max_num_seqs", 0)),
        enable_prefix_caching=bool(candidate.serve_shape.get("enable_prefix_caching", True)),
        enforce_eager=bool(candidate.serve_shape.get("enforce_eager", False)),
        tool_call_parser=str(candidate.serve_shape.get("tool_call_parser", "") or ""),
        reasoning_parser=str(candidate.serve_shape.get("reasoning_parser", "") or ""),
        chat_template=(str(candidate.serve_shape["chat_template"])
                       if candidate.serve_shape.get("chat_template") else ""),
        extra_args=tuple(candidate.serve_shape.get("extra_args") or ()),
    )
    runtime = VllmServeRuntime()
    options = {}
    if owner_dir or candidate.owner_dir:
        options["owner_dir"] = owner_dir or candidate.owner_dir
    if gpu_ids:
        options["gpu_ids"] = gpu_ids
    endpoint = runtime.serve(candidate.weights, port, served_name="qwen3.5-9b",
                             shape=shape, **options)
    try:
        yield {
            "base_url": endpoint.base_url,
            "model_name": endpoint.model_name,
            "port": port,
        }
    finally:
        endpoint.stop()


def run_candidate_episode(
    candidate: Candidate,
    task_id: str,
    served: dict,
    *,
    bench: str,
    task_data: dict,
    sandbox_runtime,
    step_limit: int = 80,
    wall_cap_s: int = 900,
    max_tokens: int = 5120,
    temperature: float | None = None,
) -> dict:
    """One episode: the candidate harness drives its agent in a sandbox.

    Returns the submission artifact (SWE: the diff string; TB: judged
    inline) plus the episode's exit status.
    """
    import os
    import subprocess
    import sys
    import tempfile


    spec = SandboxSpec(
        image=task_data.get("image", "python:3.11-slim"),
        cpu=2,
        mem_gib=8,
        ttl_s=max(wall_cap_s * 2, 1800),
    )
    sandbox = sandbox_runtime.create(spec)
    try:
        prompt = task_data.get("problem_statement") or task_data.get("instruction", "")
        env = dict(os.environ)
        env.update(
            {
                "CEB_SANDBOX_ID": sandbox.id,
                "VLLM_BASE": served["base_url"],
                # litellm keys OpenAI-compatible endpoints by the openai/
                # prefix; the
                # served-model name stays unprefixed at the endpoint.
                "MODEL": f"openai/{served['model_name']}",
                "PYTHONPATH": f"{candidate.harness_src}:{_platform_src()}",
                "CEB_CANDIDATE_LABEL": candidate.label,
                "CEB_TASK_ID": task_id,
                # the driver connects to this sandbox in the family that
                # created it — stamped here, read by episode_driver
                "CEB_SANDBOX_RUNTIME": getattr(sandbox_runtime, "name", "docker"),
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as result_file:
            result_path = Path(result_file.name)
        env["CEB_RESULT_FILE"] = str(result_path)
        driver = _episode_driver_module(bench)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                driver,
                json.dumps(prompt),
                str(step_limit),
                str(wall_cap_s),
                str(max_tokens),
                str(0.0 if temperature is None else temperature),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=wall_cap_s + 120,
            cwd=str(candidate.harness_src),
        )
        result = {}
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
            except ValueError:
                result = {}
        result_path.unlink(missing_ok=True)
        if proc.returncode != 0 and not result:
            result = {
                "exit_status": f"HarnessError:{proc.returncode}",
                "stderr": proc.stderr[-500:],
            }
        result.setdefault("exit_status", "LimitsExceeded")
        result["sandbox_id"] = sandbox.id
        return result
    except BaseException:
        sandbox.kill()
        raise


def extract_swe_submission(sandbox_id: str, sandbox_runtime) -> str:
    """The real git diff from the episode's sandbox (integrity #3)."""
    sb = sandbox_runtime.connect(sandbox_id)
    try:
        out = sb.run(
            "cd /testbed && git add -N . && git diff HEAD 2>/dev/null | head -c 200000",
            timeout_s=60,
        )
        return out["text"]
    finally:
        sb.close()


def _platform_src() -> str:
    return str(Path(__file__).resolve().parents[2])


@contextmanager
def _remote_serve(serve_target, candidate: Candidate) -> Iterator[dict]:
    """Serve on a dedicated Modal GPU sandbox; yield the tunnel endpoint.

    Two live-verified Modal facts shape this:
    - exec children die with the exec call — the service must BE the
      sandbox entrypoint (it runs vLLM in the foreground, gated on the
      weights upload landing);
    - sandboxes are isolated network namespaces — judges reach the
      served model only through the public tunnel port, so vLLM binds
      0.0.0.0 (tunnels forward at the container network layer; loopback
      never surfaces).

    The serve sandbox is independent of the run sandbox and dies with
    this context manager.
    """
    import contextlib

    from coevobench.runtime.modal_util import require_modal
    from coevobench.runtime.runhost.modal_host import SERVE_PORT

    if candidate.weights is None or not Path(candidate.weights).is_dir():
        raise RuntimeError("remote serve needs materialized weights on the operator host")

    modal = require_modal()
    app = modal.App.lookup("coevobench-serve", create_if_missing=True)
    image = modal.Image.from_registry(_remote_serve.image or "coevobench-run:latest")
    sandbox = modal.Sandbox.create(
        "sh", "-c", _serve_entrypoint(SERVE_PORT),
        app=app,
        image=image,
        gpu=_remote_serve.gpu or "A100-40GB",
        cpu=8,
        memory=48 * 1024,
        timeout=60 * 60 * 6,
        unencrypted_ports=[SERVE_PORT],
    )
    try:
        _upload_weights(sandbox, candidate.weights)
        tunnel = _await_health(sandbox, SERVE_PORT)
        host, tunnel_port = tunnel.tcp_socket
        yield {
            "base_url": f"http://{host}:{tunnel_port}/v1",
            "model_name": "qwen3.5-9b",
            "port": SERVE_PORT,
        }
    finally:
        with contextlib.suppress(Exception):
            sandbox.terminate(wait=False)


_remote_serve.image = None
_remote_serve.gpu = None


def _serve_entrypoint(port: int) -> str:
    return (
        "while [ ! -f /workspace/.weights-ready ]; do sleep 2; done; "
        "vllm serve /workspace/weights --served-model-name qwen3.5-9b "
        f"--host 0.0.0.0 --port {port} > /workspace/.serve.log 2>&1"
    )


def _await_health(sandbox, port: int, timeout_s: int = 1200):
    """Gate on /health; raise with the serve log on failure.

    The probe is python urllib, not curl: slim images carry python but
    not curl, and a missing tool would read as PENDING forever.
    """
    import time

    from coevobench.runtime.modal_util import read_stream

    sandbox.exec("sh", "-c", "touch /workspace/.weights-ready", timeout=30).wait()
    probe_cmd = (
        "python -c \"import urllib.request,sys;"
        "sys.exit(0 if urllib.request.urlopen("
        f"'http://127.0.0.1:{port}/health', timeout=5"
        ").status == 200 else 1)\" && echo HEALTHY || echo PENDING"
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = sandbox.exec("sh", "-c", probe_cmd, timeout=30)
        probe.wait()
        if "HEALTHY" in read_stream(probe.stdout):
            return sandbox.tunnels()[port]
        time.sleep(10)
    log = sandbox.exec("sh", "-c", "tail -30 /workspace/.serve.log", timeout=30)
    log.wait()
    raise RuntimeError(
        "remote vLLM did not become healthy; log tail: "
        + read_stream(log.stdout)[-1500:]
    )


def _upload_weights(sandbox, weights: Path) -> None:
    """Ship the weights directory in through a tar stream."""
    import tarfile
    import tempfile

    from coevobench.runtime.modal_util import read_stream

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
        tar_path = Path(tf.name)
    with tarfile.open(tar_path, "w") as tar:
        for path in sorted(Path(weights).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(weights)))
    sandbox.exec("sh", "-c", "mkdir -p /workspace", timeout=30).wait()
    sandbox.filesystem.copy_from_local(str(tar_path), "/tmp/weights.tar")
    extract = sandbox.exec(
        "sh", "-c",
        "mkdir -p /workspace/weights && tar -xf /tmp/weights.tar -C /workspace/weights"
        " && rm /tmp/weights.tar && echo uploaded",
        timeout=900,
    )
    extract.wait()
    tar_path.unlink(missing_ok=True)
    if "uploaded" not in read_stream(extract.stdout):
        raise RuntimeError("weights upload failed")


__all__ = [
    "Candidate",
    "candidate_serve",
    "extract_swe_submission",
    "run_candidate_episode",
]
