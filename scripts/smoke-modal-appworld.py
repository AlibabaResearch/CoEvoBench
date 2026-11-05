"""Short, billed Modal acceptance: policy egress isolation, optionally four-GPU CUDA/vLLM."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import tempfile
import time
import urllib.request
from pathlib import Path

from coevobench.config.cloud_env import load_env
from coevobench.runtime.gpu.modal_jobs import ModalGpuSession
from coevobench.runtime.modal_platform import ModalPlatform
from coevobench.runtime.run_resources import stop_resources
from coevobench.runtime.sandbox.base import SandboxSpec
from coevobench.runtime.sandbox.modal import ModalSandboxRuntime
from coevobench.runtime.serve.modal import ModalServeRuntime
from coevobench.runtime.serve.vllm import ServeShape

PROJECT = Path(__file__).resolve().parents[1]
PROBE = r"""
import json,socket,ssl,sys,urllib.request
rows=[]
for host,ip in json.loads(sys.argv[1]):
 row={"host":host}
 try:
  with socket.create_connection((ip,443),timeout=4) as s:
   with ssl.create_default_context().wrap_socket(s,server_hostname=host): pass
  row["direct_tls"]=True
 except OSError: row["direct_tls"]=False
 try:
  with urllib.request.urlopen("https://"+host,timeout=4) as response: response.read(1)
  row["https"]=True
 except OSError: row["https"]=False
 rows.append(row)
print(json.dumps(rows))
"""
GPU_CHECK = r"""
import os,torch,torch.distributed as dist
rank=int(os.environ["LOCAL_RANK"])
assert torch.cuda.device_count()==4
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
x=torch.tensor([rank+1.0],device=f"cuda:{rank}")
dist.all_reduce(x)
assert x.item()==10.0
print(rank,torch.cuda.get_device_name(rank),"CUDA/NCCL OK",flush=True)
dist.destroy_process_group()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="allocate four GPUs and test NCCL/vLLM")
    args = parser.parse_args()
    load_env(PROJECT / ".local/appworld-cloud.env")
    root = PROJECT / ".local/modal-acceptance"
    root.mkdir(parents=True, exist_ok=True)
    owner = Path(tempfile.mkdtemp(dir=root))
    (owner / "lifecycle.json").write_text(
        json.dumps({"phase": "running", "deadline_at": time.time() + 1800})
    )
    secrets = owner / ".cloud-secrets.json"
    with secrets.open("x") as stream:
        secrets.chmod(0o600)
        json.dump(
            {k: v for k, v in os.environ.items() if k.startswith(("MODAL_", "CEB_MODAL_"))}, stream
        )
    report = {"cloud_verified": False}
    platform = ModalPlatform()
    runtime = ModalSandboxRuntime(platform=platform, owner_dir=owner)
    print("Acceptance receipts: " + str(owner), flush=True)
    try:
        online = runtime.create(
            SandboxSpec(
                image=os.environ["CEB_MODAL_RESEARCHER_IMAGE"],
                cpu=4,
                mem_gib=8,
                ttl_s=1800,
                network=True,
            )
        )
        offline = runtime.create(
            SandboxSpec(
                image=os.environ["CEB_APPWORLD_POLICY_IMAGE"],
                cpu=2,
                mem_gib=4,
                ttl_s=1800,
                network=False,
            )
        )
        offline.write_bytes("/tmp/ceb-roundtrip.bin", bytes(range(256)))
        assert offline.read_bytes("/tmp/ceb-roundtrip.bin") == bytes(range(256))
        assert online.run("id researcher && codex --version")["exit_code"] == 0
        targets = [
            (host, socket.gethostbyname(host))
            for host in ("github.com", "pypi.org", "registry.npmjs.org")
        ]
        command = "python3 -c " + shlex.quote(PROBE) + " " + shlex.quote(json.dumps(targets))
        controls = json.loads(online.run(command, timeout_s=90)["text"])
        blocked = json.loads(offline.run(command, timeout_s=90)["text"])
        report.update(online_control=controls, offline_probe=blocked)
        assert all(row["direct_tls"] and row["https"] for row in controls), (
            "online controls inconclusive"
        )
        assert all(not row["direct_tls"] and not row["https"] for row in blocked), (
            "policy network isolation failed"
        )
        if args.gpu:
            base = Path(os.environ["BASE_MODEL_DIR"]).resolve()
            (owner / "base-model.json").write_text(json.dumps({"path": str(base)}))
            session = ModalGpuSession(owner, platform=platform)
            box = session.create()
            box.write_bytes("/tmp/ceb-gpu-check.py", GPU_CHECK.encode())
            result = box.run(
                "runuser -u cebtrain -- /opt/venv-train/bin/python -m torch.distributed.run "
                "--standalone --nproc_per_node=4 /tmp/ceb-gpu-check.py",
                timeout_s=180,
            )
            assert result["exit_code"] == 0, result["text"][-1000:]
            report["cuda_nccl"] = result["text"]
            shape = ServeShape(tensor_parallel=4, max_model_len=65536, max_num_seqs=1)
            endpoint = ModalServeRuntime(platform=platform).serve(
                base, 0, owner_dir=owner, shape=shape
            )
            try:
                request = urllib.request.Request(
                    endpoint.base_url + "/completions",
                    data=json.dumps(
                        {
                            "model": endpoint.model_name,
                            "prompt": "1 + 1 =",
                            "max_tokens": 8,
                            "temperature": 0,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                assert result.get("choices"), "vLLM did not return a completion"
                report["vllm_completion"] = True
            finally:
                endpoint.stop()
        report["cloud_verified"] = True
    finally:
        (owner / "report.json").write_text(json.dumps(report, indent=2))
        # May raise if the provider is unavailable; receipts remain retryable.
        stop_resources(owner)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
