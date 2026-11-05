"""Build the network-isolated BFCL generator without benchmark data or answers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from coevobench.bench.bfcl.provenance import MANIFEST, VENDOR_ROOT, digest, verify_upstream

PROJECT = Path(__file__).resolve().parents[1]
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from huggingface_hub import snapshot_download

    verify_upstream()
    # The official vector backend resolves this exact model name. Cache it
    # before disabling networking, including the ref pointing to the revision.
    cache = PROJECT / ".local/bfcl-hf-cache"
    snapshot = Path(
        snapshot_download(
            "sentence-transformers/all-MiniLM-L6-v2",
            revision=EMBEDDING_REVISION,
            cache_dir=cache,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
    )
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(EMBEDDING_REVISION)
    with tempfile.TemporaryDirectory(prefix="ceb-bfcl-build-") as temporary:
        root = Path(temporary)
        for source in (VENDOR_ROOT / "bfcl_eval").rglob("*.py"):
            target = root / "engine" / source.relative_to(VENDOR_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copytree(cache, root / "cache", symlinks=False)
        shutil.copyfile(VENDOR_ROOT.parent / "LICENSE", root / "LICENSE.bfcl")
        worker = PROJECT / "src/coevobench/bench/bfcl/generation.py"
        shutil.copyfile(worker, root / "generation.py")
        dependencies = tomllib.loads((VENDOR_ROOT / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
        packages = sorted(
            {re.split(r"[<>=!~;\[]", dep)[0] for dep in dependencies}
            | {"transformers", "soundfile"}
        )
        grader = PROJECT / "src/coevobench/bench/bfcl/grading.py"
        shutil.copyfile(grader, root / "grading.py")
        (root / "requirements.txt").write_text(
            "".join(f"{name}=={importlib.metadata.version(name)}\n" for name in packages)
        )
        torch_version = importlib.metadata.version("torch")
        (root / "Dockerfile").write_text(f'''FROM python:3.11-slim
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgomp1 ca-certificates
RUN rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.3
ARG TORCH_VERSION={torch_version}
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch==$TORCH_VERSION
COPY requirements.txt /opt/requirements.txt
RUN uv pip install --system -r /opt/requirements.txt && uv pip check --system
COPY engine /opt/engine
COPY cache /opt/hf/hub
COPY generation.py /opt/generation.py
COPY grading.py /opt/grading.py
COPY LICENSE.bfcl /opt/LICENSE.bfcl
ENV PYTHONPATH=/opt/engine BFCL_PROJECT_ROOT=/tmp/bfcl HOME=/tmp HF_HOME=/opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LABEL coevobench.bfcl.manifest="{digest(MANIFEST)}"
LABEL coevobench.bfcl.generation="{digest(worker)}"
LABEL coevobench.bfcl.embedding="{EMBEDDING_REVISION}"
LABEL coevobench.bfcl.grading="{digest(grader)}"
USER 1000:1000
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/generation.py"]
''')
        subprocess.run(["docker", "build", "-t", "coevobench-bfcl:latest", str(root)], check=True)
    print(json.dumps({"image": "coevobench-bfcl:latest", "embedding_revision": EMBEDDING_REVISION}))


if __name__ == "__main__":
    main()
