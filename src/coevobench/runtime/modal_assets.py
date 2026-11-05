"""Operator checkpoint validation and sanitized Modal image build contexts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def checkpoint_manifest(root: Path, *, hash_files=False) -> dict:
    root = root.resolve()
    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("unexpected base architecture")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 4:
        raise ValueError("expected all four base checkpoint shards")
    rows = []
    names = set()
    for name in shards:
        if Path(name).name != name:
            raise ValueError("unsafe base shard path")
        path = root / name
        if path.is_symlink():
            raise ValueError("materialize base symlinks first")
        size = path.stat().st_size
        with path.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            if not 2 <= header_size <= min(size - 8, 64 * 1024**2):
                raise ValueError("invalid safetensors header")
            header = json.loads(stream.read(header_size))
            end = 0
            for tensor, desc in header.items():
                if tensor == "__metadata__":
                    continue
                start, stop = desc["data_offsets"]
                if not 0 <= start <= stop <= size - 8 - header_size:
                    raise ValueError("truncated base checkpoint shard")
                names.add(tensor)
                end = max(end, stop)
            if end != size - 8 - header_size:
                raise ValueError("base checkpoint size mismatch")
            row = {"file": name, "bytes": size}
            if hash_files:
                stream.seek(0)
                row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            rows.append(row)
    if set(index["weight_map"]) != names:
        raise ValueError("base shard index mismatch")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError("base tokenizer missing")
    metadata = []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.name in shards or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError("materialize base metadata symlinks first")
        row = {"file": path.name, "bytes": path.stat().st_size}
        if hash_files:
            with path.open("rb") as stream:
                row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata.append(row)
    return {"architecture": config["architectures"][0], "shards": rows, "metadata": metadata}


def build_context(project: Path, kind: str, destination: Path) -> Path:
    destination.mkdir(parents=True)
    if kind == "researcher":
        dockerfile = (project / "Dockerfile.run").read_text()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copyfile(project / name, destination / name)
        shutil.copytree(project / "envelope", destination / "envelope",
                        ignore=shutil.ignore_patterns("__pycache__"))
        entries = (project / "envelope/WHITELIST").read_text().splitlines()
        for name in [*entries, "coevobench/runtime/envelope.py"]:
            if not name.strip() or name.startswith("#"):
                continue
            target = destination / "src" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / "src" / name, target)
        shutil.copytree(project / "templates", destination / "templates",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(project / "Dockerfile.run", destination / "Dockerfile.run")
    else:
        if kind not in ("policy", "gpu"):
            raise ValueError(f"unknown image context: {kind}")
        file = "appworld-policy.Dockerfile" if kind == "policy" else "appworld-modal-gpu.Dockerfile"
        dockerfile = (project / "deploy" / file).read_text()
    (destination / "Dockerfile").write_text(dockerfile)
    return destination
