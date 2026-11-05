"""Download and verify the official Base checkpoint at a fixed revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY = "Qwen/Qwen3.5-9B-Base"
REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path.home() / "models/Qwen3.5-9B-Base")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    target = args.destination.resolve()
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=target,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.jinja",
            "README.md",
            "LICENSE*",
        ],
        max_workers=4,
    )
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    records = []
    for item in info.siblings:
        if not item.rfilename.endswith(".safetensors"):
            continue
        path = target / item.rfilename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if item.lfs is None or actual != item.lfs.sha256:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
        records.append({"file": item.rfilename, "bytes": path.stat().st_size, "sha256": actual})
        print(f"Verified {item.rfilename}", flush=True)
    if len(records) != 4:
        raise RuntimeError(f"Expected four checkpoint shards, found {len(records)}")
    config = json.loads((target / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise RuntimeError("Unexpected Base model architecture")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        target, local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.encode("CoEvoBench environment check"):
        raise RuntimeError("Tokenizer check returned no tokens")
    provenance = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "directory": str(target),
        "shards": records,
        "tokenizer_verified": True,
    }
    (target / ".coevobench-source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
