"""Build public TB task environments on the operator's local Docker daemon."""

from __future__ import annotations

import hashlib
import json
import subprocess

from coevobench.bench.tb.tasks import _fetch_tb_dirs, load_tb_tasks_from
from coevobench.core.protocol.protocol_assets import cache_root, frozen_splits
from coevobench.runtime.sandbox.base import validate_context


def build_images(*, task_ids: list[str] | None = None, timeout_s: int = 1800) -> dict:
    """Persist each successfully built image immediately, so retries resume safely."""
    splits = frozen_splits("tb")
    allowed = [*splits.visible, *splits.selection, *splits.test]
    selected = task_ids if task_ids is not None else allowed
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(allowed):
        raise ValueError("choose unique task IDs from the frozen TB task set")
    if timeout_s <= 0:
        raise ValueError("image build timeout must be positive")
    cache = cache_root() / "tb_tasks.jsonl"
    records = load_tb_tasks_from(cache)
    if set(selected) - records.keys():
        raise ValueError("materialize TB tasks before building images: ceb data materialize tb")
    directories = _fetch_tb_dirs()
    built = []
    for task_id in selected:
        context = directories[task_id] / "environment"
        validate_context(context)
        fingerprint = hashlib.sha256(b"linux/amd64\0")
        for path in sorted(context.rglob("*")):
            if path.is_file():
                fingerprint.update(str(path.relative_to(context)).encode() + b"\0")
                with path.open("rb") as stream:
                    fingerprint.update(hashlib.file_digest(stream, "sha256").digest())
        image = f"coevobench-tb:{task_id}-{fingerprint.hexdigest()[:12]}"
        present = subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                                 text=True, timeout=30)
        log = cache.parent / "tb-image-builds" / f"{task_id}.log"
        log.parent.mkdir(exist_ok=True)
        if present.returncode:
            with log.open("w") as stream:
                try:
                    result = subprocess.run(
                        ["docker", "build", "--platform", "linux/amd64", "-t", image, str(context)],
                        stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"TB image build timed out for {task_id}; see {log}"
                    ) from exc
            if result.returncode:
                raise RuntimeError(f"TB image build failed for {task_id}; see {log}")
        identity = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True, timeout=30
        ).strip()
        records[task_id].update(image=image, image_id=identity,
                                environment_sha256=fingerprint.hexdigest())
        temporary = cache.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in records.values()))
        temporary.replace(cache)
        built.append({"task_id": task_id, "image": image, "image_id": identity})
    return {"bench": "tb", "images": built, "cache": str(cache)}
