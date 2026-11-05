"""Public task image preparation persists successful builds and supports retries."""

import json
import subprocess

import pytest

from coevobench.bench.tb import images
from coevobench.core.protocol.splits import from_mapping


def test_local_build_records_identity_and_retains_progress(tmp_path, monkeypatch):
    cache = tmp_path / "tb_tasks.jsonl"
    cache.write_text('\n'.join(json.dumps({"task_id": x, "image": None}) for x in ("one", "two")))
    for name in ("one", "two"):
        context = tmp_path / name / "environment"
        context.mkdir(parents=True)
        (context / "Dockerfile").write_text("FROM python:3.11-slim\nCOPY input /input\n")
        (context / "input").write_text(name)
    monkeypatch.setattr(images, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(images, "frozen_splits", lambda _: from_mapping(
        {"visible": ["one"], "selection": ["two"], "test": []}
    ))
    monkeypatch.setattr(images, "_fetch_tb_dirs", lambda: {x: tmp_path / x for x in ("one", "two")})
    built = set()
    fail = True

    def run(argv, **kwargs):
        if argv[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0 if argv[-1] in built else 1)
        assert argv[1:4] == ["build", "--platform", "linux/amd64"]
        tag = argv[argv.index("-t") + 1]
        if fail and "two-" in tag:
            return subprocess.CompletedProcess(argv, 1)
        assert (tmp_path / tag.split(":")[1].split("-")[0] / "environment/input").is_file()
        built.add(tag)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(images.subprocess, "run", run)
    monkeypatch.setattr(images.subprocess, "check_output", lambda *a, **k: "sha256:fixture\n")
    with pytest.raises(RuntimeError, match="two"):
        images.build_images()
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert rows[0]["image"] and rows[1]["image"] is None
    fail = False
    result = images.build_images()
    assert len(result["images"]) == 2 and len(built) == 2
    assert all(r["image_id"] == "sha256:fixture" for r in result["images"])
