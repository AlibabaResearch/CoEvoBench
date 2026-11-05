"""TB-Lite task pipeline: tarball fetch, public views, judge records.

The judge record carries the verifier's own files (test.sh plus every
data file the tests directory holds — uploaded only after the agent
finishes) and the gold solve.sh. One GitHub tarball request replaces
per-file raw fetches (anonymous API rate limits cannot survive a
100-task walk). Binary grader files serialize as base64.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coevobench.config.schema import Config
from coevobench.core.protocol.splits import Splits

_TB_SOURCE = "https://github.com/open-thoughts/OpenThoughts-TBLite/archive/refs/heads/main.tar.gz"
_JUDGE_FIELDS_TB = ("task_id", "instruction", "test_sh", "test_files", "solution_sh", "image")


def _fetch_tb_dirs() -> dict[str, Path]:
    """Extract the OpenThoughts-TBLite task dirs from the GitHub tarball.

    One tarball request replaces per-file raw fetches (the anonymous
    API rate limit would not survive a 100-task walk).
    """
    import tarfile
    import tempfile
    import urllib.request

    from coevobench.core.protocol.protocol_assets import cache_root

    root = cache_root(Config()) / "tb_dataset"
    inner = root / "OpenThoughts-TBLite-main"
    if inner.is_dir():
        return {
            p.name: p
            for p in inner.iterdir()
            if p.is_dir() and (p / "task.toml").is_file()
        }
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "tblite.tar.gz"
        urllib.request.urlretrieve(_TB_SOURCE, archive)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            wanted = [
                m for m in tf.getmembers()
                if m.name.startswith("OpenThoughts-TBLite-main/")
                and "/" in m.name.removeprefix("OpenThoughts-TBLite-main/")
            ]
            tf.extractall(root, members=wanted, filter="data")
    return {
        p.name: p
        for p in inner.iterdir()
        if p.is_dir() and (p / "task.toml").is_file()
    }


def _tb_test_entries(tests_dir: Path) -> list[dict]:
    """Serialise every grader file: text stays text, binaries go base64.

    Skips interpreter and pytest caches — the graders themselves never
    read them.
    """
    entries = []
    for p in sorted(tests_dir.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(tests_dir).parts
        if "__pycache__" in parts or ".pytest_cache" in parts:
            continue
        rel = f"tests/{p.relative_to(tests_dir).as_posix()}"
        data = p.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            import base64

            entries.append(
                {"path": rel, "encoding": "base64", "data": base64.b64encode(data).decode("ascii")}
            )
            continue
        entries.append({"path": rel, "data": text})
    return entries


def materialize_tb(
    splits: Splits, workspace: Path, *, judge_cache: Path, mirror=None
) -> dict[str, Any]:
    """Write TB public views into the workspace; judge records to the cache.

    The judge record carries the verifier's own files (test.sh plus
    every data file the tests directory holds — uploaded only after the
    agent finishes) and the gold solve.sh. The image field is the
    mirror-resolved reference when a task_id-keyed mirror is
    configured; None otherwise (no public registry serves per-task TB
    images — they build from each task's environment/Dockerfile).
    """
    dirs = _fetch_tb_dirs()
    frozen = [
        tid for split in ("visible", "selection", "test")
        for tid in splits.ids(split)
    ]
    missing = [t for t in frozen if t not in dirs]
    if missing:
        raise ValueError(
            f"{len(missing)} frozen tasks missing from the dataset (e.g. {missing[:3]})"
        )
    base = ""
    if mirror is not None and getattr(mirror, "rewrite", ""):
        rw = mirror.rewrite.rstrip("/")
        base = rw if rw.endswith(":") else rw + ":"

    public_dir = workspace / "tasks_visible" / "tb"
    public_dir.mkdir(parents=True, exist_ok=True)
    visible = set(splits.visible)
    # Remove only generated non-Visible task files left by earlier materialization.
    for task_id in set(frozen) - visible:
        (public_dir / f"{task_id}.json").unlink(missing_ok=True)
    existing = load_tb_tasks_from(judge_cache)
    judge_records: dict[str, dict] = {}
    for task_id in frozen:
        d = dirs[task_id]
        instruction = (d / "instruction.md").read_text()
        if task_id in visible:
            (public_dir / f"{task_id}.json").write_text(
                json.dumps({"task_id": task_id, "instruction": instruction}, ensure_ascii=False)
            )
        test_files = _tb_test_entries(d / "tests")
        test_sh = next((f["data"] for f in test_files if f["path"] == "tests/test.sh"), "")
        solution_sh = ""
        solve = d / "solution" / "solve.sh"
        if solve.is_file():
            solution_sh = solve.read_text()
        judge_records[task_id] = {
            "task_id": task_id,
            "instruction": instruction,
            "test_sh": test_sh,
            "test_files": test_files,
            "solution_sh": solution_sh,
            "image": f"{base}{task_id}" if base else existing.get(task_id, {}).get("image"),
        }
        if not base:
            for field in ("image_id", "environment_sha256"):
                if field in existing.get(task_id, {}):
                    judge_records[task_id][field] = existing[task_id][field]
    judge_cache.parent.mkdir(parents=True, exist_ok=True)
    judge_cache.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in judge_records.values()) + "\n"
    )
    return {
        "visible_tasks": len(splits.ids("visible")),
        "judge_records": len(judge_records),
    }


def load_tb_tasks() -> dict[str, dict]:
    """Judge-side task details from the cache (no network).

    CEB_DATA_ROOT (remote runs carry their cache inside the workspace)
    wins over the operator host's ~/.ceb/cache.
    """
    import os

    if os.environ.get("CEB_DATA_ROOT"):
        path = Path(os.environ["CEB_DATA_ROOT"]) / "tb_tasks.jsonl"
    else:
        from coevobench.core.protocol.protocol_assets import cache_root

        path = cache_root(Config()) / "tb_tasks.jsonl"
    if not path.is_file():
        raise RuntimeError(
            "tb task details not materialized; run: ceb data materialize tb"
        )
    return load_tb_tasks_from(path)


def load_tb_tasks_from(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    tasks = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[row["task_id"]] = row
    return tasks


__all__ = ["load_tb_tasks", "materialize_tb"]
