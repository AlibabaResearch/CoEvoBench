"""SWE task pipeline: dataset fetch, public views, judge records.

Two distinct surfaces:

- workspace material (researcher-visible): public_task.json per visible
  task — problem statement only, gold stripped, written into the
  workspace under tasks_visible/swe/
- judge details (operator-side): the full instance record (patch,
  test_patch, FAIL_TO_PASS, PASS_TO_PASS) the SweBench judge consumes,
  cached under ~/.ceb/cache/swe_tasks.jsonl so eval.run does not depend
  on network access

Both derive from the official dataset fetch (HF) and are validated
against the frozen splits: a task missing from the dataset fails
loudly instead of silently grading wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coevobench.config.schema import Config
from coevobench.core.protocol.splits import Splits

_PUBLIC_FIELDS_SWE = ("instance_id", "repo", "version", "problem_statement")
_JUDGE_FIELDS_SWE = (
    "instance_id", "repo", "version", "problem_statement",
    "base_commit", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    "test_cmd", "image",
)


def _fetch_verified_rows() -> dict[str, dict]:

    from datasets import load_dataset

    from coevobench.core.protocol.protocol_assets import cache_root

    cache = cache_root(Config()) / "swe_tasks.jsonl"
    if cache.is_file():
        rows = {}
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["instance_id"]] = row
        if rows:
            return rows
    ds = load_dataset(
        "princeton-nlp/SWE-bench_Verified", split="test", trust_remote_code=False
    )
    rows = {r["instance_id"]: dict(r) for r in ds}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows.values()) + "\n"
    )
    return rows


def official_eval_image(instance_id: str) -> str:
    """The official swebench-5.0 harness formula (task/checks.py).

    swebench/sweb.eval.x86_64.<instance_id>.lower() with '__' replaced
    by '_1776_' — resolves for all 493 frozen instances on docker.io.
    """
    key = f"sweb.eval.x86_64.{instance_id}".lower()
    return f"swebench/{key}".replace("__", "_1776_")


def materialize_swe(
    splits: Splits,
    workspace: Path,
    *,
    judge_cache: Path,
    mirror=None,
) -> dict[str, Any]:
    """Write public views into the workspace; judge records to the cache.

    The judge record's image field is what candidate episodes spawn
    from (the episode sandbox needs the official testbed). A registry-
    keyed mirror (tag_source=instance_id) bakes its fast-path refs in;
    otherwise the official docker.io formula.
    """
    rows = _fetch_verified_rows()
    missing = [t for t in splits.ids("visible") if t not in rows]
    if missing:
        raise ValueError(
            f"{len(missing)} visible tasks missing from the dataset (e.g. {missing[:3]})"
        )
    base = ""
    if mirror is not None and getattr(mirror, "rewrite", ""):
        rw = mirror.rewrite.rstrip("/")
        base = rw if rw.endswith(":") else rw + ":"

    public_dir = workspace / "tasks_visible" / "swe"
    public_dir.mkdir(parents=True, exist_ok=True)
    for task_id in splits.ids("visible"):
        row = rows[task_id]
        public = {k: row.get(k) for k in _PUBLIC_FIELDS_SWE}
        (public_dir / f"{task_id}.json").write_text(
            json.dumps(public, ensure_ascii=False, indent=2)
        )

    judge_records = {}
    for split in ("visible", "selection", "test"):
        for task_id in splits.ids(split):
            row = rows.get(task_id)
            if row is None:
                continue
            record = {k: row.get(k) for k in _JUDGE_FIELDS_SWE}
            record.setdefault("test_cmd", "pytest -rA")
            record["image"] = (
                f"{base}{task_id}" if base else official_eval_image(task_id)
            )
            judge_records[task_id] = record
    judge_cache.parent.mkdir(parents=True, exist_ok=True)
    judge_cache.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in judge_records.values())
        + "\n"
    )
    return {
        "visible_tasks": len(splits.ids("visible")),
        "judge_records": len(judge_records),
    }


def load_swe_instances() -> dict[str, dict]:
    """Judge-side instance details from the cache (no network)."""
    from coevobench.core.protocol.protocol_assets import cache_root

    path = cache_root(Config()) / "swe_tasks.jsonl"
    if not path.is_file():
        raise RuntimeError(
            "swe task details not materialized; run: ceb data materialize swe"
        )
    instances = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            instances[row["instance_id"]] = row
    return instances


__all__ = ["load_swe_instances", "materialize_swe", "official_eval_image"]
