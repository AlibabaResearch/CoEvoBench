"""Materialize only public Visible views; worlds and evaluator assets stay private."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from coevobench.bench.appworld.assets import DATA_VERSION, data_root, sha256
from coevobench.core.protocol.protocol_assets import cache_root, frozen_splits


def load_tasks(root: Path | None = None) -> dict[str, dict]:
    root = root or data_root()
    splits = frozen_splits("appworld")
    if splits is None:
        raise ValueError("AppWorld frozen splits missing")
    tasks = {}
    for task_id in (*splits.visible, *splits.selection, *splits.test):
        directory = root / "data/tasks" / task_id
        specs = directory / "specs.json"
        metadata = directory / "ground_truth/metadata.json"
        if not specs.is_file() or not metadata.is_file():
            raise ValueError(f"AppWorld task assets missing: {task_id}; run setup-appworld.py")
        raw = json.loads(specs.read_text())
        meta = json.loads(metadata.read_text())
        if raw.get("db_version") != DATA_VERSION or meta.get("difficulty") != 1:
            raise ValueError(f"AppWorld task version/difficulty mismatch: {task_id}")
        supervisor = raw.get("supervisor", raw.get("main_user"))
        if not isinstance(supervisor, dict) or not isinstance(raw.get("instruction"), str):
            raise ValueError(f"invalid AppWorld public task: {task_id}")
        tasks[task_id] = {
            "task_id": task_id,
            "instruction": raw["instruction"],
            "supervisor": supervisor,
            "scene_id": task_id.rsplit("_", 1)[0],
            "specs_sha256": sha256(specs),
        }
    for name in ("visible", "selection", "test"):
        counts = Counter(tasks[key]["scene_id"] for key in splits.ids(name))
        if set(counts.values()) != {3}:
            raise ValueError(f"incomplete AppWorld scene in frozen {name} split")
    return tasks


def materialize_appworld(workspace: Path, judge_cache: Path | None = None) -> dict:
    tasks = load_tasks()
    splits = frozen_splits("appworld")
    target = workspace / "tasks_visible/appworld"
    target.mkdir(parents=True, exist_ok=True)
    for key in splits.visible:
        public = {name: tasks[key][name] for name in ("task_id", "instruction", "supervisor")}
        (target / f"{key}.json").write_text(json.dumps(public, ensure_ascii=False, indent=2))
    private = judge_cache or cache_root() / "appworld_tasks.jsonl"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks.values())
    )
    private.chmod(0o600)
    return {"n_tasks": len(tasks), "n_visible": len(splits.visible), "judge_cache": str(private)}
