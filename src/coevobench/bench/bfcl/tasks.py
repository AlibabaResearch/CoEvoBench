"""Host-only BFCL tasks, official answers, and complete memory prerequisites."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

from coevobench.core.evaluation.tasks import load_splits
from coevobench.core.protocol.protocol_assets import cache_root

CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
)


def component(task_id: str) -> str:
    for category in CATEGORIES:
        if task_id.startswith(category + "_"):
            domain = "memory" if category.startswith("memory") else "multi_turn"
            return f"{domain}:{category}"
    raise ValueError(f"unknown BFCL category: {task_id}")


def category(task_id: str) -> str:
    return component(task_id).split(":", 1)[1]


@lru_cache(maxsize=1)
def source_records() -> dict[str, dict]:
    from bfcl_eval.constants.default_prompts import (
        ADDITIONAL_SYSTEM_PROMPT_FOR_AGENTIC_RESPONSE_FORMAT,
    )
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    from coevobench.bench.bfcl.provenance import verify_upstream

    verify_upstream()
    records = {}
    for name in CATEGORIES:
        entries = load_dataset_entry(name)
        scored = [entry for entry in entries if "_prereq_" not in entry["id"]]
        answers = load_ground_truth_entry(name)
        if len(scored) != len(answers):
            raise ValueError(f"BFCL answer alignment differs for {name}")
        by_id = {
            entry["id"]: answer["ground_truth"]
            for entry, answer in zip(scored, answers, strict=True)
        }
        for entry in entries:
            entry = copy.deepcopy(entry)
            task_id = entry["id"]
            # This is inserted by load_dataset_entry using the stock prompt.
            # Generation inserts it again from the candidate prompt module.
            if name.startswith("memory") and task_id in by_id:
                first = entry["question"][0]
                if first[0].get("content") != ADDITIONAL_SYSTEM_PROMPT_FOR_AGENTIC_RESPONSE_FORMAT:
                    raise ValueError("unexpected upstream agentic prompt preprocessing")
                first.pop(0)
            records[task_id] = {
                "entry": entry,
                "category": name,
                "component": component(task_id),
                "scoring": task_id in by_id,
            }
            if task_id in by_id:
                records[task_id]["ground_truth"] = by_id[task_id]
    splits = load_splits("bfcl")
    frozen = set(splits.visible + splits.selection + splits.test)
    actual = {key for key, row in records.items() if row["scoring"]}
    if frozen != actual:
        raise ValueError("BFCL data does not match the frozen 1265-record pool")
    return records


def generation_plan(ids: list[str], records: dict[str, dict] | None = None) -> list[dict]:
    """Topological dependency closure; returns task inputs without answers."""
    records = records if records is not None else source_records()
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("BFCL needs unique, nonempty scoring ids")
    ordered, visiting, done = [], set(), set()

    def visit(task_id: str) -> None:
        if task_id in done:
            return
        if task_id in visiting or task_id not in records:
            raise ValueError("missing or cyclic BFCL prerequisite")
        visiting.add(task_id)
        row = records[task_id]
        for dependency in row["entry"].get("depends_on", []):
            visit(dependency)
        visiting.remove(task_id)
        done.add(task_id)
        ordered.append(copy.deepcopy(row["entry"]))

    for task_id in ids:
        if task_id not in records or not records[task_id]["scoring"]:
            raise ValueError("requested id is not a BFCL scoring record")
        visit(task_id)
    return ordered


def materialize_bfcl(workspace: Path, judge_cache: Path | None = None) -> dict:
    from coevobench.bench.bfcl.provenance import verify_upstream

    verify_upstream()
    records = source_records()
    judge_cache = judge_cache or cache_root() / "bfcl_tasks.jsonl"
    judge_cache.parent.mkdir(parents=True, exist_ok=True)
    judge_cache.write_text(
        "".join(json.dumps({"id": key, **row}) + "\n" for key, row in sorted(records.items()))
    )
    judge_cache.chmod(0o600)
    visible = list(load_splits("bfcl").visible)
    entries = generation_plan(visible, records)
    target = workspace / "tasks_visible" / "bfcl"
    target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (target / f"{entry['id']}.json").write_text(json.dumps(entry, indent=2) + "\n")
    return {
        "n_visible": len(visible),
        "n_prerequisites": len(entries) - len(visible),
        "n_scoring": 1265,
        "judge_cache": str(judge_cache),
    }
