"""Official data/checker tests; no model API, GPU, or fabricated grader."""

import copy
import json

import pytest

from coevobench.bench.bfcl.provenance import VENDOR_ROOT, verify_upstream
from coevobench.bench.bfcl.replay import aggregate_score
from coevobench.bench.bfcl.runtime import grading_env, read_results
from coevobench.bench.bfcl.tasks import CATEGORIES, generation_plan, source_records
from coevobench.core.evaluation.evaluate import evaluate_split
from coevobench.core.evaluation.tasks import load_splits
from coevobench.core.protocol.spec import Verdict


@pytest.fixture
def records():
    pytest.importorskip("bfcl_eval")
    return source_records()


def test_official_pool_and_dependency_closure(records):
    splits = load_splits("bfcl")
    assert [len(splits.ids(name)) for name in ("visible", "selection", "test")] == [235, 470, 560]
    assert sum(row["scoring"] for row in records.values()) == 1265
    for split in (splits.visible, splits.selection, splits.test):
        seen = set()
        for entry in generation_plan(list(split)):
            assert set(entry.get("depends_on", [])) <= seen
            assert "ground_truth" not in entry
            seen.add(entry["id"])
    # Variants and memory backends/scenarios must remain together.
    owner = {key: name for name in ("visible", "selection", "test") for key in splits.ids(name)}
    families = {}
    for key, name in owner.items():
        row = records[key]
        family = (
            ("memory", row["entry"]["scenario"])
            if key.startswith("memory")
            else ("multi", key.rsplit("_", 1)[1])
        )
        families.setdefault(family, set()).add(name)
    assert all(len(names) == 1 for names in families.values())


def test_generation_plan_rejects_duplicates_and_missing_dependency():
    records = {"a": {"entry": {"id": "a", "depends_on": ["missing"]}, "scoring": True}}
    with pytest.raises(ValueError, match="unique"):
        generation_plan(["a", "a"], records)
    with pytest.raises(ValueError, match="missing"):
        generation_plan(["a"], records)


def test_weighted_score_and_repeat_rows(tmp_path):
    ids = ["multi_turn_base_0", "multi_turn_base_1", "multi_turn_miss_func_0", "memory_kv_0-x"]

    class Judge:
        aggregate_score = staticmethod(aggregate_score)

        def evaluate_batch(self, task_ids, k, workers):
            assert task_ids == ids and k == 2
            return [Verdict(bool(s), s, "Submitted") for s in [1, 1, 0, 0, 0, 0, 1, 1]]

        def __call__(self, task_id):
            return self.evaluate_batch([task_id], 1, 1)[0]

    record = evaluate_split(
        run_id="r",
        candidate="c",
        split="visible",
        task_ids=ids,
        judge_fn=Judge(),
        k=2,
        out_dir=tmp_path,
    )
    assert record.score == pytest.approx(0.5)
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()][
        1:
    ]
    assert [row["score"] for row in rows] == [1, 1, 0, 0, 0, 0, 1, 1]
    assert aggregate_score(
        ids, [Verdict(True, s, "Submitted") for s in [1, 1, 0, 0]], 1
    ) == pytest.approx(0.3)


def test_official_checkers_accept_oracle_and_reject_wrong_outputs(records):
    from bfcl_eval.eval_checker.eval_runner import (
        _evaluate_single_agentic_entry,
        _evaluate_single_multi_turn_entry,
    )
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler

    handler = QwenHandler("coevo-bfcl", 0, "coevo-bfcl", False)
    try:
        for category in CATEGORIES:
            key = (
                category + "_10"
                if category.startswith("multi_turn")
                else next(
                    key
                    for key in load_splits("bfcl").visible
                    if records[key]["category"] == category
                )
            )
            assert key in load_splits("bfcl").visible
            row = records[key]
            if category.startswith("memory"):
                checker = _evaluate_single_agentic_entry
                good = [
                    [json.dumps({"answer": row["ground_truth"][0], "context": "oracle fixture"})]
                ]
                wrong = [["{'answer':'UNMATCHED_ORACLE_FIXTURE','context':'fixture'}"]]
            else:
                checker = _evaluate_single_multi_turn_entry
                good = [["[" + ", ".join(turn) + "]"] for turn in row["ground_truth"]]
                wrong = [["[]"] for _ in row["ground_truth"]]
            passed = checker(
                handler,
                key,
                good,
                row["ground_truth"],
                copy.deepcopy(row["entry"]),
                "coevo-bfcl",
                category,
            )
            failed = checker(
                handler,
                key,
                wrong,
                row["ground_truth"],
                copy.deepcopy(row["entry"]),
                "coevo-bfcl",
                category,
            )
            assert passed["valid"] is True, category
            assert failed["valid"] is False, category
    finally:
        handler.client.close()


def test_grading_environment_and_result_completeness(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_TOKEN", "test-secret")
    monkeypatch.setenv("BFCL_POLICY_DIR", "/untrusted")
    env = grading_env(tmp_path)
    assert "RESEARCHER_TOKEN" not in env and "BFCL_POLICY_DIR" not in env
    (tmp_path / "grading-input.json").write_text(json.dumps({"records": {"a": {}, "b": {}}}))
    (tmp_path / "verdicts.jsonl").write_text('{"id":"a","correct":true}\n')
    with pytest.raises(ValueError, match="incomplete"):
        read_results(tmp_path)


def test_upstream_manifest_detects_drift(tmp_path):
    import shutil

    shutil.copytree(
        VENDOR_ROOT / "bfcl_eval",
        tmp_path / "bfcl_eval",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(VENDOR_ROOT / "pyproject.toml", tmp_path)
    verify_upstream(tmp_path)
    (tmp_path / "bfcl_eval/constants/default_prompts.py").write_text("tampered = True\n")
    with pytest.raises(ValueError, match="changed"):
        verify_upstream(tmp_path)


def test_broker_refuses_test_ids_and_workspace_pool_excludes_test(tmp_path):
    from coevobench.bench.anon import load_mapping
    from coevobench.orchestrate.provision import provision_workspace
    from coevobench.research.server import Broker

    workspace = tmp_path / "workspace"
    provision_workspace(workspace, "bfcl")
    mapping = load_mapping("bfcl")
    splits = load_splits("bfcl")
    pool = set((workspace / "anonymous_pool_bfcl.txt").read_text().splitlines())
    assert pool == {mapping[key] for key in splits.selection}
    broker = Broker(workspace_root=workspace, bench="bfcl")
    with pytest.raises(ValueError, match="unknown anonymous"):
        broker.dispatch(
            {
                "op": "eval.run",
                "bench": "bfcl",
                "split": "selection",
                "tasks": [mapping[splits.test[0]]],
            }
        )
