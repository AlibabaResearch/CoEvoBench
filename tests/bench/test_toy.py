import json

from coevobench.bench.toy.adapter import SPLITS, TASKS, ToyBench
from coevobench.core.protocol.spec import TaskView


def test_splits_partition_tasks() -> None:
    all_ids = set(TASKS)
    split_ids = {tid for ids in SPLITS.values() for tid in ids}
    assert split_ids == all_ids
    assert len(set(SPLITS["visible"]) & set(SPLITS["test"])) == 0


def test_public_view_strips_answer() -> None:
    bench = ToyBench()
    view: TaskView = bench.public_view("toy-sum-1")
    assert str(sum(i * i for i in range(1, 11))) not in view.prompt
    assert "sum of squares" in view.prompt
    assert view.image == "python:3.11-slim"


def test_judge_requires_answer_json() -> None:
    class FakeSandbox:
        def read_file(self, path: str) -> str:
            assert path == "answer.json"
            return str(sum(i * i for i in range(1, 101)))

    verdict = ToyBench().judge("toy-sum-2", "submitted", FakeSandbox())
    assert verdict.resolved
    assert verdict.score == 1.0
    assert verdict.exit_status == "Submitted"


def test_judge_format_error_on_garbage() -> None:
    class BadSandbox:
        def read_file(self, path: str) -> str:
            return "not-json"

    verdict = ToyBench().judge("toy-sum-2", "", BadSandbox())
    assert not verdict.resolved
    assert verdict.exit_status == "FormatError"


def test_rollout_script_writes_valid_json() -> None:
    script = ToyBench().rollout_script("toy-sum-1")
    assert "answer.json" in script
    json.loads(script.split("json.dump(")[1].split(", open")[0])
