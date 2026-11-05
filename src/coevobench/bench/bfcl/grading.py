"""Trusted grading subprocess: official checker, never imports candidate code."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


def grade(project: Path) -> list[dict]:
    from bfcl_eval.eval_checker.eval_runner import (
        _evaluate_single_agentic_entry,
        _evaluate_single_multi_turn_entry,
    )
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler

    requested = json.loads((project / "grading-input.json").read_text())
    by_id = {}
    with (project / "generation.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in by_id:
                raise ValueError("duplicate generation records")
            by_id[row["id"]] = {
                key: row[key] for key in ("id", "result", "infrastructure_error") if key in row
            }
    expected = {row["id"] for row in requested["entries"]}
    if set(by_id) != expected:
        raise ValueError("missing or extra generation records")
    handler = QwenHandler("coevo-bfcl", 0, "coevo-bfcl", False)
    results = []
    for task_id, record in requested["records"].items():
        row = by_id[task_id]
        closure = set(record["entry"].get("depends_on", [])) | {task_id}
        if any(by_id[dep].get("infrastructure_error") for dep in closure):
            results.append(
                {"id": task_id, "correct": False, "exit_status": "HarnessInfra:BFCLGeneration"}
            )
            continue
        checker = (
            _evaluate_single_agentic_entry
            if record["category"].startswith("memory")
            else _evaluate_single_multi_turn_entry
        )
        verdict = checker(
            handler,
            task_id,
            row["result"],
            record["ground_truth"],
            copy.deepcopy(record["entry"]),
            "coevo-bfcl",
            record["category"],
        )
        if type(verdict.get("valid")) is not bool:
            raise ValueError("official checker did not return a boolean verdict")
        results.append(
            {
                "id": task_id,
                "correct": verdict["valid"],
                "exit_status": verdict_status(verdict),
                "official": verdict,
            }
        )
    handler.client.close()
    return results


def verdict_status(verdict: dict) -> str:
    """Completion status is separate from correctness (a wrong answer can finish)."""
    error = verdict.get("error", {})
    kind = error.get("error_type", "") if isinstance(error, dict) else ""
    if kind.endswith(":inference_error"):
        return "FormatError"
    if kind.endswith(":force_terminated") or kind.endswith(":no_last_message"):
        return "StepLimitExceeded"
    return "Submitted"


if __name__ == "__main__":
    project = Path(sys.argv[1])
    Path(sys.argv[2]).write_text("".join(json.dumps(row) + "\n" for row in grade(project)))
