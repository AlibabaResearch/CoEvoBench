"""Runs only inside the isolated generation container; no private answers."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def load_policy_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate(input_path: Path, output: Path, policy: Path, options: dict, workers: int):
    # overrides resolves base classes from module globals. Import only after prompts.
    global QwenHandler
    # Load editable prompts before any BFCL imports bind their symbols.
    prompts = load_policy_module(
        "bfcl_eval.constants.default_prompts", policy / "default_prompts.py"
    )
    prompts.MAXIMUM_STEP_LIMIT = options["max_steps"]
    weak = load_policy_module("coevo_candidate_policy", policy / "weak_policy.py")
    import httpx
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    from bfcl_eval.utils import (
        make_json_serializable,
        populate_initial_settings_for_memory_test_cases,
        process_agentic_test_case,
    )
    from openai import OpenAI
    from overrides import override
    from transformers import AutoTokenizer

    class CandidateHandler(QwenHandler):
        @override
        def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
            data = super()._pre_query_processing_prompting(test_entry)
            data["coevo_turn"] = 0
            data["coevo_releases"] = {int(i) for i in test_entry.get("missed_function", {})}
            return data

        @override
        def add_first_turn_message_prompting(
            self, inference_data: dict, first_turn_message: list[dict]
        ) -> dict:
            weak.begin_turn(inference_data, first_turn_message, is_tool_update=False)
            inference_data["coevo_turn"] = 1
            return super().add_first_turn_message_prompting(inference_data, first_turn_message)

        @override
        def _add_next_turn_user_message_prompting(
            self, inference_data: dict, user_message: list[dict]
        ) -> dict:
            turn = inference_data["coevo_turn"]
            weak.begin_turn(
                inference_data,
                user_message,
                is_tool_update=turn in inference_data["coevo_releases"],
            )
            inference_data["coevo_turn"] += 1
            return super()._add_next_turn_user_message_prompting(inference_data, user_message)

        @override
        def _query_prompting(self, inference_data: dict):
            # Full transcript remains in inference_data; policy selects context only.
            prompt = self._format_prompt(
                weak.select_messages(inference_data), inference_data["function"]
            )
            inference_data["inference_input_log"] = {"formatted_prompt": prompt}
            n = len(self.tokenizer.tokenize(prompt))
            limit = options["context_limit"]
            tokens = min(options["max_tokens"], limit - n - 2) if limit >= n + 2 else 1000
            started = time.monotonic()
            response = self.client.completions.create(
                model=options["name"],
                prompt=prompt,
                temperature=options["temperature"],
                max_tokens=tokens,
                timeout=72000,
            )
            return response, time.monotonic() - started

    handler = CandidateHandler(options["name"], options["temperature"], "coevo-bfcl", False)
    handler.client.close()
    handler.client = OpenAI(
        base_url="http://model/v1",
        api_key="EMPTY",
        http_client=httpx.Client(transport=httpx.HTTPTransport(uds="/model.sock")),
    )
    handler.tokenizer = AutoTokenizer.from_pretrained(
        "/tokenizer", local_files_only=True, trust_remote_code=False
    )
    entries = process_agentic_test_case(json.loads(input_path.read_text()))
    entries = populate_initial_settings_for_memory_test_cases(entries, output.parent)
    pending = {entry["id"]: entry for entry in entries}
    completed, failed = set(), set()
    result_file = output.open("w")

    def write_row(row):
        result_file.write(json.dumps(make_json_serializable(row)) + "\n")
        result_file.flush()

    def one(entry):
        try:
            result, metadata = handler.inference(copy.deepcopy(entry), True, False)
            return {"id": entry["id"], "result": result, **metadata}
        except Exception as exc:
            # No swallowed endpoint/tool exceptions masquerading as model failures.
            return {"id": entry["id"], "result": None, "infrastructure_error": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while pending or active:
            for task_id, entry in list(pending.items()):
                if len(active) >= workers:
                    break
                dependencies = set(entry.get("depends_on", []))
                if dependencies & failed:
                    write_row(
                        {
                            "id": task_id,
                            "result": None,
                            "infrastructure_error": "PrerequisiteFailed",
                        }
                    )
                    failed.add(task_id)
                    del pending[task_id]
                elif dependencies <= completed:
                    active[pool.submit(one, entry)] = task_id
                    del pending[task_id]
            if not active:
                if pending:
                    raise ValueError("unresolved BFCL dependencies")
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = active.pop(future)
                row = future.result()
                write_row(row)
                (failed if row.get("infrastructure_error") else completed).add(task_id)
    handler.client.close()
    result_file.close()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate(
        Path("/input/tasks.json"),
        Path("/output/generation.jsonl"),
        Path("/policy"),
        json.loads(Path("/input/options.json").read_text()),
        int(sys.argv[1]),
    )
