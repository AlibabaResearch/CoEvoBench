"""Real isolated generation and official grading with a local deterministic API."""

import json
import os
import threading
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from coevobench.bench.bfcl.runtime import image_identity, read_results, run_repeat, settings
from coevobench.bench.bfcl.tasks import source_records
from coevobench.core.evaluation.tasks import load_splits

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def endpoint(responses):
    queue = deque(responses)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            value = queue.popleft() if queue else "Done."
            payload = json.dumps(
                {
                    "id": "offline-fixture",
                    "object": "text_completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [{"text": value, "index": 0, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def tokenizer():
    import subprocess

    path = Path(os.environ.get("BFCL_TEST_TOKENIZER", Path.home() / "models/Qwen3.5-9B-Base"))
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip("set BFCL_TEST_TOKENIZER to a local tokenizer")
    try:
        image_identity()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("build the isolated BFCL image first")
    return path


def test_real_generation_and_official_grading(tmp_path, tokenizer):
    import shutil

    records = source_records()
    task = "multi_turn_base_10"
    assert task in load_splits("bfcl").visible
    responses = []
    for turn in records[task]["ground_truth"]:
        if turn:
            responses.append("[" + ", ".join(turn) + "]")
        responses.append("Done.")
    policy = tmp_path / "policy"
    shutil.copytree(ROOT / "templates/bfcl/harness/src/bfclharness", policy)
    weak = policy / "weak_policy.py"
    weak.write_text(
        """from pathlib import Path
import os
import socket
assert not Path('/project').exists()
assert not Path('/opt/engine/bfcl_eval/data').exists()
assert 'RESEARCHER_TOKEN' not in os.environ
assert not Path('/var/run/docker.sock').exists()
s = socket.socket()
s.settimeout(0.1)
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('generation unexpectedly has external network')
finally:
    s.close()
"""
        + weak.read_text()
    )
    with endpoint(responses) as (url, requests):
        project = run_repeat(
            [task],
            0,
            settings(tokenizer=str(tokenizer), base_url=url),
            policy,
            tmp_path,
            workers=1,
        )
    verdicts = read_results(project)
    assert len(verdicts) == 1 and verdicts[0]["correct"] is True
    assert requests and all(
        r["model"] == "qwen3.5-9b" and r["max_tokens"] <= 4096 for r in requests
    )
    assert all(r["temperature"] == 0.001 for r in requests)


@pytest.mark.parametrize("backend", ["kv", "rec_sum", "vector"])
def test_memory_prerequisites_replayed_from_empty_state(tmp_path, tokenizer, backend):
    task = next(key for key in load_splits("bfcl").visible if key.startswith(f"memory_{backend}_"))
    with endpoint([]) as (url, requests):
        options = settings(tokenizer=str(tokenizer), base_url=url)
        for sample in range(2):
            project = run_repeat(
                [task],
                sample,
                options,
                ROOT / "templates/bfcl/harness/src/bfclharness",
                tmp_path,
                workers=2,
            )
            rows = [
                json.loads(line) for line in (project / "generation.jsonl").read_text().splitlines()
            ]
            assert any("_prereq_" in row["id"] for row in rows)
            assert not any(row.get("infrastructure_error") for row in rows)
            assert read_results(project)[0]["correct"] is False
        assert requests
    first = [
        json.loads(line)["id"]
        for line in (tmp_path / "repeat_0/generation.jsonl").read_text().splitlines()
    ]
    second = [
        json.loads(line)["id"]
        for line in (tmp_path / "repeat_1/generation.jsonl").read_text().splitlines()
    ]
    assert set(first) == set(second)


def test_broker_selection_and_frozen_weight_reversion(tmp_path, tokenizer, monkeypatch):
    import shutil

    from coevobench.bench.bfcl import replay
    from coevobench.cli.deliver import _register_bench_bindings
    from coevobench.core.lifecycle.final_eval import evaluate, prepare
    from coevobench.core.protocol.splits import from_mapping
    from coevobench.orchestrate.provision import provision_workspace
    from coevobench.research.server import Broker

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for pattern in ("*.json", "*.txt"):
        for path in tokenizer.glob(pattern):
            shutil.copyfile(path, checkpoint / path.name)
    (checkpoint / "model.safetensors").write_bytes(
        b"offline fixture; serving is replaced by test API"
    )
    workspace = tmp_path / "run/workspace"
    provision_workspace(workspace, "bfcl")
    (workspace.parent / "base-model.json").write_text(json.dumps({"path": str(checkpoint)}))
    served_weights = []
    with endpoint([]) as (url, _requests):

        @contextmanager
        def serve(candidate, **kwargs):
            served_weights.append(candidate.weights)
            yield {"model_name": "qwen3.5-9b", "base_url": url}

        monkeypatch.setattr(replay, "candidate_serve", serve)
        response = Broker(workspace_root=workspace, bench="bfcl").dispatch(
            {
                "op": "eval.run",
                "bench": "bfcl",
                "split": "selection",
                "model": str(checkpoint),
                "n": 1,
            }
        )
        assert response["score"] == 0.0 and response["feedback"].startswith("score_mean=")
        assert set(response) >= {"n_tasks", "k", "score", "feedback"}
        assert list((workspace.parent / "select_runs").glob("*.jsonl"))

    submission = workspace / "out/candidate"
    submission.mkdir(parents=True)
    shutil.copytree(workspace / "harness", submission / "harness")
    (submission / "model_path.txt").write_text(str(checkpoint))
    (submission / "meta.json").write_text('{"bench":"bfcl"}')
    (submission / "serve_config.json").write_text("{}")
    bundle = prepare(submission, "bfcl", tmp_path / "final")
    _register_bench_bindings("bfcl")
    task = "multi_turn_base_10"
    responses = []
    for turn in source_records()[task]["ground_truth"]:
        if turn:
            responses.append("[" + ", ".join(turn) + "]")
        responses.append("Done.")
    with endpoint(responses * 2) as (url, _requests):
        for override in (None, checkpoint):
            report = evaluate(
                bundle,
                bench="bfcl",
                splits=from_mapping({"test": [task]}),
                weights_override=override,
            )
            assert report.score == 1.0 and report.reverted == (override is not None)
    assert served_weights[-2:] == [bundle / "weights", checkpoint]
