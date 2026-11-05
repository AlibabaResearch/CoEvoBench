"""Reproduce successful-exit garbage, context errors and incomplete finalization."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from coevobench.core.grading.submission import inspect_submission, request
from coevobench.research.loop import LoopState
from coevobench.research.progress import Progress


def event(kind, block):
    return {"type": kind, "message": {"content": [block]}}


def test_only_novel_successful_tools_count_as_progress():
    progress = Progress()
    call = {"type": "tool_use", "name": "Bash", "id": "x", "input": {"command": "ls"}}
    progress.observe(event("assistant", call))
    assert progress.pending and not progress.completed
    progress.observe(event("user", {"type": "tool_result", "tool_use_id": "x",
                                     "content": "exit 1", "is_error": True}))
    assert not progress.pending and not progress.completed
    progress.observe(event("assistant", call))
    result = {"type": "tool_result", "tool_use_id": "x", "content": "file.py"}
    assert progress.observe(event("user", result))
    progress.observe(event("assistant", call))
    assert not progress.observe(event("user", result))
    assert progress.completed == 1
    progress.observe(event("assistant", {**call, "name": "Bash</arg_value>command"}))
    assert progress.malformed == 1 and not progress.pending
    progress.observe({"type": "assistant", "message": None})
    progress.observe(event("assistant", {"type": "text", "text": None}))
    progress.observe(event("assistant", {**call, "name": {"broken": "tool"}}))
    assert progress.malformed == 2


def test_context_error_starts_fresh_and_keeps_workspace_state():
    state = LoopState(resume=True)
    old = state.session
    progress = Progress()
    progress.observe(event("assistant", {"type": "text", "text":
        'API Error: 400 {"message":"Range of input length should be [1, 1048576]"}'}))
    state.advance(progress, 1, {"status": "incomplete"})
    assert not state.resume and state.session != old
    assert state.recoveries == 1


def test_success_exit_without_tools_cannot_loop_forever():
    state = LoopState()
    sessions = set()
    for _ in range(20):
        sessions.add(state.session)
        state.advance(Progress(), 0, {"status": "incomplete"})
        if state.phase != "running":
            break
    assert state.phase == "incomplete"
    assert state.mode == "finalize"
    assert len(sessions) > 1
    assert state.final_rounds <= 3


def test_submission_checks_are_cross_namespace_and_never_manufacture_signal(tmp_path):
    ws = tmp_path / "workspace"
    final = ws / "out/tb_final"
    (final / "harness").mkdir(parents=True)
    weights = ws / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    (final / "model_path.txt").write_text("/workspace/weights")
    for name in ("meta.json", "serve_config.json"):
        (final / name).write_text("{}")
    assert inspect_submission(ws, "tb")["status"] == "incomplete"
    (ws / "out/DONE.md").write_text("Finished with limitations.")
    assert inspect_submission(ws, "tb")["status"] == "unconfirmed"
    assert not (final / "submission.json").exists()
    request(ws, "tb")
    assert inspect_submission(ws, "tb")["status"] == "confirmed"
    (final / "meta.json").write_text('{"changed":true}')
    assert inspect_submission(ws, "tb")["status"] == "invalid"


@pytest.mark.parametrize("mode", ["recover", "exhaust", "finalize"])
def test_real_loop_with_cli_double(tmp_path, mode):
    ws = tmp_path / "workspace"
    (ws / "out").mkdir(parents=True)
    (ws / "keep.txt").write_text("research artifacts survive recovery")
    if mode == "finalize":
        (ws / "out/DONE.md").write_text("Premature DONE must not end this run")
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_text(json.dumps({"phase": "running", "deadline_at": time.time() + 3600}))
    executable = tmp_path / "claude"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json,sys
from pathlib import Path
from coevobench.core.grading.submission import request
ws=Path.cwd(); path=ws/'invocations.jsonl'
prior=path.read_text().splitlines() if path.exists() else []
with path.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
mode=''' + repr(mode) + '''
if mode=='exhaust' or (mode=='recover' and len(prior)<3):
 block={'type':'text','text':'</arg_value> ?? ≈ 随机'}
 print(json.dumps({'type':'assistant','message':{'content':[block]}}))
 print(json.dumps({'type':'result','is_error':False}))
 raise SystemExit(0)
if mode=='recover': assert '--session-id' in sys.argv
if mode=='finalize': assert 'Finalize now' in sys.argv[-1]
final=ws/'out/tb_final'; (final/'harness').mkdir(parents=True)
weights=ws/'weights'; weights.mkdir(); (weights/'config.json').write_text('{}')
(final/'model_path.txt').write_text(str(weights))
for name in ('meta.json','serve_config.json'): (final/name).write_text('{}')
(ws/'out/DONE.md').write_text('Final result, with limitations.')
request(ws,'tb')
print(json.dumps({'type':'result','is_error':False}))
''')
    executable.chmod(0o755)
    runner = (
        "from pathlib import Path; from coevobench.research.loop import run_loop; "
        f"raise SystemExit(run_loop(Path({str(ws)!r}),Path({str(lifecycle)!r}),"
        f"'tb','test',executable={str(executable)!r},pause_s=0))"
    )
    proc = subprocess.run([sys.executable, "-c", runner],
                          capture_output=True, text=True, timeout=20)
    state = json.loads((ws / "out/researcher-state.json").read_text())
    assert (ws / "keep.txt").read_text() == "research artifacts survive recovery"
    if mode == "exhaust":
        assert proc.returncode == 2
        assert state["phase"] == "incomplete" and state["iteration"] <= 12
        assert not (ws / "out/tb_final/submission.json").exists()
    else:
        assert proc.returncode == 0, proc.stderr
        assert state["phase"] == "submitted"
        assert inspect_submission(ws, "tb")["confirmed"]
    if mode == "recover":
        commands = [json.loads(line)
                    for line in (ws / "invocations.jsonl").read_text().splitlines()]
        assert "--session-id" in commands[0] and "--resume" in commands[1]
        assert "--session-id" in commands[3]
        assert commands[0][commands[0].index("--session-id") + 1] != state["session"]
