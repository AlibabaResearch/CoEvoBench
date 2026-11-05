"""SWE adapter: protection gates and apply/judge flow (no docker needed)."""

from __future__ import annotations

import json
from pathlib import Path

from coevobench.bench.swe.adapter import SweBench

INSTANCE = {
    "problem_statement": "Fix the parser crash on empty input.",
    "test_patch": "--- a/tests/test_parser.py\n+++ b/tests/test_parser.py\n",
    "test_cmd": "pytest tests/ -q",
}


def _swe(tmp_path: Path) -> SweBench:
    ids = [f"astropy__astropy-{i}" for i in range(493)]
    (tmp_path / "swe_splits.json").write_text(
        json.dumps(
            {"visible": ids[:40], "selection": ids[40:240], "test": ids[240:493]}
        )
    )
    return SweBench(tmp_path, instances={tid: INSTANCE for tid in ids})


def test_frozen_sizes_validated_on_load(tmp_path: Path) -> None:
    bench = _swe(tmp_path)
    assert len(bench.task_ids("visible")) == 40
    assert len(bench.task_ids("test")) == 253


def test_public_view_strips_gold(tmp_path: Path) -> None:
    view = _swe(tmp_path).public_view("astropy__astropy-0")
    assert "problem_statement" not in view.prompt
    assert "parser crash" in view.prompt
    assert "test_patch" not in view.prompt


def test_protected_test_files_rejected(tmp_path: Path) -> None:
    bench = _swe(tmp_path)
    malicious = (
        "--- a/tests/test_parser.py\n+++ b/tests/test_parser.py\n"
        "@@ -1 +1 @@\n-assert False\n+assert True\n"
    )
    verdict = bench.judge("astropy__astropy-0", malicious, _OkSandbox())
    assert verdict.exit_status == "ProtectedTestFile"
    assert verdict.score == 0.0


def test_empty_patch_is_not_resolved(tmp_path: Path) -> None:
    verdict = _swe(tmp_path).judge("astropy__astropy-0", "", _OkSandbox())
    assert verdict.exit_status == "EmptyPatch"


def test_apply_failure_short_circuits(tmp_path: Path) -> None:
    verdict = _swe(tmp_path).judge(
        "astropy__astropy-0", "diff --git a/x b/x\n", _FailApplySandbox()
    )
    assert verdict.exit_status == "PatchApplyFailed"


class _OkSandbox:
    def write_files(self, files) -> None: ...

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> dict:
        if "apply_one" in command or "CODE_M" in command:
            return {
                "exit_code": 0,
                "text": "CODE_M=git apply CODE_RC=0\nTEST_M=git apply TEST_RC=0",
            }
        return {"exit_code": 0, "text": "exit=0\n5 passed"}

    def read_file(self, path: str) -> str:
        return "5 passed, 0 failed"


class _FailApplySandbox:
    def write_files(self, files) -> None: ...

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None) -> dict:
        return {"exit_code": 1, "text": "CODE_M=fatal CODE_RC=1\nTEST_M=git apply TEST_RC=0"}

    def read_file(self, path: str) -> str:
        return ""


class _CalibrationSandbox:
    """Fresh-sandbox double: gold resolves, empty fails."""

    def __init__(self) -> None:
        self.files: list[str] = []

    def write_files(self, files) -> None:
        self.files.extend(f.path for f in files)

    def run(self, command: str, *, timeout_s: int = 180, cwd: str | None = None):
        if "apply_one" in command:
            return {"exit_code": 0, "text": "CODE_M=ok CODE_RC=0 TEST_M=ok TEST_RC=0"}
        return {"exit_code": 0, "text": "exit=0"}

    def read_file(self, path: str) -> str:
        return "2 passed"

    def kill(self) -> None:
        pass


class _CalibrationRuntime:
    def __init__(self, resolves_gold: bool) -> None:
        self._resolves_gold = resolves_gold
        self.created = 0

    def create(self, spec, ready_timeout_s: int | None = None):
        self.created += 1
        box = _CalibrationSandbox()
        if not self._resolves_gold:
            box.read_file = lambda path: "1 failed"
        return box


GOLD_INSTANCE = {
    **INSTANCE,
    "patch": "diff --git a/parser.py b/parser.py\n--- a/parser.py\n+++ b/parser.py\n",
}


def _calibration_bench(tmp_path: Path) -> SweBench:
    all_ids = [f"astropy__astropy-{i}" for i in range(493)]
    (tmp_path / "swe_splits.json").write_text(
        json.dumps(
            {"visible": all_ids[:40], "selection": all_ids[40:240], "test": all_ids[240:]}
        )
    )
    instances = {tid: GOLD_INSTANCE for tid in all_ids}
    instances["astropy__astropy-12907"] = GOLD_INSTANCE
    return SweBench(tmp_path, instances=instances)


def test_calibration_two_gate(tmp_path: Path) -> None:
    from coevobench.bench.swe.adapter import calibrate

    bench = _calibration_bench(tmp_path)
    state = tmp_path / "trusted.json"
    sampled = ["astropy__astropy-12907", "astropy__astropy-1"]
    result = calibrate(
        bench, sampled, runtime=_CalibrationRuntime(True), trusted_state=state
    )
    assert result["trusted"] is True
    assert result["n"] == len(sampled)
    assert json.loads(state.read_text())["trusted"] is True


def test_calibration_flags_broken_graders(tmp_path: Path) -> None:
    from coevobench.bench.swe.adapter import calibrate

    bench = _calibration_bench(tmp_path)
    sampled = ["astropy__astropy-12907", "astropy__astropy-1"]
    result = calibrate(bench, sampled, runtime=_CalibrationRuntime(False))
    assert result["trusted"] is False
    assert result["gold_failed"] == sampled


def test_calibration_unknown_instance_fails_gold_gate(tmp_path: Path) -> None:
    from coevobench.bench.swe.adapter import calibrate

    bench = _calibration_bench(tmp_path)
    result = calibrate(bench, ["repo__missing-1"], runtime=_CalibrationRuntime(True))
    assert result["gold_failed"] == ["repo__missing-1"]
    assert result["trusted"] is False


def test_repo_env_prep_dispatch() -> None:
    from coevobench.bench.swe.adapter import _repo_env_prep

    assert _repo_env_prep({"repo": "psf/requests"}) .startswith("python -m pip install -q httpbin")
    assert "httpbin.org" in _repo_env_prep({"repo": "psf/requests"})
    assert "tomlkit==0.11.1" in _repo_env_prep(
        {"repo": "pylint-dev/pylint", "version": "2.14"}
    )
    assert "tomlkit" not in _repo_env_prep(
        {"repo": "pylint-dev/pylint", "version": "3.0"}
    )
    assert _repo_env_prep({"repo": "astropy/astropy"}) == ""
    assert _repo_env_prep({}) == ""
