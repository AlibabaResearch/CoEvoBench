"""SWE-bench Verified adapter (design document Section 4.3).

Grading applies the submission
patch and the official test patch inside a fresh sandbox, run the task's
test command, and parse the log. The anti-cheating gate rejects patches
that touch protected test files (touching a gold test file
scores negative).

Splits and per-instance specs come from the fetched dataset; this module
holds only protocol constants, so the frozen numbers stay reviewable in
one place (paper Table 3 and Table 5).
"""

from __future__ import annotations

import time
from pathlib import Path

from coevobench.bench.adapter import FileBench
from coevobench.core.protocol.spec import TaskView, Verdict
from coevobench.core.protocol.splits import validate_frozen

WALL_LIMIT_S = 900
STEP_LIMIT = 80
OUTPUT_TOKENS = 5120
CONTEXT_LENGTH = 32768

PROTECTED_PATH_MARKERS = (
    "test_patch",
    "tests/",
    "test_",
    "_test.py",
)

APPLY_SCRIPT = """set +e
cd /testbed
apply_one() {
  local f="$1"
  [ -s "$f" ] || { echo empty; return 0; }
  for st in "apply --verbose" "apply -3 --verbose" "apply -C1 --verbose" \\
            "apply --recount --verbose" "apply --reject --verbose"; do
    if git $st "$f" >/tmp/ga.out 2>&1; then echo "git ${st#apply }"; return 0; fi
  done
  tail -1 /tmp/ga.out; return 1
}
CODE_M=$(apply_one /tmp/code.patch); CODE_RC=$?
TEST_M=$(apply_one /tmp/test.patch); TEST_RC=$?
echo "CODE_M=${CODE_M} CODE_RC=${CODE_RC}"
echo "TEST_M=${TEST_M} TEST_RC=${TEST_RC}"
"""


ACTIVATE = (
    "source /opt/miniconda3/bin/activate testbed 2>/dev/null || "
    "source /opt/conda/bin/activate testbed 2>/dev/null || true; "
    "PYBIN=/opt/miniconda3/envs/testbed/bin/python; "
    "[ -x $PYBIN ] || PYBIN=/opt/conda/envs/testbed/bin/python; "
    "[ -x $PYBIN ] || PYBIN=python"
)

def _load_repo_prep(name: str) -> str:
    """Read a per-repo environment-prep script shipped next to the adapter."""
    return (Path(__file__).parent / f"prep-{name}.sh").read_text()


_PYLINT_PREP = (
    "python -m pip install -q appdirs >/tmp/prep.log 2>&1 || true\n"
)


def _repo_env_prep(instance: dict) -> str:
    """Per-repo environment alignment: install missing deps / start local services —
    never touch judging logic or test targets.

    - psf/requests: the legacy suite mixes httpbin() (honors
      HTTPBIN_URL) with cases hardcoding https://httpbin.org; offline
      sandboxes would 503/timeout. A dual-scheme local httpbin (:80
      http, :443 https) with httpbin.org resolving to loopback and a
      self-signed CA (SAN: httpbin.org/localhost) makes both classes
      of tests hit the local mirror — responses match the official
      service, only offline-stable.
    - pylint: some base_commits' conftest needs appdirs; pylint 2.14's
      loose `pip install -e .` resolves tomlkit too new (dotted-key
      strictness breaks test_generate_toml_config* P2P). Pin to the
      2022-era tomlkit 0.11.1 the version shipped with (both 2.14
      instances verified gold RESOLVED_FULL).
    """
    repo = instance.get("repo", "")
    if repo == "psf/requests":
        return _load_repo_prep("httpbin")
    if repo == "pylint-dev/pylint":
        pins = (
            "tomlkit==0.11.1 appdirs"
            if str(instance.get("version", "")) == "2.14"
            else "appdirs"
        )
        return f"python -m pip install -q {pins} >/tmp/prep.log 2>&1 || true\n"
    return ""


class SweBench(FileBench):
    name = "swe"

    def __init__(self, data_root: Path, instances: dict[str, dict] | None = None) -> None:
        super().__init__(data_root)
        self._instances = instances or {}
        validate_frozen(self.name, self.splits)

    def public_view(self, task_id: str) -> TaskView:
        instance = self._instances[task_id]
        return TaskView(
            task_id=task_id,
            prompt=instance["problem_statement"],
            image=f"{instance.get('image_namespace', 'swebench')}/{task_id}",
        )

    def touches_protected_tests(self, submission: str) -> bool:
        """The anti-cheating gate: gold test files are off-limits."""
        for line in submission.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith(("+++", "---")):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) < 2:
                continue
            path = parts[1].strip().removeprefix("a/").removeprefix("b/")
            if any(marker in path for marker in PROTECTED_PATH_MARKERS):
                return True
        return False

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict:
        if self.touches_protected_tests(submission):
            return Verdict(resolved=False, score=0.0, exit_status="ProtectedTestFile")
        if not submission.strip():
            return Verdict(resolved=False, score=0.0, exit_status="EmptyPatch")

        sandbox.write_files([_file("/tmp/code.patch", submission)])
        test_patch = self._instances[task_id].get("test_patch", "")
        sandbox.write_files([_file("/tmp/test.patch", test_patch)])
        apply_result = sandbox.run(APPLY_SCRIPT, timeout_s=180)
        if "CODE_RC=0" not in apply_result["text"] or "TEST_RC=0" not in apply_result["text"]:
            return Verdict(resolved=False, score=0.0, exit_status="PatchApplyFailed")

        instance = self._instances[task_id]
        raw_cmd = instance.get("test_cmd", "pytest -rA")
        test_cmd = raw_cmd.replace("python -m pytest", "$PYBIN -m pytest")
        test_cmd = test_cmd.replace("pytest", "$PYBIN -m pytest")
        targets = _directive_tests(instance)
        if targets:
            test_cmd = f"$PYBIN -m pytest {targets} -q"
        run = sandbox.run(
            f"cd /testbed && {ACTIVATE} && {_repo_env_prep(instance)}"
            f"({test_cmd}) > /tmp/test.log 2>&1; echo exit=$?",
            timeout_s=WALL_LIMIT_S,
        )
        log = _read_log(sandbox)
        lower_log = log.lower()
        resolved = "exit=0" in run["text"] and " failed" not in lower_log.split("passed")[0]
        if targets:
            expected = _directive_count(instance)
            resolved = resolved and f"{expected} passed" in log
        return Verdict(
            resolved=resolved,
            score=float(resolved),
            exit_status="Submitted" if resolved else "TestFailed",
        )


def _directive_tests(instance: dict) -> str:
    """FAIL_TO_PASS + PASS_TO_PASS test ids, quoted for the pytest CLI."""
    import json as _json

    targets: list[str] = []
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        raw = instance.get(key, "")
        if not raw:
            continue
        try:
            parsed = _json.loads(raw)
        except ValueError:
            parsed = [t.strip() for t in raw.split(",") if t.strip()]
        targets.extend(t for t in parsed if t)
    return " ".join(f'"{t}"' for t in targets)


def _directive_count(instance: dict) -> int:
    return len(_directive_tests(instance).split('"')) // 2


def _file(path: str, data: str):
    from coevobench.runtime.sandbox.base import FileWrite

    return FileWrite(path=path, data=data, mode=0o644)


def _read_log(sandbox: object) -> str:
    try:
        return sandbox.read_file("/tmp/test.log")
    except Exception:
        return ""


def calibrate(
    bench: SweBench,
    task_ids: list[str],
    *,
    runtime,
    trusted_state: Path | None = None,
) -> dict:
    """Two-gate oracle self-calibration (paper: judge reliability is
    decided by pre-registered calibration, never by candidate scores).

    For every sampled instance, judged in a fresh sandbox:
    - gold gate: the instance's own gold patch must resolve
    - empty gate: an empty submission must fail

    Any gold failure marks the instance untrusted (its grader or image
    is broken — scoring it would be noise); any empty-gate success marks
    the whole run untrusted (tests that pass without the fix measure
    nothing). The result persists to `trusted_state` so evaluation can
    refuse untrusted instances loudly instead of grading them silently.

    Returns {"n": len(task_ids), "gold_failed": [...], "empty_passed":
    [...], "trusted": bool}.
    """
    import json

    from coevobench.runtime.sandbox.base import SandboxSpec

    gold_failed: list[str] = []
    empty_passed: list[str] = []
    for task_id in task_ids:
        instance = bench._instances.get(task_id)
        if instance is None:
            gold_failed.append(task_id)
            continue
        spec = SandboxSpec(
            image=bench.public_view(task_id).image, cpu=2, mem_gib=8, ttl_s=1800
        )
        sandbox = runtime.create(spec, ready_timeout_s=600)
        try:
            gold = bench.judge(task_id, instance.get("patch", ""), sandbox)
            if not gold.resolved:
                gold_failed.append(task_id)
                continue
        finally:
            sandbox.kill()
        sandbox = runtime.create(spec, ready_timeout_s=600)
        try:
            empty = bench.judge(task_id, "", sandbox)
            if empty.resolved:
                empty_passed.append(task_id)
        finally:
            sandbox.kill()

    result = {
        "n": len(task_ids),
        "gold_failed": gold_failed,
        "empty_passed": empty_passed,
        "trusted": not gold_failed and not empty_passed,
        "calibrated_at": time.time(),
    }
    if trusted_state is not None:
        trusted_state = Path(trusted_state)
        trusted_state.parent.mkdir(parents=True, exist_ok=True)
        trusted_state.write_text(json.dumps(result, indent=2))
    return result


__all__ = [
    "SweBench",
    "WALL_LIMIT_S",
    "calibrate",
]
