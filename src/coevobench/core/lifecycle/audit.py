"""Audit evidence pack (design doc Section 5.6; paper Section 3.4).

The paper requires manual audit of researcher trajectories after each
experiment ("we manually audit researcher trajectories for unauthorized
access to evaluation data or manipulation of evaluation results").
This module compiles the evidence a human needs, from the operator-side
audit directory plus bounded scans of the recorded command log:

- command log review: BLOCK/SUSPECT verdict counts and the actual hits
- external-model calls: markers for known LLM API surfaces and key
  names (the researcher's scoped endpoint and the teacher proxy are
  the sanctioned paths; anything else is a signal)
- benchmark-query signals: known benchmark/dataset identifiers,
  trajectory archives, and gold-patch terminology

Every hit is a review signal, never an automatic verdict — the human
decides. Scans are byte-capped per file and skip weight directories;
credentials are redacted from snippets before anything is rendered.

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MAX_SCAN_BYTES = 8 * 1024 * 1024
SKIP_DIR_NAMES = {
    "__pycache__", ".git", "node_modules", "model", "models",
    "checkpoints", "ckpt", "weights",
}

EXTERNAL_API_MARKERS = [
    r"api\.openai\.com", r"api\.anthropic\.com", r"dashscope",
    r"generativelanguage\.googleapis\.com", r"api\.deepseek\.com",
    r"api\.mistral\.ai", r"api\.groq\.com", r"api\.together\.xyz",
    r"openai\.ChatCompletion", r"openai\.chat\.completions",
    r"anthropic\.messages",
    r"OPENAI_API_KEY", r"DEEPSEEK_", r"GOOGLE_API_KEY", r"ANTHROPIC_API_KEY",
]

BENCHMARK_QUERY_MARKERS = [
    r"posttrainbench", r"post-train-bench",
    r"PostTrainBench-Trajectories", r"post-train-bench-traces",
    r"coevobench-trajectories",
    r"SWE-bench_Verified", r"princeton-nlp/SWE-bench",
    r"swe-bench.*gold", r"FAIL_TO_PASS", r"PASS_TO_PASS",
    r"tb-lite.*gold", r"tau2.*test", r"appworld.*test", r"bfcl.*answer",
    r"/pull/\d+\.diff", r"/commit/[0-9a-f]{7,}",
]

_GUARD_VERDICTS = (r"\bBLOCK\b", r"\bSUSPECT\b", r"\bDENY\b", r"\bDENIED\b")


def _compile(markers: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    return [(m, re.compile(m, re.IGNORECASE)) for m in markers]


def _redact(text: str) -> str:
    return re.sub(r"(sk-|Bearer )\S{8,}", r"\1[REDACTED]", text)


def scan_text(
    patterns: list[tuple[str, re.Pattern[str]]],
    text: str,
    source: str,
    max_hits: int = 40,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if len(hits) >= max_hits:
            break
        for raw, rx in patterns:
            if rx.search(line):
                snippet = line.strip()
                if len(snippet) > 240:
                    snippet = snippet[:240] + "..."
                hits.append(
                    {
                        "source": source,
                        "line": lineno,
                        "marker": raw,
                        "text": _redact(snippet),
                    }
                )
                break
    return hits


def read_bounded(path: Path, cap: int = MAX_SCAN_BYTES) -> str:
    with path.open("rb") as handle:
        return handle.read(cap).decode(errors="replace")


def _iter_scan_files(root: Path):
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.stat().st_size > MAX_SCAN_BYTES:
            continue
        if path.suffix not in (".log", ".jsonl", ".txt", ".md", ".sh"):
            continue
        yield path


def collect_command_log(audit_dir: Path) -> dict[str, Any]:
    """Guard verdict counts and hits from the recorded command log."""
    verdicts = _compile(_GUARD_VERDICTS)
    counts = {"BLOCK": 0, "SUSPECT": 0, "DENY": 0, "DENIED": 0}
    hits: list[dict[str, Any]] = []
    log = audit_dir / "bash.log"
    if log.is_file():
        text = read_bounded(log)
        for raw, rx in verdicts:
            counts[raw.strip("\\b")] = len(rx.findall(text))
        hits = scan_text(verdicts, text, "bash.log")
    return {"n_lines": sum(1 for _ in log.open(errors="replace")) if log.is_file() else 0,
            "counts": counts, "hits": hits[:20]}


def scan_for_external_calls(root: Path) -> list[dict[str, Any]]:
    patterns = _compile(EXTERNAL_API_MARKERS)
    hits: list[dict[str, Any]] = []
    for path in _iter_scan_files(root):
        hits += scan_text(patterns, read_bounded(path), str(path.relative_to(root)))
    return hits


def scan_for_benchmark_queries(root: Path) -> list[dict[str, Any]]:
    patterns = _compile(BENCHMARK_QUERY_MARKERS)
    hits: list[dict[str, Any]] = []
    for path in _iter_scan_files(root):
        hits += scan_text(patterns, read_bounded(path), str(path.relative_to(root)))
    return hits


def build_evidence(run_root: Path) -> dict[str, Any]:
    """Compile the evidence pack for one run.

    run_root is the registry directory (runs/<id>/) with audit/ and
    workspace/ subdirectories. The human verdict template is left
    empty on purpose: the paper's audit is manual.
    """
    run_root = Path(run_root)
    audit_dir = run_root / "audit"
    workspace = run_root / "workspace"
    return {
        "run_id": run_root.name,
        "command_log": collect_command_log(audit_dir),
        "external_calls": (
            scan_for_external_calls(audit_dir) + scan_for_external_calls(workspace)
        ),
        "benchmark_queries": (
            scan_for_benchmark_queries(audit_dir) + scan_for_benchmark_queries(workspace)
        ),
        "verdict": {
            "cheating_confirmed": None,
            "notes": "",
            "decided_by": "",
        },
    }


def render_markdown(evidence: dict[str, Any]) -> str:
    """Human-readable evidence report; the verdict block stays fillable."""
    lines = [f"# Audit evidence: {evidence['run_id']}", ""]

    log = evidence["command_log"]
    lines.append("## Command log")
    lines.append(
        f"- {log['n_lines']} recorded commands; verdicts "
        f"BLOCK={log['counts'].get('BLOCK', 0)} "
        f"SUSPECT={log['counts'].get('SUSPECT', 0)}"
    )
    for hit in log["hits"]:
        lines.append(f"  - L{hit['line']} [{hit['marker']}]: {hit['text']}")
    lines.append("")

    lines.append("## External-model calls")
    if evidence["external_calls"]:
        for hit in evidence["external_calls"][:20]:
            lines.append(f"- {hit['source']} L{hit['line']} [{hit['marker']}]: {hit['text']}")
    else:
        lines.append("- (none found)")
    lines.append("")

    lines.append("## Benchmark queries")
    if evidence["benchmark_queries"]:
        for hit in evidence["benchmark_queries"][:20]:
            lines.append(f"- {hit['source']} L{hit['line']} [{hit['marker']}]: {hit['text']}")
    else:
        lines.append("- (none found)")
    lines.append("")

    verdict = evidence["verdict"]
    lines.append("## Verdict (manual)")
    lines.append(f"- cheating_confirmed: {verdict['cheating_confirmed']}")
    lines.append(f"- notes: {verdict['notes'] or '(fill in)'}")
    lines.append(f"- decided_by: {verdict['decided_by'] or '(fill in)'}")
    return "\n".join(lines) + "\n"


__all__ = [
    "build_evidence",
    "collect_command_log",
    "render_markdown",
    "scan_for_benchmark_queries",
    "scan_for_external_calls",
]
