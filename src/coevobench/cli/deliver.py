"""ceb freeze / ceb verify / ceb report: candidate freezing, tamper check,
and result aggregation. Thin CLI wrappers over core functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from coevobench.core.grading.freeze import freeze, verify_unchanged
from coevobench.core.grading.provenance import verify as verify_provenance
from coevobench.core.lifecycle.report import aggregate_dir


def freeze_cmd(candidate_dir: str, label: str) -> int:
    report = freeze(Path(candidate_dir), label)
    if not report.ok:
        for error in report.gate_errors:
            print(f"[freeze] FAIL {error}", file=sys.stderr)
        return 1
    audit = _verify_training_audit(Path(candidate_dir))
    if audit is not None:
        status, detail = audit
        print(f"[freeze] training-audit: {status} ({detail})", file=sys.stderr)
        if status == "blocked":
            return 1
    print(
        f"[freeze] {label}: {report.n_files} files, manifest {report.manifest_sha256[:12]}, "
        f"weights {report.weights_fingerprint[:12]}"
    )
    return 0


def _verify_training_audit(candidate_dir: Path) -> tuple[str, str] | None:
    """Verify the corpus receipt when the candidate declares training.

    Returns None when the submission declares no corpus (weights-only or
    harness-only submissions need no training audit), ("verified", ...)
    when the receipt matches, or ("blocked", reason) when the freeze
    must stop: no receipt, an unreviewed overlap, or a rejected review.
    """
    from coevobench.core.contamination.policy import ReceiptLedger

    manifest = candidate_dir / "train" / "corpus.json"
    if not manifest.is_file():
        return None
    state_dir = candidate_dir.parent / "research" / "contamination"
    if not state_dir.is_dir():
        return ("blocked", "train/corpus.json present but no screening state")
    ledger = ReceiptLedger(state_dir)
    try:
        result = ledger.verify_submission(manifest)
    except ValueError as exc:
        return ("blocked", str(exc))
    reviewed = " reviewed" if result["reviewed"] else ""
    return ("verified", f"check {result['check_id']}{reviewed}")


def verify_cmd(candidate_dir: str) -> int:
    if not verify_unchanged(Path(candidate_dir)):
        print("[verify] FAIL: frozen tree changed after freezing", file=sys.stderr)
        return 1
    print("[verify] OK: frozen tree unchanged")
    return 0


def report_cmd(results_dir: str) -> int:
    try:
        payload = aggregate_dir(Path(results_dir))
    except ValueError as exc:
        print(f"[report] {exc}", file=sys.stderr)
        return 1
    _record_to_store(payload)
    for row in payload["benches"]:
        print(
            f"[report] {row['bench']:10s} base={row['base']:.4f} "
            f"instruct={row['instruct']:.4f} reported={row['reported']:.4f} "
            f"J={row['improvement']:+.4f} (n={row['n_runs']})"
        )
    print(f"[report] Avg. = {payload['avg']:.4f}")
    if payload.get("exploratory"):
        print(
            "[report] EXPLORATORY: a domain rests on a single run; the paper's "
            "protocol needs three independent runs per researcher-benchmark pair — "
            "these numbers cannot back conclusions",
            file=sys.stderr,
        )
    return 0




def audit_cmd(run_root: str, out: str | None = None) -> int:
    """Compile the audit evidence pack for a run directory."""
    from coevobench.core.lifecycle.audit import build_evidence, render_markdown

    evidence = build_evidence(Path(run_root))
    report = render_markdown(evidence)
    target = Path(out) if out else Path(run_root) / "audit" / "evidence.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report)
    log = evidence["command_log"]
    print(
        f"[audit] {evidence['run_id']}: {log['n_lines']} commands, "
        f"BLOCK={log['counts'].get('BLOCK', 0)} "
        f"SUSPECT={log['counts'].get('SUSPECT', 0)}, "
        f"ext-calls={len(evidence['external_calls'])}, "
        f"bench-queries={len(evidence['benchmark_queries'])}"
    )
    print(f"[audit] evidence: {target} (verdict left for the human)")
    return 0


def provenance_cmd(submission: str, base: str, roots: str) -> int:
    """Run the weights-lineage gates on a submission."""
    allowed = [Path(r) for r in roots.split(",") if r.strip()]
    report = verify_provenance(Path(submission), Path(base), allowed)
    for name, passed, detail in report.gates:
        mark = "OK  " if passed else "FAIL"
        print(f"[provenance] {mark} {name}: {detail}")
    if not report.ok:
        return 1
    return 0


def _record_to_store(payload: dict) -> None:
    """Every report pass lands in the result store ledger."""
    from coevobench.core.lifecycle.results import ResultStore

    store = ResultStore(Path.home() / ".ceb" / "results")
    for row in payload.get("benches", []):
        store.record(
            {
                "run_id": f"final-{row['bench']}",
                "bench": row["bench"],
                "role": "final",
                "score": row["reported"],
                "n_runs": row.get("n_runs", 1),
            }
        )


__all__ = ["audit_cmd", "freeze_cmd", "provenance_cmd", "report_cmd", "verify_cmd"]


def final_eval_cmd(args) -> int:
    """ceb final-eval prepare|evaluate: the operator's Test-split path."""
    from coevobench.core.lifecycle.final_eval import evaluate, prepare
    from coevobench.core.protocol.protocol_assets import frozen_splits
    from coevobench.core.protocol.splits import from_mapping

    if args.final_eval_command == "prepare":
        bundle = prepare(
            Path(args.submission_dir),
            args.bench,
            Path(args.destination),
            label=args.label,
            base_weights=Path(args.base_weights) if args.base_weights else None,
        )
        print(f"[final-eval] bundle: {bundle}")
        return 0
    if args.final_eval_command == "evaluate":
        _register_bench_bindings(args.bench)
        frozen = frozen_splits(args.bench)
        if frozen is None:
            print(f"[final-eval] unknown bench {args.bench!r}", file=sys.stderr)
            return 2
        splits = from_mapping(
            {
                "visible": list(frozen.visible),
                "selection": list(frozen.selection),
                "test": list(frozen.test),
            }
        )
        weights = Path(args.revert_weights) if args.revert_weights else None
        report = evaluate(
            Path(args.bundle),
            bench=args.bench,
            splits=splits,
            k=int(args.k),
            weights_override=weights,
        )
        suffix = " (reverted, Eq.5)" if report.reverted else ""
        print(
            f"[final-eval] {args.bench}: score={report.score:.4f} "
            f"n={report.n_tasks} k={report.k_trials}{suffix}"
        )
        return 0
    return 2


def comparison_cmd(args) -> int:
    """ceb comparison prepare|evaluate: the M0H0/M0H1/M1H1 controls."""
    from coevobench.core.lifecycle.comparison import evaluate_comparison, prepare_comparison
    from coevobench.core.protocol.protocol_assets import frozen_splits
    from coevobench.core.protocol.splits import from_mapping

    if args.comparison_command == "prepare":
        bundles = prepare_comparison(
            Path(args.submission_dir),
            args.bench,
            Path(args.destination),
            base_weights=Path(args.base_weights),
            seed_harness_src=Path(args.seed_harness),
        )
        manifest = Path(args.destination) / "bundles.json"
        manifest.write_text(
            json.dumps({role: str(p) for role, p in bundles.items()}, indent=2)
        )
        for role, path in bundles.items():
            print(f"[comparison] {role}: {path}")
        print(f"[comparison] manifest: {manifest}")
        return 0
    if args.comparison_command == "evaluate":
        _register_bench_bindings(args.bench)
        frozen = frozen_splits(args.bench)
        if frozen is None:
            print(f"[comparison] unknown bench {args.bench!r}", file=sys.stderr)
            return 2
        splits = from_mapping(
            {
                "visible": list(frozen.visible),
                "selection": list(frozen.selection),
                "test": list(frozen.test),
            }
        )
        bundles = {
            role: Path(p)
            for role, p in json.loads(Path(args.bundles).read_text()).items()
        }
        result = evaluate_comparison(bundles, args.bench, splits, k=int(args.k))
        for role, row in result.scores.items():
            print(f"[comparison] {role}: {row}")
        return 0
    return 2


def _register_bench_bindings(bench: str) -> None:
    """Import the bench layer's replay binding so evaluate() can resolve it."""
    from coevobench.core.lifecycle.final_eval import register_replay_binding

    if bench == "swe":
        from coevobench.bench.swe.replay import replay_judge

        register_replay_binding("swe", replay_judge)
