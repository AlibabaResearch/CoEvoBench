"""Contamination screening: training corpus vs. the protected splits.

Derived from PostTrainBench decon SIMPLE (engine.py) with the CoevoBench
addition of problem_statement/instruction as question fields. The
service builds a reference index from the frozen split definitions and
the fetched task content, screens researcher-supplied corpora, and
reports overlap as a review signal — never an automatic verdict:

- any text/identity hit means REVIEW, not violation
- NO_MATCHES is not proof of absence (paraphrase/translation evade it);
  provenance review still applies
- Visible tasks are legal development data; matching visible is not
  contamination. This service screens against Selection and Test only.

Screened inputs must be complete and readable; unreadable or empty
records are refused rather than silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coevobench.core.protocol.splits import Splits

BENCHES = ("swe", "tb")


class ScreeningError(ValueError):
    pass


def build_reference_items(
    bench: str,
    splits: Splits,
    *,
    task_reader: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Identity + problem text for every Selection/Test task; no gold.

    task_reader maps task_id -> problem/instruction text. The caller
    owns acquisition (fetched dataset or task definitions); this module
    never reads private reference paths.
    """
    if bench not in BENCHES:
        raise ScreeningError(f"screening is not configured for {bench}")
    protected = set(splits.ids("selection")) | set(splits.ids("test"))
    if not protected:
        raise ScreeningError("empty protected splits")
    reader = task_reader or {}
    missing = [tid for tid in sorted(protected) if not reader.get(tid, "").strip()]
    if missing:
        raise ScreeningError(
            f"{len(missing)} protected tasks lack reference text (e.g. {missing[:3]}); "
            "run data fetch and pass the task reader"
        )
    return [
        {"instance_id": tid, "problem_statement": reader[tid]} for tid in sorted(protected)
    ]


def _document_text(record: dict) -> str:
    from coevobench.core.contamination import engine

    try:
        return engine.document_text(record)
    except ValueError:
        return ""


def screen_corpus(
    reference_items: list[dict[str, str]],
    corpus_files: list[Path],
    *,
    tolerance: float = 0.2,
) -> dict[str, Any]:
    """Screen every record of every corpus file against the reference.

    corpus_files are JSONL (one record per line; every string field is
    scanned). Returns the report dict; the caller persists it. Hits are
    review signals (see the module docstring).
    """
    from coevobench.core.contamination import engine

    index = engine.ReferenceIndex(reference_items, include_short=True)
    if not index.items:
        raise ScreeningError("reference index is empty")

    documents: list[dict[str, Any]] = []
    for path in corpus_files:
        path = Path(path)
        if not path.is_file():
            raise ScreeningError(f"corpus file not readable: {path}")
        for number, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ScreeningError(f"{path}:{number} is not valid JSON: {exc}") from None
            text = _document_text(record)
            if not text.strip():
                raise ScreeningError(f"{path}:{number} has no text content")
            documents.append({"text": text, "source": f"{path.name}:{number}"})

    if not documents:
        raise ScreeningError("corpus is empty; refusing to screen nothing")

    near_threshold = 0.5
    results = []
    review_required = 0
    for document in documents:
        matches, variant = engine.score_document(
            index, document, near_threshold, patch_aware=True
        )
        contaminated = any(m["combined_score"] >= 0.8 for m in matches)
        near = not contaminated and bool(matches)
        if contaminated or near:
            review_required += 1
        results.append(
            {
                "source": document["source"],
                "band": "contaminated" if contaminated else "near" if near else "none",
                "n_matches": len(matches),
                "best_score": matches[0]["combined_score"] if matches else 0.0,
                "variant": variant,
            }
        )

    return {
        "n_documents": len(documents),
        "n_review_required": review_required,
        "near_threshold": near_threshold,
        "tolerance": tolerance,
        "verdict": (
            "REVIEW_REQUIRED (overlap signal; inspect provenance)"
            if review_required
            else "NO_MATCHES (screening only)"
        ),
        "results": results,
    }


def verify_no_protected_overlap(
    bench: str,
    splits: Splits,
    corpus_files: list[Path],
    *,
    task_reader: dict[str, str] | None = None,
    tolerance: float = 0.2,
) -> dict[str, Any]:
    """One-shot: build the reference and screen, raising on inputs that
    would silently produce a clean verdict (unreadable/empty corpora)."""
    items = build_reference_items(bench, splits, task_reader=task_reader)
    return screen_corpus(items, corpus_files, tolerance=tolerance)


__all__ = [
    "BENCHES",
    "ScreeningError",
    "build_reference_items",
    "screen_corpus",
    "verify_no_protected_overlap",
]
