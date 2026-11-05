"""Contamination screening: signal, never verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coevobench.core.contamination import (
    ScreeningError,
    build_reference_items,
    screen_corpus,
    verify_no_protected_overlap,
)
from coevobench.core.protocol.splits import from_mapping

SPLITS = from_mapping(
    {
        "visible": ["v1"],
        "selection": ["t1", "t2"],
        "test": ["t3"],
    }
)
READER = {
    "t1": "The parser crashes on empty input when the buffer is exhausted "
    "before the header terminator is reached, leaving the reader thread "
    "in an unrecoverable state that requires a full process restart.",
    "t2": "Database connections leak under concurrent load because the "
    "pool never releases failed handshakes, and the watchdog reaps idle "
    "sockets too late to prevent exhaustion during burst traffic peaks.",
    "t3": "Migration scripts run twice because the ledger table lacks a "
    "unique constraint on batch identifiers, so retries duplicate rows "
    "and the reconciliation job then silently drops the older copies.",
}


def _corpus(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_reference_build_requires_full_coverage() -> None:
    with pytest.raises(ScreeningError, match="lack reference text"):
        build_reference_items("swe", SPLITS, task_reader={"t1": "only one"})
    items = build_reference_items("swe", SPLITS, task_reader=READER)
    assert {i["instance_id"] for i in items} == {"t1", "t2", "t3"}


def test_reference_build_rejects_unsupported_bench() -> None:
    with pytest.raises(ScreeningError, match="not configured"):
        build_reference_items("toy", SPLITS, task_reader=READER)


def test_clean_corpus_reports_no_matches(tmp_path: Path) -> None:
    items = build_reference_items("swe", SPLITS, task_reader=READER)
    corpus = _corpus(
        tmp_path,
        "clean.jsonl",
        [{"messages": [{"role": "user", "content": "Write a haiku about filesystems."}]}],
    )
    report = screen_corpus(items, [corpus])
    assert report["n_review_required"] == 0
    assert report["verdict"].startswith("NO_MATCHES")


def test_copied_task_text_is_flagged(tmp_path: Path) -> None:
    items = build_reference_items("swe", SPLITS, task_reader=READER)
    corpus = _corpus(
        tmp_path,
        "dirty.jsonl",
        [{"messages": [{"role": "user", "content": READER["t1"] + " Please fix."}]}],
    )
    report = screen_corpus(items, [corpus])
    assert report["n_review_required"] == 1
    assert report["results"][0]["band"] == "contaminated"
    assert report["results"][0]["best_score"] >= 0.8


def test_unreadable_input_is_refused(tmp_path: Path) -> None:
    items = build_reference_items("swe", SPLITS, task_reader=READER)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ScreeningError, match="not valid JSON"):
        screen_corpus(items, [bad])
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"messages": []}\n')
    with pytest.raises(ScreeningError, match="no text content"):
        screen_corpus(items, [empty])


def test_missing_corpus_refused(tmp_path: Path) -> None:
    items = build_reference_items("swe", SPLITS, task_reader=READER)
    with pytest.raises(ScreeningError, match="not readable"):
        screen_corpus(items, [tmp_path / "nothing.jsonl"])


def test_one_shot_helper(tmp_path: Path) -> None:
    corpus = _corpus(
        tmp_path, "c.jsonl", [{"text": "Unrelated cooking recipe content."}]
    )
    report = verify_no_protected_overlap("swe", SPLITS, [corpus], task_reader=READER)
    assert report["n_documents"] == 1
