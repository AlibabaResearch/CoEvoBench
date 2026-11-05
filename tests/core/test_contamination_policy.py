"""Contamination policy: quotas, receipts, reference freeze, review gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coevobench.core.contamination.policy import (
    MAX_CHECKS,
    MIN_RECORDS,
    POLICY,
    ReceiptLedger,
)
from coevobench.core.contamination.service import ScreeningError
from coevobench.core.protocol.splits import from_mapping


def _splits() -> object:
    ids = [f"repo__item-{i:03d}" for i in range(10)]
    return from_mapping({"visible": ids[:2], "selection": ids[2:6], "test": ids[6:]})


def _reader() -> dict[str, str]:
    ids = [f"repo__item-{i:03d}" for i in range(10)]
    return {tid: f"public problem statement for {tid}" for tid in ids}


def _corpus(tmp_path: Path, name: str, n: int = MIN_RECORDS) -> Path:
    path = tmp_path / name
    lines = [
        json.dumps({"text": f"training record {i} about unrelated topics {i}"})
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _contaminated_corpus(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    lines = [
        json.dumps({"text": f"training record {i} unrelated content {i}"})
        for i in range(MIN_RECORDS - 1)
    ]
    lines.append(json.dumps({"text": "public problem statement for repo__item-002"}))
    path.write_text("\n".join(lines) + "\n")
    return path


def _reference() -> list[dict[str, str]]:
    from coevobench.core.contamination.service import build_reference_items

    return build_reference_items("swe", _splits(), task_reader=_reader())


def test_freeze_reference_is_idempotent_and_detects_drift(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    first = ledger.freeze_reference("swe", _splits(), task_reader=_reader())
    again = ledger.freeze_reference("swe", _splits(), task_reader=_reader())
    assert first["identity"] == again["identity"]
    assert first["n_items"] == 8  # selection + test only

    drifted = dict(_reader())
    drifted["repo__item-002"] = "changed text"
    with pytest.raises(ScreeningError, match="reference changed"):
        ledger.freeze_reference("swe", _splits(), task_reader=drifted)


def test_check_writes_incomplete_then_completes(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    corpus = _corpus(tmp_path, "corpus.jsonl")
    receipt = ledger.check([str(corpus)], _reference())
    assert receipt["status"] == "complete"
    assert receipt["policy"] == POLICY
    assert receipt["n_documents"] == MIN_RECORDS

    # The receipt ledger persisted and is the operator-side evidence.
    rows = [json.loads(ln) for ln in (tmp_path / "state" / "checks.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["check_id"] == receipt["check_id"]


def test_crash_leaves_incomplete_never_clean(tmp_path: Path, monkeypatch) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    corpus = _corpus(tmp_path, "corpus.jsonl")

    def boom(*args, **kwargs):
        raise RuntimeError("scanner crashed")

    monkeypatch.setattr(
        "coevobench.core.contamination.policy.screen_corpus", boom
    )
    with pytest.raises(RuntimeError):
        ledger.check([str(corpus)], _reference())
    rows = [json.loads(ln) for ln in (tmp_path / "state" / "checks.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "incomplete"


def test_identical_content_is_cached_and_free_of_quota(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    corpus = _corpus(tmp_path, "corpus.jsonl")
    copy = tmp_path / "same-bytes.jsonl"
    copy.write_text(corpus.read_text())
    ledger.check([str(corpus)], _reference())
    second = ledger.check([str(copy)], _reference())
    assert second["cached"] is True
    assert ledger.n_quota_checks() == 1


def test_quota_blocks_the_sixth_distinct_corpus(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    for version in range(MAX_CHECKS):
        corpus = _corpus(tmp_path, f"v{version}.jsonl")
        # Make each version's content distinct (same seed text would be
        # one corpus version served from cache).
        text = corpus.read_text().replace("training record", f"training v{version} record")
        corpus.write_text(text)
        receipt = ledger.check([str(corpus)], _reference())
        assert receipt["status"] == "complete"
    with pytest.raises(ScreeningError, match="quota exhausted"):
        ledger.check([str(_corpus(tmp_path, "v-extra.jsonl"))], _reference())


def test_small_corpus_is_refused_before_quota(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    small = _corpus(tmp_path, "small.jsonl", n=MIN_RECORDS - 1)
    with pytest.raises(ScreeningError, match="below the"):
        ledger.check([str(small)], _reference())
    assert ledger.n_quota_checks() == 0


def test_file_count_and_unreadable_inputs_fail_closed(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    many = [str(_corpus(tmp_path, f"f{i}.jsonl")) for i in range(33)]
    with pytest.raises(ScreeningError, match="file limit"):
        ledger.check(many, _reference())

    missing = [str(tmp_path / "nope.jsonl")]
    with pytest.raises(ScreeningError, match="not readable"):
        ledger.check(missing, _reference())


def test_public_view_never_leaks_private_evidence(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    corpus = _contaminated_corpus(tmp_path, "dirty.jsonl")
    receipt = ledger.check([str(corpus)], _reference())
    assert receipt["n_review_required"] >= 1
    view = ReceiptLedger.public_view(receipt)
    text = json.dumps(view)
    assert "repo__item" not in text          # no reference ids
    assert "selection" not in text           # no split membership
    assert "combined_score" not in text      # no match scores
    assert view["findings"][0]["band"] in ("contaminated", "near")


def test_submission_requires_receipt_and_review(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    dirty = _contaminated_corpus(tmp_path, "dirty.jsonl")
    receipt = ledger.check([str(dirty)], _reference())

    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"files": [str(dirty)]}))

    with pytest.raises(ScreeningError, match="operator review"):
        ledger.verify_submission(manifest)

    ledger.review(receipt["check_id"], "approved", "audited: acceptable provenance")
    result = ledger.verify_submission(manifest)
    assert result["reviewed"] is True

    ledger.review(receipt["check_id"], "rejected", "copied benchmark tasks")
    with pytest.raises(ScreeningError, match="rejected"):
        ledger.verify_submission(manifest)

    other = tmp_path / "never-screened.json"
    other.write_text(json.dumps({"files": [str(_corpus(tmp_path, "x.jsonl"))]}))
    with pytest.raises(ScreeningError, match="no completed screening receipt"):
        ledger.verify_submission(other)


def test_review_only_accepts_known_checks_and_decisions(tmp_path: Path) -> None:
    ledger = ReceiptLedger(tmp_path / "state")
    with pytest.raises(ScreeningError, match="unknown check"):
        ledger.review("missing", "approved", "x")
    corpus = _corpus(tmp_path, "ok.jsonl")
    receipt = ledger.check([str(corpus)], _reference())
    with pytest.raises(ScreeningError, match="approved|rejected"):
        ledger.review(receipt["check_id"], "maybe", "x")
