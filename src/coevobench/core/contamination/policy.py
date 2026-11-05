"""Quota, receipt, and review discipline for corpus screening (policy
`private-corpus-v1`).

The paper's audit protocol (Section 3.4) treats the corpus screen as one
leg of a three-way human audit; this module keeps that leg honest:

- quotas bound the screening surface per run: at most MAX_CHECKS
  distinct corpus versions, each batch at least MIN_RECORDS distinct
  text records, at most MAX_FILES files and MAX_BYTES total
- a check is recorded `incomplete` *before* scanning, so a crash can
  never turn into a clean verdict; the record is completed atomically
- identical content under different paths is served from cache and does
  not consume quota
- the private receipt (full match evidence) stays operator-side; the
  researcher sees the public view only: file, line, overlap band —
  never reference ids, scores, or split membership
- `verify_submission` matches the workspace's train/corpus.json
  manifest against stored receipts and the operator review file; an
  unreviewed overlap blocks the submission path
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from coevobench.core.contamination.service import (
    ScreeningError,
    build_reference_items,
    screen_corpus,
)

POLICY = "private-corpus-v1"
MAX_CHECKS = 5
MIN_RECORDS = 100
MAX_FILES = 32
MAX_BYTES = 1 << 30


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReceiptLedger:
    """Persistent, append-only screening state for one run.

    All state lives under `state_dir` (operator-side, never mounted into
    the run container): reference.json (frozen reference), checks.jsonl
    (receipt ledger), review-<id>.json (operator decisions).
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ---- reference freeze ----

    def freeze_reference(
        self,
        bench: str,
        splits,
        *,
        task_reader: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Freeze the screening reference on first use.

        A later call with a different reference identity (split
        membership or task text changed) refuses to re-freeze: the
        operator must review what moved before screening continues.
        """
        items = build_reference_items(bench, splits, task_reader=task_reader)
        identity = hashlib.sha256(
            json.dumps(items, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = self.state_dir / "reference.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if record["identity"] != identity:
                raise ScreeningError(
                    "protected-splits reference changed after freeze; "
                    "operator review required before further screening"
                )
            return record
        record = {
            "policy": POLICY,
            "bench": bench,
            "identity": identity,
            "n_items": len(items),
            "frozen_at": time.time(),
        }
        path.write_text(json.dumps(record, indent=2))
        return record

    # ---- receipts ----

    def _receipts(self) -> list[dict[str, Any]]:
        path = self.state_dir / "checks.jsonl"
        if not path.is_file():
            return []
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    def _append(self, record: dict[str, Any]) -> None:
        with (self.state_dir / "checks.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _replace(self, check_id: str, record: dict[str, Any]) -> None:
        path = self.state_dir / "checks.jsonl"
        rows = self._receipts()
        rows = [r if r.get("check_id") != check_id else record for r in rows]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(path)

    def n_quota_checks(self) -> int:
        """Completed checks with distinct content (cache hits excluded)."""
        seen: set[str] = set()
        count = 0
        for row in self._receipts():
            if row.get("status") != "complete" or not row.get("consumes_quota", True):
                continue
            key = row.get("content_sha")
            if key in seen:
                continue
            seen.add(key)
            count += 1
        return count

    def check(
        self,
        inputs: list[str],
        reference_items: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Screen one corpus batch under the policy; returns the receipt.

        The receipt is the private record; hand `public_view` of it to
        the researcher.
        """
        files = [Path(p) for p in inputs]
        if not files:
            raise ScreeningError("no input files supplied")
        if len(files) > MAX_FILES:
            raise ScreeningError(f"{len(files)} files exceeds the {MAX_FILES}-file limit")
        if len({f.name for f in files}) != len(files):
            raise ScreeningError("duplicate file names in the input list")

        sizes = {}
        total = 0
        for f in files:
            if not f.is_file():
                raise ScreeningError(f"corpus file not readable: {f}")
            size = f.stat().st_size
            sizes[str(f)] = size
            total += size
        if total > MAX_BYTES:
            raise ScreeningError(f"{total} bytes exceeds the 1 GiB corpus limit")

        content = {str(f): _sha256_file(f) for f in files}
        # Cache identity is the multiset of file hashes — same bytes under
        # different paths is the same corpus version and reuses the receipt.
        content_sha = hashlib.sha256(
            json.dumps(sorted(content.values())).encode()
        ).hexdigest()

        cached = next(
            (
                row
                for row in self._receipts()
                if row.get("status") == "complete" and row.get("content_sha") == content_sha
            ),
            None,
        )
        if cached is not None:
            return {**cached, "cached": True}

        if self.n_quota_checks() >= MAX_CHECKS:
            raise ScreeningError(
                f"corpus-version quota exhausted ({MAX_CHECKS} distinct versions)"
            )

        # Minimum distinct text records: validate before consuming quota.
        distinct = self._distinct_records(files)
        if distinct < MIN_RECORDS:
            raise ScreeningError(
                f"{distinct} distinct text records is below the {MIN_RECORDS}-record minimum"
            )

        check_id = hashlib.sha256(f"{content_sha}:{time.time()}".encode()).hexdigest()[:16]
        receipt = {
            "check_id": check_id,
            "policy": POLICY,
            "status": "incomplete",
            "content_sha": content_sha,
            "files": {str(f): content[str(f)] for f in files},
            "n_bytes": total,
            "consumes_quota": True,
            "started_at": time.time(),
        }
        self._append(receipt)
        try:
            report = screen_corpus(reference_items, files)
        except BaseException:
            # The incomplete record stays: a crash is never a clean pass.
            raise

        receipt.update(
            {
                "status": "complete",
                "n_documents": report["n_documents"],
                "n_review_required": report["n_review_required"],
                "verdict": report["verdict"],
                "findings": [
                    {
                        "file": row["source"].rsplit(":", 1)[0],
                        "line": row["source"].rsplit(":", 1)[1],
                        "band": row["band"],
                    }
                    for row in report["results"]
                    if row["band"] != "none"
                ],
                "finished_at": time.time(),
            }
        )
        self._replace(check_id, receipt)
        return receipt

    def _distinct_records(self, files: list[Path]) -> int:
        from coevobench.core.contamination import engine

        seen: set[str] = set()
        for path in files:
            for number, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ScreeningError(
                        f"{path}:{number} is not valid JSON: {exc}"
                    ) from None
                text = engine.document_text(record)
                if not text.strip():
                    raise ScreeningError(f"{path}:{number} has no text content")
                seen.add(hashlib.sha256(text.encode()).hexdigest())
        return len(seen)

    # ---- public view / review / submission verification ----

    @staticmethod
    def public_view(receipt: dict[str, Any]) -> dict[str, Any]:
        """The researcher-facing allowlist: never reference ids, scores,
        or split membership — file, line, and overlap band only."""
        return {
            "check_id": receipt["check_id"],
            "policy": receipt["policy"],
            "status": receipt["status"],
            "cached": bool(receipt.get("cached")),
            "n_documents": receipt.get("n_documents", 0),
            "n_review_required": receipt.get("n_review_required", 0),
            "verdict": receipt.get("verdict", "incomplete"),
            "findings": receipt.get("findings", []),
        }

    def review(self, check_id: str, decision: str, reason: str) -> dict[str, Any]:
        """Record the operator's human-audit decision for one check."""
        if decision not in ("approved", "rejected"):
            raise ScreeningError("decision must be approved|rejected")
        receipt = next(
            (r for r in self._receipts() if r.get("check_id") == check_id), None
        )
        if receipt is None:
            raise ScreeningError(f"unknown check {check_id}")
        record = {
            "check_id": check_id,
            "receipt_sha256": hashlib.sha256(
                json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "decision": decision,
            "reason": reason,
            "decided_at": time.time(),
        }
        (self.state_dir / f"review-{check_id}.json").write_text(
            json.dumps(record, indent=2)
        )
        return record

    def verify_submission(self, manifest_path: Path) -> dict[str, Any]:
        """Match train/corpus.json against stored receipts and reviews.

        The manifest lists the corpus files the researcher declares as
        the complete training corpus; verification requires (a) a
        completed receipt covering exactly those files, and (b) an
        operator review when that receipt's verdict is not NO_MATCHES.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise ScreeningError("train/corpus.json manifest is missing")
        manifest = json.loads(manifest_path.read_text())
        declared = manifest.get("files")
        if not isinstance(declared, list) or not declared:
            raise ScreeningError("manifest declares no files")

        receipts = [
            r for r in self._receipts() if r.get("status") == "complete"
        ]
        match = next(
            (r for r in receipts if sorted(r.get("files", {})) == sorted(declared)), None
        )
        if match is None:
            raise ScreeningError(
                "no completed screening receipt covers the declared corpus; "
                "screen the complete corpus before submitting"
            )
        result = {
            "check_id": match["check_id"],
            "manifest_files": len(declared),
            "verdict": match["verdict"],
            "reviewed": False,
        }
        if match.get("n_review_required", 0):
            review_path = self.state_dir / f"review-{match['check_id']}.json"
            if not review_path.is_file():
                raise ScreeningError(
                    "overlap findings require an operator review before submission"
                )
            review = json.loads(review_path.read_text())
            if review["decision"] != "approved":
                raise ScreeningError(
                    f"operator review is {review['decision']}; submission blocked"
                )
            result["reviewed"] = True
        return result


__all__ = [
    "MAX_BYTES",
    "MAX_CHECKS",
    "MAX_FILES",
    "MIN_RECORDS",
    "POLICY",
    "ReceiptLedger",
]
