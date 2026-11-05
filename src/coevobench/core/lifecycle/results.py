"""Result store (B9): the append-only record of every graded run.

JSONL ledgers and a scoreboard index record results. All writes are
flock-guarded appends; reads resolve through the index (rebuilt on
demand). One store lives under ~/.ceb/results/.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any


class ResultStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.ledger = self.root / "ledger.jsonl"
        self.index_path = self.root / "index.json"

    def record(self, entry: dict[str, Any]) -> dict:
        """Append one graded entry (run id, bench, role, score, wall clock,
        serve summary, exit distribution — the ledger discipline)."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"time": time.time(), **entry}
        with self.ledger.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def entries(self, run_id: str | None = None, bench: str | None = None) -> list[dict]:
        """Read the ledger, optionally filtered."""
        if not self.ledger.is_file():
            return []
        out = []
        for line in self.ledger.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if run_id and entry.get("run_id") != run_id:
                continue
            if bench and entry.get("bench") != bench:
                continue
            out.append(entry)
        return out

    def scoreboard(self, bench: str | None = None) -> list[dict]:
        """Best score per (run, bench, role), newest first."""
        best: dict[tuple, dict] = {}
        for entry in self.entries(bench=bench):
            key = (entry.get("run_id"), entry.get("bench"), entry.get("role", "final"))
            score = entry.get("score")
            if not isinstance(score, (int, float)):
                continue
            current = best.get(key)
            if current is None or score > current["score"]:
                best[key] = entry
        rows = sorted(best.values(), key=lambda e: -e.get("time", 0))
        return rows

    def rebuild_index(self) -> int:
        """Persist the scoreboard snapshot (index.json)."""
        rows = self.scoreboard()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(rows, indent=2))
        return len(rows)


__all__ = ["ResultStore"]
