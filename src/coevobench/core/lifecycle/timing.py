"""Append-only span timing for research jobs and evaluator operations
(B5; paper Figure 6's research-duration data source).

Only labels and numerical metadata are recorded — never prompts, tool
outputs, credentials, or complete shell command arguments. Elapsed
time is monotonic. The recorder is flock-guarded so concurrent
processes (eval workers, the supervisor, the broker daemon) append
safely to one file.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import math
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path


class Recorder:
    """JSONL span recorder; a null path disables everything."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None

    def write(self, event: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @contextlib.contextmanager
    def span(self, category: str, label: str, **fields) -> Iterator[dict]:
        """Record a span (start/end events with monotonic duration)."""
        ident = uuid.uuid4().hex
        start_wall = time.time()
        start_mono = time.monotonic()
        common = {
            "span_id": ident,
            "category": category,
            "label": label,
            "pid": os.getpid(),
        }
        self.write({**common, "event": "start", "time": start_wall, **fields})
        result: dict = {}
        try:
            yield result
        except BaseException as exc:
            self.write(
                {
                    **common,
                    "event": "end",
                    "time": time.time(),
                    "duration_s": round(time.monotonic() - start_mono, 3),
                    "error": type(exc).__name__,
                }
            )
            raise
        self.write(
            {
                **common,
                "event": "end",
                "time": time.time(),
                "duration_s": round(time.monotonic() - start_mono, 3),
                **result,
            }
        )


def summarize(path: Path | str) -> dict:
    """Aggregate spans by category+label: count, total, mean, p95."""
    path = Path(path)
    if not path.is_file():
        return {"spans": {}, "open": 0}
    open_spans: dict[str, dict] = {}
    closed: dict[tuple, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        key = (event.get("category", "?"), event.get("label", "?"))
        if event.get("event") == "start":
            open_spans[event["span_id"]] = event
        elif event.get("event") == "end":
            open_spans.pop(event["span_id"], None)
            duration = event.get("duration_s")
            if isinstance(duration, (int, float)):
                closed[key].append(duration)
    spans = {}
    for (category, label), durations in closed.items():
        durations.sort()
        total = sum(durations)
        spans[f"{category}/{label}"] = {
            "count": len(durations),
            "total_s": round(total, 1),
            "mean_s": round(total / len(durations), 3),
            "p95_s": round(durations[max(0, math.ceil(0.95 * len(durations)) - 1)], 3),
        }
    return {"spans": spans, "open": len(open_spans)}


def write_csv(path: Path | str, summary: dict, destination: Path | str) -> None:
    """Flat CSV of the span summary for spreadsheets."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "label", "count", "total_s", "mean_s", "p95_s"])
        for key, row in summary.get("spans", {}).items():
            category, _, label = key.partition("/")
            writer.writerow(
                [category, label, row["count"], row["total_s"], row["mean_s"], row["p95_s"]]
            )


def import_research_stream(
    stream_path: Path | str,
    recorder: Recorder,
    *,
    run_id: str = "",
) -> int:
    """Convert a researcher agent's event stream into research spans.

    The agent CLI logs tool_use/tool_result events with ids; each pair
    brackets one tool call. This importer walks the stream, pairs the
    events, and appends spans under category `researcher` (label =
    tool name) so research activity joins the same timing ledger as
    platform operations (paper Figure 6's researcher-activity source).
    Unmatched tool_use events are closed at the next stream timestamp
    (an aborted call is still activity). Returns the number of spans
    written.
    """
    import json as _json

    stream_path = Path(stream_path)
    if not stream_path.is_file():
        return 0
    written = 0
    open_calls: dict[str, tuple[str, float]] = {}
    for line in stream_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = _json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type") or event.get("event")
        call_id = (
            event.get("tool_use_id")
            or event.get("id")
            or (event.get("message") or {}).get("id")
            or ""
        )
        ts = float(event.get("timestamp") or event.get("time") or 0)
        name = (
            event.get("name")
            or (event.get("message") or {}).get("name")
            or "tool"
        )
        if kind in ("tool_use", "assistant", "start"):
            if call_id:
                open_calls[call_id] = (str(name), ts)
        elif kind in ("tool_result", "user", "end") and call_id in open_calls:
            started_name, started_ts = open_calls.pop(call_id)
            duration = max(0.0, ts - started_ts) if ts else 0.0
            recorder.write(
                {
                    "span_id": call_id,
                    "category": "researcher",
                    "label": started_name,
                    "pid": 0,
                    "event": "end",
                    "time": started_ts or time.time(),
                    "duration_s": round(duration, 3),
                    "run_id": run_id,
                }
            )
            written += 1
    # Aborted calls: close them at the last observed time.
    for call_id, (name, started_ts) in open_calls.items():
        recorder.write(
            {
                "span_id": call_id,
                "category": "researcher",
                "label": name,
                "pid": 0,
                "event": "end",
                "time": started_ts or time.time(),
                "duration_s": 0.0,
                "run_id": run_id,
                "aborted": True,
            }
        )
        written += 1
    return written


__all__ = ["Recorder", "import_research_stream", "summarize", "write_csv"]
