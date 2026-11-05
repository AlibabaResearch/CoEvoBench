"""Timing: research-stream import and span summaries."""

from __future__ import annotations

import json
from pathlib import Path


def test_import_research_stream_pairs_tool_events(tmp_path: Path) -> None:
    from coevobench.core.lifecycle.timing import Recorder, import_research_stream, summarize

    stream = tmp_path / "stream.jsonl"
    stream.write_text(
        "\n".join(
            [
                json.dumps({"type": "tool_use", "id": "t1", "name": "Bash", "timestamp": 100.0}),
                json.dumps({"type": "tool_result", "tool_use_id": "t1", "timestamp": 103.5}),
                json.dumps({"type": "tool_use", "id": "t2", "name": "Read", "timestamp": 104.0}),
                json.dumps({"type": "tool_use", "id": "t3", "name": "Edit", "timestamp": 105.0}),
                json.dumps({"type": "tool_result", "tool_use_id": "t3", "timestamp": 105.5}),
            ]
        )
        + "\n"
    )
    recorder = Recorder(tmp_path / "timing.jsonl")
    written = import_research_stream(stream, recorder, run_id="r1")
    assert written == 3  # t1, t3 closed; t2 aborted

    summary = summarize(tmp_path / "timing.jsonl")
    spans = summary["spans"]
    assert spans["researcher/Bash"]["count"] == 1
    assert spans["researcher/Edit"]["count"] == 1
    assert spans["researcher/Read"]["count"] == 1
    assert spans["researcher/Bash"]["total_s"] == 3.5


def test_import_research_stream_tolerates_garbage(tmp_path: Path) -> None:
    from coevobench.core.lifecycle.timing import Recorder, import_research_stream

    stream = tmp_path / "garbage.jsonl"
    stream.write_text("not json\n\n[]\n")
    recorder = Recorder(tmp_path / "timing.jsonl")
    assert import_research_stream(stream, recorder) == 0
    assert import_research_stream(tmp_path / "missing.jsonl", recorder) == 0
