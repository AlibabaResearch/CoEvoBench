"""Candidate-side mailbox driver. No world, private data, or API credentials."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/tmp/ceb-policy")
counter = 0


def receive(number: int):
    path = ROOT / f"in-{number}.json"
    while True:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)


def emit(value: dict):
    global counter
    counter += 1
    value["seq"] = counter
    path = ROOT / f"out-{counter}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)
    return counter


def completion(*, messages, **_ignored):
    number = emit({"type": "completion", "messages": messages})
    result = receive(number)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    sys.path.insert(0, str(ROOT / "harness"))
    from appworldharness import agent_factory

    initial = receive(0)
    agent = agent_factory(
        task=initial["task"],
        model_config=initial["config"],
        completion=completion,
    )
    event = emit({"type": "ready"})
    while True:
        request = receive(event)
        if request["type"] == "close":
            break
        code = agent.next_code(request.get("observation"))
        event = emit({"type": "code", "code": code})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({"type": "error", "error": type(exc).__name__})
