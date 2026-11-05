#!/usr/bin/env python3
"""Example: local model -> custom remote task -> reward and JSONL trajectory.

Supply an environment (a prebuilt image or a build context directory with a
Dockerfile), task text and a verifier command. No fixed task dataset.
"""

import argparse
import json
import os
import time
from pathlib import Path
import httpx
from coevobench.research.client import Client


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="prebuilt image reference")
    src.add_argument(
        "--from-dir", type=Path, help="build context (Dockerfile at its root)"
    )
    p.add_argument("--task-file", required=True, type=Path)
    p.add_argument(
        "--verify", required=True, help="Command printing your reward/feedback"
    )
    p.add_argument("--model", required=True, help="Local served model name")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", type=Path, default=Path("work/rollout.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    client = Client()
    session = client.call(
        "sandbox.create",
        **({"image": a.image} if a.image else {"from_dir": str(a.from_dir)}),
        ttl_s=1800,
    )
    messages = [
        {
            "role": "system",
            "content": "Solve the task using shell commands. Reply with only a JSON object containing command (string) and done (boolean).",
        },
        {"role": "user", "content": a.task_file.read_text()},
    ]

    def log(event):
        with a.out.open("a") as f:
            f.write(
                json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n"
            )

    log({"kind": "start", "session": session, "messages": messages})
    try:
        with httpx.Client(timeout=300) as http:
            for _ in range(a.steps):
                r = http.post(
                    os.environ.get("VLLM_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
                    + "/chat/completions",
                    headers={"Authorization": "Bearer dummy"},
                    json={
                        "model": a.model,
                        "messages": messages,
                        "temperature": a.temperature,
                        "max_tokens": 2048,
                    },
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                log({"kind": "model", "content": text})
                messages.append({"role": "assistant", "content": text})
                try:
                    action = json.loads(text)
                    if not isinstance(action, dict) or not isinstance(
                        action.get("command", ""), str
                    ):
                        raise ValueError("Expected a JSON object with a string command")
                except (ValueError, TypeError) as e:
                    messages.append({"role": "user", "content": str(e)})
                    continue
                if action.get("command"):
                    result = client.call(
                        "sandbox.exec",
                        sandbox_id=session["sandbox_id"],
                        command=action["command"],
                        timeout_s=120,
                    )
                    log({"kind": "environment", "command": action["command"], **result})
                    messages.append({"role": "user", "content": json.dumps(result)})
                if action.get("done"):
                    break
        result = client.call(
            "sandbox.exec", sandbox_id=session["sandbox_id"], command=a.verify, timeout_s=180
        )
        log({"kind": "verification", **result})
        print(json.dumps(result))
    finally:
        try:
            client.call("sandbox.close", sandbox_id=session["sandbox_id"])
        finally:
            client.call(
                "sandbox.export",
                sandbox_id=session["sandbox_id"],
                local=str(a.out.with_suffix(".environment.jsonl").resolve()),
            )


if __name__ == "__main__":
    main()
