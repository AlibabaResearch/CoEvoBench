"""Operator-only AppWorld worker, launched with the pinned AppWorld interpreter."""

from __future__ import annotations

import contextlib
import json
import sys


def main() -> None:
    output = sys.stdout

    def reply(value):
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()

    with contextlib.redirect_stdout(sys.stderr):
        from appworld import AppWorld

        with AppWorld(task_id=sys.argv[1], experiment_name="ceb") as world:
            reply(
                {
                    "instruction": world.task.instruction,
                    "supervisor": dict(world.task.supervisor),
                }
            )
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request["op"] == "close":
                        reply({"ok": True})
                        break
                    if request["op"] == "execute":
                        observation = world.execute(request["code"])
                        reply({"observation": observation, "completed": world.task_completed()})
                    elif request["op"] == "evaluate":
                        tracker = world.evaluate()
                        reply({"tgc": bool(tracker.success), "tests": tracker.to_dict()})
                    else:
                        raise ValueError("unsupported world operation")
                except Exception as exc:
                    reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
