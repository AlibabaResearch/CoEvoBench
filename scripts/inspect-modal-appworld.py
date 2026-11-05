"""Read-only inspection of an existing Modal container; never creates or alters cloud resources."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from coevobench.config.cloud_env import load_env

PROJECT = Path(__file__).resolve().parents[1]
CHECK = (PROJECT / "src/coevobench/runtime/modal_probe.py").read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", help="existing ta-... or sb-... ID")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:ta|sb)-[A-Za-z0-9]+", args.container):
        parser.error("provide an existing Modal container or Sandbox ID")
    load_env(PROJECT / ".local/appworld-cloud.env")
    modal = Path(sys.executable).parent / "modal"
    result = subprocess.run(
        [
            str(modal),
            "shell",
            args.container,
            "--no-pty",
            "--cmd",
            shlex.join(["python3", "-c", CHECK]),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        detail = result.stderr.strip()
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "TEACHER_TOKEN", "RESEARCHER_TOKEN"):
            if os.environ.get(key):
                detail = detail.replace(os.environ[key], "[REDACTED]")
        print(
            json.dumps({"ok": False, "container": args.container, "error": detail[:700]}, indent=2)
        )
        return 1
    try:
        report = json.loads(result.stdout)
    except ValueError:
        print(
            json.dumps({"ok": False, "error": "container did not return a valid inspection report"})
        )
        return 1
    report["container"] = args.container
    destination = PROJECT / ".local/modal-container-inspection.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
