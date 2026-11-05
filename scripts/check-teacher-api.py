"""Make one small live teacher request through the HTTP bridge and operator broker."""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from coevobench.config.cloud_env import read_env
from coevobench.research.server import Broker, BrokerServer
from coevobench.runtime.proxy.teacher import TeacherForwarder
from coevobench.runtime.proxy.teacher_bridge import make_server

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT / ".local/appworld-cloud.env")
    args = parser.parse_args()
    forwarder = TeacherForwarder(read_env(args.env_file))
    report = {
        "model": forwarder.model,
        "transport": "local HTTP bridge -> Unix broker -> teacher",
        "cost_accounting": "disabled",
        "modal_verified": False,
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ceb-teacher-") as td:
        socket = str(Path(td) / "broker.sock")
        with BrokerServer(socket, Broker(teacher=forwarder)) as broker:
            broker.serve_forever_bg()
            with make_server(socket, forwarder.model) as bridge:
                threading.Thread(target=bridge.serve_forever, daemon=True).start()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{bridge.server_port}/v1/chat/completions",
                    data=json.dumps(
                        {
                            "model": forwarder.model,
                            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                            "max_tokens": 8,
                            "temperature": 0,
                        }
                    ).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer local-relay",
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=150) as response:
                        result = json.load(response)
                    text = result["choices"][0]["message"].get("content", "")
                    report.update(
                        ok=bool(text.strip()), reply=text[:80], usage=result.get("usage", {})
                    )
                except urllib.error.HTTPError as exc:
                    report.update(ok=False, http_status=exc.code)
                except Exception as exc:
                    report.update(ok=False, error=type(exc).__name__)
                finally:
                    bridge.shutdown()
                    broker.shutdown()
    report["seconds"] = round(time.monotonic() - started, 2)
    output = PROJECT / ".local/teacher-forwarding-smoke.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
