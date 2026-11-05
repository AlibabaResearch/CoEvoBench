"""Researcher-side broker client: Python + CLI, no credentials inside.

This is the in-envelope surface the workspace entry scripts (./sandbox,
./contamination, ./track) exec into, and the module the research_rollout
examples import. It speaks the same broker wire as
BrokerSandboxRuntime — every operation is one JSON round trip over the
mounted unix socket; nothing here can judge, hold split data, or read
secrets (design Sections 2.1.1 and 7).

CLI exit codes preserve command outcomes: sandbox exec propagates
the command's returncode, contamination returns 0 on no_matches and 3
on review_required, everything else 0/1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SOCKET_ENV = "CEB_BROKER_SOCKET"
DEFAULT_SOCKET = "/broker.sock"


class Client:
    """Thin call interface over the run's broker socket."""

    def __init__(self, socket_path: str | None = None) -> None:
        self.socket = str(
            socket_path
            or os.environ.get(SOCKET_ENV)
            or DEFAULT_SOCKET
        )

    def call(self, op: str, **params) -> dict:
        from coevobench.runtime.broker_wire import BrokerError, call

        try:
            result = call(self.socket, {"op": op, **params}, timeout=3600.0)
            if "error" in result:
                raise RuntimeError(str(result["error"]))
            return result
        except BrokerError as exc:
            raise RuntimeError(
                f"broker unavailable ({exc}); launch this run with its broker"
            ) from exc


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ceb client",
        description="Development sandboxes and corpus screening",
    )
    roots = ap.add_subparsers(dest="kind", required=True)
    roots.add_parser("ping")

    screening = roots.add_parser("contamination").add_subparsers(
        dest="action", required=True
    )
    p = screening.add_parser("check")
    p.add_argument(
        "--input", dest="inputs", nargs="+", required=True,
        help="every final training JSONL file, complete batch",
    )
    p.add_argument(
        "--manifest", type=Path, default=None,
        help="write the corpus declaration (train/corpus.json; create-only)",
    )

    sandbox = roots.add_parser("sandbox").add_subparsers(dest="action", required=True)
    p = sandbox.add_parser("create")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="prebuilt image reference")
    src.add_argument(
        "--from-dir",
        help="build context directory (Dockerfile at its root); the run's "
        "sandbox provider builds it and runs the sandbox on the result",
    )
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--memory", type=int, default=4, help="GiB")
    p.add_argument("--ttl", type=int, default=1800)
    p.add_argument("--network", action="store_true", help="internet access")
    p.add_argument("--build-timeout", type=int, default=1800)
    p.add_argument("--ready-timeout", type=int, default=120)
    p = sandbox.add_parser("exec")
    p.add_argument("id")
    p.add_argument("command")
    p.add_argument("--cwd", default=None)
    p.add_argument("--timeout", type=int, default=180)
    for name in ("status", "close"):
        sandbox.add_parser(name).add_argument("id")
    for name in ("upload", "download"):
        p = sandbox.add_parser(name)
        p.add_argument("id")
        p.add_argument("local")
        p.add_argument("remote")
    p = sandbox.add_parser("export")
    p.add_argument("id")
    p.add_argument("local")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = vars(parser().parse_args(argv))
    kind = args.pop("kind")
    action = args.pop("action", None)
    manifest = args.pop("manifest", None)

    # CLI flags to broker field names (the wire keeps snake_case).
    if "id" in args:
        args["sandbox_id"] = args.pop("id")
    if "memory" in args:
        args["mem_gib"] = args.pop("memory")
    if "ttl" in args:
        args["ttl_s"] = args.pop("ttl")
    if "ready_timeout" in args:
        args["ready_timeout_s"] = args.pop("ready_timeout")
    if "timeout" in args:
        args["timeout_s"] = args.pop("timeout")
    if "build_timeout" in args:
        args["build_timeout_s"] = args.pop("build_timeout")

    if "inputs" in args:
        args["inputs"] = [str(Path(p).resolve()) for p in args["inputs"]]
    # Resolve caller-relative paths: the broker host has a different cwd.
    for key in ("from_dir", "local"):
        if args.get(key) is not None:
            args[key] = str(Path(args[key]).resolve())
    # The mutually-exclusive create source: send only the chosen one.
    if args.get("image") is None:
        args.pop("image", None)
    if args.get("from_dir") is None:
        args.pop("from_dir", None)

    client = Client()
    try:
        result = client.call(kind + ("." + action if action else ""), **args)
        if "error" in result:
            # the broker reports failures as {"error": ...}; surface
            # them as a failed command, not as a success payload
            raise RuntimeError(str(result["error"]))
        if manifest is not None:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Create-only: an earlier declaration is never silently replaced.
            with manifest.open("x") as handle:
                json.dump(
                    {
                        "schema": "private-corpus-v1",
                        "check_id": result["check_id"],
                        "files": result.get("files", args.get("inputs", [])),
                    },
                    handle,
                    indent=2,
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if kind == "sandbox" and action == "exec":
            return 0 if result.get("returncode", result.get("exit_code", 0)) == 0 else 1
        if kind == "contamination":
            verdict = str(result.get("verdict", "")).lower()
            return 0 if "no_matches" in verdict else 3
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
