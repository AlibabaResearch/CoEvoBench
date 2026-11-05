"""ceb entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coevobench import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ceb", description="CoEvoBench control CLI")
    parser.add_argument("--version", action="version", version=f"ceb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="environment preflight")
    doctor_parser.add_argument("--project-dir", default=".")

    run_parser = sub.add_parser("run", help="development run lifecycle")
    run_sub = run_parser.add_subparsers(dest="run_command", required=True)

    launch_parser = run_sub.add_parser("launch", help="launch a development run")
    launch_parser.add_argument("--bench", default="toy")
    launch_parser.add_argument("--researcher-model", default="example")
    launch_parser.add_argument("--researcher-cli", default="example")
    launch_parser.add_argument("--budget-hours", type=float, default=0.05)
    launch_parser.add_argument("--gpus", default="")
    launch_parser.add_argument("--image", default=None)
    launch_parser.add_argument(
        "--runhost", default="docker",
        choices=["local", "docker"],
        help="where the broker-connected researcher runs: local process or Docker",
    )
    launch_parser.add_argument(
        "--base-dir", default=None, help="base weights dir, mounted read-only at /base"
    )
    launch_parser.add_argument("--registry-root", default=None)

    status_parser = run_sub.add_parser("status", help="inspect runs")
    status_parser.add_argument("--run-id", default=None)
    status_parser.add_argument("--registry-root", default=None)
    status_parser.add_argument("--json", action="store_true", dest="as_json")

    stop_parser = run_sub.add_parser("stop", help="stop a run")
    stop_parser.add_argument("run_id")
    stop_parser.add_argument("--registry-root", default=None)

    eval_parser = sub.add_parser("eval", help="development evaluation")
    eval_parser.add_argument("--bench", default="toy")
    eval_parser.add_argument("--split", default="visible")
    eval_parser.add_argument("--k", type=int, default=1)
    eval_parser.add_argument("--workers", type=int, default=1)
    eval_parser.add_argument("--n", type=int, default=None)
    eval_parser.add_argument("--tasks", default=None)
    eval_parser.add_argument("--candidate", default="cand0")
    eval_parser.add_argument("--out-dir", default=None)
    eval_parser.add_argument("--registry-root", default=None)
    eval_parser.add_argument("--model", default=None, help="weights under evaluation")
    eval_parser.add_argument(
        "--serve-config", default=None, help="candidate serving recipe (JSON file)"
    )
    eval_parser.add_argument("--tool-call-parser", default=None)
    eval_parser.add_argument("--reasoning-parser", default=None)
    eval_parser.add_argument("--chat-template", default=None)
    eval_parser.add_argument("--temperature", type=float, default=None)
    eval_parser.add_argument("--max-tokens", type=int, default=None)
    eval_parser.add_argument("--wall-cap", type=int, default=None)
    eval_parser.add_argument("--step-limit", type=int, default=None)
    eval_parser.add_argument("--max-model-len", type=int, default=None)
    eval_parser.add_argument(
        "--formal", action="store_true",
        help="shared SWE evaluation presets; use explicit limits for other benchmarks",
    )
    eval_parser.add_argument(
        "--list-tasks", action="store_true",
        help="list anonymous task ids without evaluating",
    )

    freeze_parser = sub.add_parser("freeze", help="freeze a nominated candidate")
    freeze_parser.add_argument("candidate_dir")
    freeze_parser.add_argument("--label", required=True)

    calibrate_parser = sub.add_parser(
        "calibrate", help="two-gate oracle calibration (gold must resolve, empty must fail)"
    )
    calibrate_parser.add_argument("bench", choices=["swe"])
    calibrate_parser.add_argument(
        "--tasks", default="",
        help="comma-separated instance ids; default: a sample across repos",
    )
    calibrate_parser.add_argument(
        "--n", type=int, default=5, help="sample size when --tasks is not given"
    )
    calibrate_parser.add_argument(
        "--state", default=None,
        help="trusted-state path (default: cache/swe_oracle_trusted.json)",
    )

    verify_parser = sub.add_parser("verify", help="check a frozen candidate")
    verify_parser.add_argument("candidate_dir")

    audit_parser = sub.add_parser("audit", help="compile the audit evidence pack")
    audit_parser.add_argument("run_root")
    audit_parser.add_argument("--out", default=None)

    provenance_parser = sub.add_parser("provenance", help="weights lineage gates")
    provenance_parser.add_argument("submission")
    provenance_parser.add_argument("--base", required=True)
    provenance_parser.add_argument("--roots", default="")

    report_parser = sub.add_parser("report", help="aggregate final-eval results")
    report_parser.add_argument("results_dir")

    fe_parser = sub.add_parser("final-eval", help="the operator's Test-split path")
    fe_sub = fe_parser.add_subparsers(dest="final_eval_command", required=True)
    fe_prepare = fe_sub.add_parser("prepare", help="freeze a nomination into a bundle")
    fe_prepare.add_argument("submission_dir")
    fe_prepare.add_argument("--bench", required=True)
    fe_prepare.add_argument("--destination", required=True)
    fe_prepare.add_argument("--label", default=None)
    fe_prepare.add_argument("--base-weights", default=None)
    fe_eval = fe_sub.add_parser("evaluate", help="replay the bundle on Test")
    fe_eval.add_argument("bundle")
    fe_eval.add_argument("--bench", required=True)
    fe_eval.add_argument("--k", type=int, default=1)
    fe_eval.add_argument(
        "--revert-weights", default=None,
        help="base weights for the Eq.5 cheating outcome (theta_0, H*)",
    )

    cmp_parser = sub.add_parser(
        "comparison", help="the M0H0/M0H1/M1H1 matched-configuration controls"
    )
    cmp_sub = cmp_parser.add_subparsers(dest="comparison_command", required=True)
    cmp_prepare = cmp_sub.add_parser("prepare", help="stage the three role bundles")
    cmp_prepare.add_argument("submission_dir")
    cmp_prepare.add_argument("--bench", required=True)
    cmp_prepare.add_argument("--destination", required=True)
    cmp_prepare.add_argument("--base-weights", required=True)
    cmp_prepare.add_argument("--seed-harness", required=True)
    cmp_run = cmp_sub.add_parser("evaluate", help="grade all three roles on Test")
    cmp_run.add_argument("--bench", required=True)
    cmp_run.add_argument("--bundles", required=True, help="bundles.json from prepare")
    cmp_run.add_argument("--k", type=int, default=1)

    data_parser = sub.add_parser("data", help="dataset acquisition")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    fetch_parser = data_sub.add_parser("fetch", help="fetch and cache bench data")
    fetch_parser.add_argument("bench")
    materialize_parser = data_sub.add_parser(
        "materialize", help="write public views + judge details"
    )
    materialize_parser.add_argument("bench")
    materialize_parser.add_argument("--workspace", default=None)
    images_parser = data_sub.add_parser("build-images", help="build public task images locally")
    images_parser.add_argument("bench", choices=["tb"])
    images_parser.add_argument("--tasks", help="comma-separated task IDs; default: frozen pool")
    images_parser.add_argument("--timeout", type=int, default=1800)

    web_parser = sub.add_parser("web", help="localhost console + JSON API")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8765)
    web_parser.add_argument("--registry-root", default=None)

    mcp_parser = sub.add_parser("mcp", help="MCP server for agents (stdio)")
    mcp_parser.add_argument("--registry-root", default=None)

    runtime_parser = sub.add_parser(
        "runtime", help="registered runtime implementations"
    )
    runtime_sub = runtime_parser.add_subparsers(
        dest="runtime_command", required=True
    )
    runtime_sub.add_parser("list", help="list runtimes by SPI kind")

    sub.add_parser(
        "client",
        help="researcher-side broker client: sandbox / contamination (see `ceb client -h`)",
    )

    submit_parser = sub.add_parser(
        "submit", help="explicit submission signal (request/confirm)"
    )
    submit_parser.add_argument("--workspace", default=".")
    submit_parser.add_argument("--bench", required=True)
    submit_parser.add_argument(
        "--confirm", action="store_true",
        help="operator side: verify the request digest still matches",
    )

    track_parser = sub.add_parser(
        "track", help="research-effort span timing (the ./track entry)"
    )
    track_parser.add_argument(
        "--output", default="out/research-timing.jsonl",
        help="span ledger path (append-only)",
    )
    # the positional must not collide with the dispatch's own
    # "command" name (it would overwrite the subcommand and every
    # later match failed: `ceb track` alone died as "unknown command []")
    track_parser.add_argument(
        "track_command", metavar="COMMAND", nargs="*",
        help="command to time; omit for a summary",
    )

    results_parser = sub.add_parser("results", help="recorded results (scoreboard)")
    results_parser.add_argument("--bench", default=None)
    results_parser.add_argument("--run-id", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    # The researcher-side client owns its own subcommand tree; route it
    # before argparse so `ceb client -h` shows that tree's real help
    # (a REMAINDER shim would swallow -h into the outer usage).
    rest = list(sys.argv[1:]) if argv is None else list(argv)
    if rest and rest[0] == "client":
        from coevobench.research.client import main as client_main

        return client_main(rest[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    # The envelope distribution installs only the researcher-facing
    # surface; operator commands die with a one-line explanation here
    # instead of an import traceback to debug.
    _ENVELOPE_COMMANDS = {"eval", "submit", "track", "results"}
    try:
        import os

        from coevobench.research.eval_client import BROKER_SOCKET  # noqa: F401

        in_envelope = os.path.exists(BROKER_SOCKET)
    except Exception:
        in_envelope = False
    if in_envelope and args.command not in _ENVELOPE_COMMANDS:
        print(
            f"[ceb] {args.command!r} is an operator command; inside the run "
            "envelope the researcher surface is: eval, submit, track, "
            "results, client",
            file=sys.stderr,
        )
        return 2

    if args.command == "doctor":
        from coevobench.cli import doctor
        from coevobench.config.loader import load_config

        cfg = load_config(project_dir=None)
        return doctor.run(cfg)
    if args.command == "run":
        if args.run_command == "launch":
            gpus = tuple(int(g) for g in args.gpus.split(",") if g.strip().isdigit())
            from coevobench.cli import run as run_cli

            return run_cli.launch(
                bench=args.bench,
                researcher_model=args.researcher_model,
                researcher_cli=args.researcher_cli,
                budget_hours=args.budget_hours,
                gpus=gpus,
                image=args.image,
                base_dir=args.base_dir,
                registry_root=Path(args.registry_root) if args.registry_root else None,
                runhost=args.runhost,
            )
        if args.run_command == "status":
            from coevobench.cli import run as run_cli

            return run_cli.status(
                run_id=args.run_id,
                registry_root=Path(args.registry_root) if args.registry_root else None,
                as_json=args.as_json,
            )
        if args.run_command == "stop":
            from coevobench.cli import run as run_cli

            return run_cli.stop(
                args.run_id,
                registry_root=Path(args.registry_root) if args.registry_root else None,
            )
    if args.command == "eval":
        from coevobench.cli.eval import run_eval

        return run_eval(
            args.bench,
            args.split,
            k=args.k,
            n=args.n,
            tasks=args.tasks,
            candidate=args.candidate,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            registry_root=Path(args.registry_root) if args.registry_root else None,
            workers=args.workers,
            model=args.model,
            serve_config=args.serve_config,
            tool_call_parser=args.tool_call_parser,
            reasoning_parser=args.reasoning_parser,
            chat_template=args.chat_template,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            wall_cap=args.wall_cap,
            step_limit=args.step_limit,
            max_model_len=args.max_model_len,
            formal=args.formal,
            list_tasks=args.list_tasks,
        )
    if args.command == "freeze":
        from coevobench.cli.deliver import freeze_cmd

        return freeze_cmd(args.candidate_dir, args.label)
    if args.command == "runtime" and args.runtime_command == "list":
        from coevobench.runtime.registry import list_runtimes

        for kind, names in sorted(list_runtimes().items()):
            print(f"{kind:10s} {', '.join(names)}")
        return 0
    if args.command == "submit":
        from coevobench.cli.workspace import submit_cmd

        return submit_cmd(args.workspace, args.bench, confirm=args.confirm)
    if args.command == "track":
        from coevobench.cli.workspace import track_cmd

        return track_cmd(args.output, list(args.track_command))
    if args.command == "results":
        from coevobench.cli.workspace import results_cmd

        return results_cmd(bench=args.bench, run_id=args.run_id)
    if args.command == "final-eval":
        from coevobench.cli.deliver import final_eval_cmd

        return final_eval_cmd(args)
    if args.command == "comparison":
        from coevobench.cli.deliver import comparison_cmd

        return comparison_cmd(args)
    if args.command == "calibrate":
        from coevobench.cli.eval import calibrate_cmd

        return calibrate_cmd(
            args.bench,
            tasks=args.tasks,
            n=args.n,
            state=Path(args.state) if args.state else None,
        )
    if args.command == "verify":
        from coevobench.cli.deliver import verify_cmd

        return verify_cmd(args.candidate_dir)
    if args.command == "report":
        from coevobench.cli.deliver import report_cmd

        return report_cmd(args.results_dir)
    if args.command == "audit":
        from coevobench.cli.deliver import audit_cmd

        return audit_cmd(args.run_root, args.out)
    if args.command == "provenance":
        from coevobench.cli.deliver import provenance_cmd

        return provenance_cmd(args.submission, args.base, args.roots)
    if args.command == "data" and args.data_command == "fetch":
        from coevobench.cli.data import fetch
        from coevobench.config.loader import load_config

        cfg = load_config(project_dir=None)
        return fetch(args.bench, cfg)
    if args.command == "data" and args.data_command == "materialize":
        from coevobench.cli.data import materialize_bench
        from coevobench.config.loader import load_config

        cfg = load_config(project_dir=None)
        result = materialize_bench(
            args.bench, cfg,
            workspace=Path(args.workspace) if args.workspace else None,
        )
        print(f"[data] {args.bench}: {result['status']} {result.get('detail', result)}")
        return 0 if result["status"] in ("ok",) else 1
    if args.command == "data" and args.data_command == "build-images":
        import json

        from coevobench.bench.tb.images import build_images

        tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
        try:
            print(json.dumps(build_images(task_ids=tasks, timeout_s=args.timeout), indent=2))
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"[data] {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "web":
        from coevobench.status.web import serve

        return serve(
            host=args.host,
            port=args.port,
            registry_root=Path(args.registry_root) if args.registry_root else None,
        )
    if args.command == "mcp":
        from coevobench.mcp.server import main as mcp_main

        argv = ["--registry-root", args.registry_root] if args.registry_root else []
        return mcp_main(argv)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
