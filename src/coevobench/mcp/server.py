"""MCP server: ceb operations for operator-side agents (stdio transport).

Tools wrap the same functions the CLI uses — one data layer, three
frontends (CLI, web, MCP). The server inherits operator privileges
(docker, run launches): expose it only in operator-controlled agent
sessions. Researchers inside run envelopes never see this server; their
surface is the workspace contract. Status documents pass through the
collector's redaction before leaving.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coevobench.config.loader import load_config
from coevobench.config.schema import Config
from coevobench.core.lifecycle.registry import RunRegistry
from coevobench.status.collect import collect_all, collect_run, redact


class ToolContext:
    """Configuration and registry shared by every tool."""

    def __init__(self, registry_root: Path | None = None, cfg: Config | None = None) -> None:
        self.registry = RunRegistry(registry_root)
        self.cfg = cfg or load_config()
        self.registry_root = registry_root


def make_tools(ctx: ToolContext) -> dict[str, Callable[..., dict]]:
    """Build the tool handlers; stdlib-only, directly callable for tests."""

    def bench_protocol() -> dict:
        """Frozen protocol constants: splits, budget, replication."""
        from coevobench.core.protocol.splits import FROZEN_SIZES

        return {
            "splits": FROZEN_SIZES,
            "budget": {
                "hours": ctx.cfg.budget.hours,
                "gpus_per_run": ctx.cfg.budget.gpus_per_run,
                "teacher_spend_usd": ctx.cfg.budget.teacher_spend_usd,
            },
            "replication": (
                "three independent development runs per researcher-benchmark "
                "pair; means reported (paper Tables 1-2)"
            ),
        }

    def doctor() -> dict:
        """Environment preflight: python, docker, GPUs, registry."""
        from coevobench.cli.doctor import collect_checks

        return {
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail}
                for c in collect_checks(ctx.cfg)
            ]
        }

    def gpu_inventory() -> dict:
        """Detected GPUs with their profiles."""
        from coevobench.runtime.gpu import profile as gpu

        devices = gpu.detect()
        return {
            "gpus": [
                {"index": d.index, "profile": d.profile.name, "vram_gib": d.profile.vram_gib}
                for d in devices
            ]
        }

    def list_runs() -> dict:
        """Overview documents for every registered run."""
        return {"runs": [redact(d) for d in collect_all(ctx.registry)]}

    def run_status(run_id: str | None = None) -> dict:
        """Full status document for one run, or all runs when omitted."""
        if run_id is None:
            return {"runs": [redact(d) for d in collect_all(ctx.registry)]}
        doc = collect_run(ctx.registry, run_id)
        if doc is None:
            return {"error": f"unknown run {run_id}"}
        return redact(doc)

    def launch_run(
        bench: str,
        researcher_model: str = "example",
        researcher_cli: str = "example",
        budget_hours: float = 0.05,
        gpus: str = "",
        image: str | None = None,
    ) -> dict:
        """Launch a development run; returns run_id, container, workspace."""
        from coevobench.cli.run import launch_run as _launch

        return _launch(
            bench=bench,
            researcher_model=researcher_model,
            researcher_cli=researcher_cli,
            budget_hours=budget_hours,
            gpus=tuple(int(g) for g in gpus.split(",") if g.strip().isdigit()),
            image=image,
            registry_root=ctx.registry_root,
        )

    def stop_run(run_id: str) -> dict:
        """Stop a run: remove its container and close the lifecycle."""
        from coevobench.cli.run import stop_run as _stop

        return _stop(run_id, registry_root=ctx.registry_root)

    def run_lifecycle(run_id: str) -> dict:
        """The persisted lifecycle record (deadline, phase, events)."""
        path = ctx.registry.dir_for(run_id) / "lifecycle.json"
        if not path.is_file():
            return {"error": f"no lifecycle record for {run_id}"}
        return json.loads(path.read_text())

    def eval_records(run_id: str, limit: int = 10) -> dict:
        """Selection-record summaries for a run, newest first."""
        select_dir = ctx.registry.select_dir(run_id)
        summaries = []
        for path in sorted(select_dir.glob("*.jsonl"), reverse=True)[: max(0, limit)]:
            lines = path.read_text().splitlines()
            if lines:
                summaries.append(json.loads(lines[0]))
        return {"run_id": run_id, "records": summaries}

    def audit_summary(run_id: str) -> dict:
        """Guard hit counts and audit log volume for a run."""
        doc = collect_run(ctx.registry, run_id)
        if doc is None:
            return {"error": f"unknown run {run_id}"}
        return {
            "run_id": run_id,
            "guard_hits": doc["guard_hits"],
            "audit_lines": doc["audit_lines"],
        }

    def report(results_dir: str) -> dict:
        """Aggregate final-eval results: per-domain rows plus the weighted Avg."""
        from coevobench.core.lifecycle.report import aggregate_dir

        try:
            return aggregate_dir(Path(results_dir))
        except ValueError as exc:
            return {"error": str(exc)}

    def data_fetch(bench: str) -> dict:
        """Fetch and cache one bench's data (on-demand acquisition)."""
        from coevobench.cli.data import fetch_bench

        return fetch_bench(bench, ctx.cfg)

    return {
        "bench_protocol": bench_protocol,
        "doctor": doctor,
        "gpu_inventory": gpu_inventory,
        "list_runs": list_runs,
        "run_status": run_status,
        "launch_run": launch_run,
        "stop_run": stop_run,
        "run_lifecycle": run_lifecycle,
        "eval_records": eval_records,
        "audit_summary": audit_summary,
        "report": report,
        "data_fetch": data_fetch,
    }


def build_server(ctx: ToolContext) -> Any:
    """Register the tools on an MCP server instance (requires the mcp extra).

    Supports both protocol generations: mcp 2.x (MCPServer) and 1.x
    (FastMCP). Both expose the same decorator and stdio run().
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "the MCP server needs the mcp extra: uv add 'coevobench[mcp]'"
            ) from exc

    server = MCPServer("coevobench")
    for name, handler in make_tools(ctx).items():
        server.tool(name=name)(handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coevobench-mcp", description="CoEvoBench MCP server (stdio)"
    )
    parser.add_argument("--registry-root", default=None)
    args = parser.parse_args(argv)
    ctx = ToolContext(Path(args.registry_root) if args.registry_root else None)
    build_server(ctx).run()
    return 0


__all__ = ["ToolContext", "build_server", "main", "make_tools"]

if __name__ == "__main__":
    raise SystemExit(main())
