"""Full MCP stdio roundtrip through the official SDK client.

This is the path real agents take: spawn the server, handshake, list
tools, call one. Requires the mcp extra; skipped when unavailable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp.client.stdio import stdio_client  # noqa: E402

from mcp import ClientSession, StdioServerParameters  # noqa: E402

_SRC = Path(__file__).resolve().parents[2] / "src"

EXPECTED_TOOLS = {
    "bench_protocol",
    "doctor",
    "gpu_inventory",
    "list_runs",
    "run_status",
    "launch_run",
    "stop_run",
    "run_lifecycle",
    "eval_records",
    "audit_summary",
    "report",
    "data_fetch",
}


async def test_stdio_handshake_and_tool_call(tmp_path: Path) -> None:
    registry = tmp_path
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "coevobench.mcp", "--registry-root", str(registry)],
        env=env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        assert names >= EXPECTED_TOOLS

        result = await session.call_tool("bench_protocol", {})
        payload = json.loads(result.content[0].text)
        assert payload["splits"]["swe"]["test"] == 253

        listing = await session.call_tool("list_runs", {})
        assert json.loads(listing.content[0].text)["runs"] == []
