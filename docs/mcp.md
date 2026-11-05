# Operator MCP server

[中文](mcp_zh.md) · [README](../README.md)

The stdio server wraps the same operation and collection functions used by the CLI.
It runs in the operator environment and inherits that environment's permissions.
The researcher workspace uses its broker clients, not this MCP server.

## Start and connect

```bash
uv sync --locked --extra mcp
uv run ceb mcp --registry-root /path/to/registry
```

Omit `--registry-root` for `~/.ceb`. Run in the repository checkout so configuration,
templates and protocol assets can be resolved. For a generic stdio MCP client:

```json
{
  "mcpServers": {
    "coevobench": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/CoEvoBench", "run", "--locked", "ceb", "mcp"]
    }
  }
}
```

With Claude Code:

```bash
claude mcp add coevobench -- uv --directory /absolute/path/to/CoEvoBench run --locked ceb mcp
```

## Current tools

| Tool | Parameters | Purpose |
|---|---|---|
| `bench_protocol` | none | Frozen split sizes and configured budget summary |
| `doctor` | none | Environment and serving feasibility checks |
| `gpu_inventory` | none | Detected GPU profiles |
| `list_runs` | none | Collected overview documents |
| `run_status` | optional `run_id` | One run or all collected status documents |
| `launch_run` | `bench`; optional model/CLI, hours, comma-separated `gpus`, `image` | Launch with the shared lifecycle operation |
| `stop_run` | `run_id` | Ordered, retryable shutdown and submission inspection |
| `run_lifecycle` | `run_id` | Persisted deadline, phase and events |
| `eval_records` | `run_id`, optional `limit` | Operator-side Selection summaries |
| `audit_summary` | `run_id` | Collected audit volume and guard counters |
| `report` | `results_dir` | Aggregate final evaluation results |
| `data_fetch` | `bench` | Cache a benchmark's frozen split definition |

The current MCP launch wrapper does not expose `base_dir` or `runhost`; use the CLI
or Python operation when those arguments are required. It does not expose
`eval.run`, `contamination.check`, GPU jobs or sandbox ops as MCP tools: those are
broker operations used by the researcher clients. Data materialization uses its
CLI/Python interfaces. Final-eval has open integration gaps.

## Interpreting status

`run_status` and the web API use the same collector as `ceb run status --json`.
Credential-shaped values in collected status documents are redacted. Lifecycle
and result tools read operator-side artifacts; they are not a researcher feedback
channel. A stopped run may have an incomplete submission. Use `audit/stop.json`,
`audit/submission-status.json` and resource receipts when diagnosing shutdown.

`container` still uses Docker inspection even for local runs. `audit_summary` counts
only `audit/bash.log`, excluding newer JSON tool logs. Zero counters are not evidence
of zero tool calls or violations; inspect actual event logs and available archives.

Example tool sequence: launch an example toy run with `image="python:3.11-slim"`,
record the returned `run_id`, inspect it with `run_status`, then call `stop_run`.
Use actual result records rather than illustrative scores when reporting outcomes.
