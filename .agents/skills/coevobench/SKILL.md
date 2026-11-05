---
name: coevobench
description: >
  CoEvoBench operations: launch and monitor bounded LLM-researcher runs, evaluate
  model-harness candidates, manage sandbox environments, screen training corpora,
  freeze nominations and run final evaluation. Use for CoEvoBench CLI, broker,
  deployment, runtime configuration, lifecycle recovery and benchmarking workflows.
---

# CoEvoBench operations

This repository is the platform. A researcher is a model driving a CLI agent to
improve a supplied 9B base model and its harness. Read `AGENTS.md` for the module
map and `docs/operations.md` for deployment. Chinese guides use the `_zh.md` suffix.

## Core contracts

- Use run, researcher, candidate, nomination, Visible, Selection, Test, freeze and
  final evaluation. The harness is the target agent scaffold.
- Reference budget: 24 hours, four GPUs and a $100 teacher allowance; three runs
  per researcher–benchmark pair. Identify smaller engineering runs explicitly.
  The broker teacher forwarder does not meter USD; spending controls are operator-owned.
- Core packages: `protocol`, `evaluation`, `grading`, `lifecycle`, `contamination`.
  Benchmark bindings live in `bench/`; providers in `runtime/`; broker handlers
  in `research/`. Follow the import graph enforced by `tests/status/test_layering.py`.
- Original code uses MIT. Preserve third-party licenses and notices.

## Prepare and launch

Use a checkout; templates and protocol assets are repository resources.

```bash
git submodule update --init
uv sync --locked --extra mcp --extra benches
uv run ceb doctor
uv run ceb runtime list
uv run ceb data fetch tb
uv run ceb data materialize tb
uv run ceb data build-images tb
```

Operator benchmark dependencies and the serving executable must be available to
the broker, not just the run image. vLLM may use an isolated venv, with its executable
linked beside broker Python or selected by Python composition. `uv.lock` does not
lock manually added benchmark/ML dependencies; `uv sync` may remove them. Use the
prepared environment or `uv run --no-sync` for those deployments.
Docker RunHost requires a daemon sharing its filesystem;
GPU runs require NVIDIA drivers and container support. The BFCL engine is pinned
in `vendor/gorilla`. Telecom uses upstream `sierra-research/tau2-bench`, not the
unrelated PyPI `tau2` package.

```bash
uv run ceb run launch --bench tb --researcher-cli claude-code \
  --researcher-model "<supported-model>" --budget-hours 24 --gpus 0,1,2,3 \
  --runhost docker --image coevobench-run:latest \
  --base-dir /path/to/materialized/base
```

Set `RESEARCHER_BASE_URL` and `RESEARCHER_TOKEN` in the operator environment.
Never print or commit tokens. Explicit launch arguments select model, CLI, budget,
GPUs and host. `[researcher]` alone does not override them. Provider configuration
uses `defaults < ~/.ceb/ceb.toml < ./ceb.toml < CEB_*`, with nested environment keys
such as `CEB_SANDBOX__RUNTIME=docker`.

## Workspace tools

```bash
./timer.sh status
./eval --split visible --model ./base --workers 4
./eval --split selection --model work/weights/v1 --k 1
./eval --list-tasks
./eval --split selection --tasks A-xxxxxxxx --model work/weights/v1
./sandbox create --from-dir work/my-task --ttl 1800
./sandbox exec SESSION_ID 'python /task/verify.py' --timeout 120
./sandbox close SESSION_ID
./track -- python work/train.py
./submit --bench tb
```

`--from-dir` builds and runs on the same sandbox backend; `--image REF` starts an
existing image. There is no standalone image-build axis. Transfer argument order
is `SESSION_ID LOCAL_PATH REMOTE_PATH` for both upload and download. Python client
calls use `sandbox_id=...`; exec output is in `text`.
Export before close; the broker requires a registered session. Its ledger omits
command stdout. Uploads preserve bytes but downloads use text I/O; base64-encode
binary downloads. CLI memory defaults to 4 GiB, raw requests to 8 GiB. Modal does
not currently enforce the context-build timeout argument.

Keep weights, data, templates and build contexts under the persistent workspace.
Container-local `/tmp` is not visible to operator serving. `/workspace/base` maps
to read-only `/base` in Docker; lifecycle is mounted at `/run/ceb` as a directory.
The standard provisioner does not create `ROUND.md` or `official_protocol.json`.

Selection exposes aggregate feedback and anonymous Selection IDs only. The current
`--formal` helper overwrites flags with SWE constants. AppWorld then applies its own
formal options; BFCL rejects the injected nonzero wall cap. Use explicit TB settings
for a TB self-check (600s, 120 steps, 5120 output tokens, context 32768, temperature 0).
Ordinary SWE/TB broker defaults are 80 steps/900s/5120, with recipe-based context.
BFCL generic `score` averages verdicts without its 0.6/0.4 domain aggregation helper.

## Serving and data

Launch snapshots `[serve.defaults]` into editable workspace `serve_config.json`;
`max_model_len` becomes `max_len`. `--serve-config` replaces the default, explicit
parser/context flags override it, and template paths are recipe-relative.
`eval.plan` supplies a workload-based client timeout. Run-owned logs are in `audit/serve/`.

For Qwen3.5-9B on two A10s, the tested text-only recipe is
`examples/serve/qwen35-9b-2xa10.json`: TP=2, 32k context, max_num_seqs=32,
gpu_mem_util=0.9, eager mode, image/video limits zero. Readiness, concurrent long
requests and a bounded Visible evaluation were tested with **vLLM 0.31.0**. Current
0.32.0/Transformers 5.19.0 image recipes still need live Docker/GPU acceptance.
Serialize GPU training, evaluation and local rollout serving.

SWE/TB corpus screening uses `./contamination check --input ... --manifest ...`:
at most five distinct versions, at least 100 distinct records, up to 32 files and
1 GiB. Keep exact bytes and provenance. NO_MATCHES is not a proof; other benchmarks
do not currently have this broker service. Materialization and provisioning filter
public views to Visible and remove stale generated non-Visible JSON. Full trajectories are not retained
uniformly across adapters.

## Monitor and stop

```bash
uv run ceb run status --run-id RUN_ID --json
uv run ceb run stop RUN_ID
```

Run evidence is in `~/.ceb/runs/RUN_ID/`. Check actual training logs/checkpoints,
`workspace/out/eval/*.jsonl`, `select_runs/`, `audit/serve/`, broker/monitor logs and
`workspace/out/researcher-state.json`. Pair tool results with tool-use IDs when
reading researcher transcripts. Log growth, GPU use, low loss and Submitted are
not task-solving scores.

Claude recovery counts novel successful tools and bounds fresh-session recovery
and finalization between CLI invocations. The 30-minute/DONE.md check does not
interrupt a continuing invocation. Codex/OpenCode do not use this loop.
Workspace and the original deadline persist. DONE.md alone is
not submission; only the real request fingerprint counts. Never manufacture a
submission during operator cleanup.
The digest covers nomination files and DONE.md, not all externally referenced weights.

The detached monitor handles deadline, researcher exit and broker loss. The
shared stop path attempts Docker logs and Claude/Codex session archival, records
`audit/submission-status.json`, and
reaps owned receipts/labels. TTL cleanup exceptions are caught with backoff;
Docker concurrent deletion is awaited. `eval-resources/cleanup-events.jsonl`
retains failure/release history; `cleanup-status` is current state. Retry pending
cleanup with `ceb run stop`, not a global prune or broad process kill.

Local/OpenCode HOME archival is not implemented. OpenCode tool events are in
`workspace/out/opencode-tools.jsonl`; Claude transitions are in
`workspace/out/researcher-loop.jsonl`. Status still uses Docker inspection and
counts only `audit/bash.log`; zero counters do not cover JSON tool streams or prove
absence of violations. Inspect actual evidence, not just the status summary.

## Deployment and validation

For servers without repository access:

```bash
python3 scripts/sync-server.py operator@server.example /srv/CoEvoBench
```

Stage new source first. The tool packages indexed files and pinned submodules as
tar.zst, transfers over SCP and verifies hashes; only previously managed stale
files are deleted. Rebuild `Dockerfile.run` after template/envelope changes and
verify its fingerprint before a new real-researcher launch. Use an idle checkout
for deployment when a run is active. Documentation or repair work does not itself
authorize launching a new long run.

CLI RunHost choices are local/docker. Modal/Daytona are remote primitives requiring
separate broker transport. Sandbox and serving providers have separate interfaces. Do not claim
every registered cloud combination has completed full lifecycle validation.
Test the selected deployment's workspace transfer, service connectivity and cleanup.

CLI bootstrap initializes the selected researcher inside its runtime. The image
includes Claude Code, Codex and OpenCode V2. AppWorld's policy Dockerfile is in
`deploy/`; native setup uses the pinned upstream package dependencies. `uv.lock`
is tracked and CI checks scripts, all tests and packaging from a submodule-complete checkout.

MCP exposes twelve operator tools, documented in `docs/mcp.md`; broker ops are not
MCP tools. Use `uv run ceb mcp` from the checkout.

## Experimental final evaluation

CLI registers only SWE; its serving lifetime and report printing have known defects.
AppWorld/BFCL factories are not CLI-registered; TB/Telecom have no CLI replay binding.
Telecom's four-trial reference is not applied automatically (`k=1` is the default).
`prepare` returns `destination/(label or nomination.name)`, not the destination itself.
Whole-bundle weight/template verification and post-replay checks are incomplete;
comparison assumes minisweagent and still has staging/re-freezing gaps. Do not present
these interfaces as a validated final-evaluation or matched-control workflow.
