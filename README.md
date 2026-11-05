# CoEvoBench

[中文](README_zh.md) · [Operations](docs/operations.md) · [Architecture](docs/architecture.md)
· [Protocol](docs/protocol.md) · [MCP](docs/mcp.md)

CoEvoBench evaluates **LLM researchers on autonomous model–harness co-evolution**.
A researcher is a model driving a CLI agent. Within a bounded development run it
improves a supplied 9B base model, its initial harness, or both. The platform
evaluates the nominated model–harness pair on held-out tasks.

The reference protocol allocates **24 hours, four GPUs and a $100 teacher-model
allowance**, with three independent runs per researcher–benchmark pair. Smaller
hardware configurations and short runs are useful for engineering validation;
record those differences when reporting results.

| Benchmark key | Domain | Visible | Selection | Test |
|---|---|---:|---:|---:|
| `swe` | SWE-bench Verified, frozen 493-instance subset | 40 | 200 | 253 |
| `tb` | TB-Lite | 20 | 40 | 40 |
| `appworld` | AppWorld easy | 27 | 51 | 51 |
| `tau` | τ²-bench Telecom | 23 | 46 | 45 |
| `bfcl` | BFCL-V4 multi-turn and memory scoring records | 235 | 470 | 560 |

**Implementation status:** Docker-hosted TB runs have exercised researcher tool
use, data preparation, SFT, merged checkpoints, candidate evaluation and resource
cleanup on an earlier deployment. The current image recipes have not yet completed
Docker/GPU acceptance. Operator final-eval and comparison still have wiring and
integrity gaps. The repository also contains other benchmark and provider adapters;
their presence does not imply that every provider/benchmark combination has
completed a full end-to-end run.

## Install from a checkout

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). Linux with NVIDIA drivers,
NVIDIA Container Toolkit and a local Docker daemon is the validated GPU-run setup.
Core development and many tests also work on macOS. The Docker daemon must share
the operator's filesystem for bind-mounted run workspaces.

```bash
git clone --recurse-submodules <repository-url> CoEvoBench
cd CoEvoBench
uv sync --locked --extra mcp --extra benches
uv run ceb --help
uv run ceb doctor
```

Run from the checkout: templates and protocol assets are repository resources.
The BFCL engine is pinned at `vendor/gorilla`; initialize an existing checkout with
`git submodule update --init`. Benchmark content is acquired separately, while
split definitions and image manifests ship in `src/coevobench/bench/`. The BFCL
submodule also contains upstream data assets.

The operator's Python environment must contain the dependencies used by its
benchmark and serving runtime. The run image is a separate environment. Follow
[the dependency and deployment guide](docs/operations.md#dependencies) before a
GPU run; the project's base dependencies alone do not install a training stack.

## Quickstart: no-GPU smoke run

The example researcher is a lifecycle fixture, not an autonomous research agent.
This checks workspace provisioning, the broker and automatic shutdown:

```bash
uv run ceb run launch --bench toy --researcher-cli example \
  --researcher-model example --image python:3.11-slim --budget-hours 0.05
uv run ceb run status --json
# Use the run_id printed by launch:
uv run ceb run stop example_toy_1
```

The built-in toy evaluator checks scripted sum-of-squares tasks without model
credentials or GPUs:

```bash
uv run ceb eval --bench toy --split visible
```

## Launch a researcher

Build the run image on the operator host, prepare the benchmark data, and supply
a materialized base checkpoint. Choose a researcher model supported by your
endpoint; the researcher CLI credentials and the candidate model are separate.

```bash
cp ceb.toml.example ceb.toml
docker build -f Dockerfile.run -t coevobench-run:latest .
export RESEARCHER_BASE_URL="https://your-researcher-endpoint.example"
export RESEARCHER_TOKEN="<your-token>"
uv run ceb data fetch tb
uv run ceb data materialize tb
uv run ceb data build-images tb
uv run ceb run launch --bench tb --researcher-cli claude-code \
  --researcher-model "<supported-model>" --budget-hours 24 \
  --gpus 0,1,2,3 --runhost docker --image coevobench-run:latest \
  --base-dir /path/to/materialized/base
```

Launch arguments explicitly select the researcher, budget, GPUs and run host.
Configuration uses `defaults < ~/.ceb/ceb.toml < ./ceb.toml < CEB_*`; individual
commands apply their CLI arguments. For example,
`CEB_SANDBOX__RUNTIME=docker` selects Docker sandboxes. `RESEARCHER_BASE_URL` and
`RESEARCHER_TOKEN` are read by the launch path; `[researcher]` alone does not
replace the explicit launch arguments or provide credentials.

`[serve.defaults]` is copied into the editable workspace `serve_config.json`.
The two-A10, text-only Qwen3.5-9B recipe tested with vLLM 0.31.0 is
[`examples/serve/qwen35-9b-2xa10.json`](examples/serve/qwen35-9b-2xa10.json).
See [serving configuration](docs/operations.md#serving) for the corresponding TOML.

## Researcher workspace

```text
workspace/
├── TASK.md, RESEARCH_TOOLS.md
├── harness/                    editable candidate harness
├── tasks_visible/<bench>/      materialized public task views
├── anonymous_pool_<bench>.txt  anonymous Selection IDs
├── base -> /base               read-only base checkpoint in Docker runs
├── serve_config.json           initial serving recipe, editable
├── eval, sandbox, timer.sh, submit, track, contamination
└── out/                       evaluations, notes and final nomination
```

The researcher uses the workspace entries. The envelope installs a small
allowlisted client distribution; the operator holds the grading engine and broker.
Development sandboxes execute commands and transfer files; model calls originate
in the harness/rollout driver, not in command sandboxes.

```bash
./timer.sh status
./eval --split visible --model ./base --workers 4
./eval --split selection --model work/weights/v1 --k 1
./eval --list-tasks
./sandbox create --from-dir work/my-task --ttl 1800
./track -- python work/train.py
./submit --bench tb
```

Building a development environment is part of `./sandbox create --from-dir DIR`;
there is no separate image-build command. `--image REF` remains available.
Selection feedback is aggregate-only; Test evaluation is operator-only.

The Docker image includes Claude Code, Codex and OpenCode V2. Launch provisions
the selected CLI inside its runtime before starting it. The generic launch command
supports Docker and local processes; remote RunHost primitives require a separately
configured broker transport and are not selectable as a complete run deployment.

## Recovery, results and shutdown

Claude Code recovery observes completed, successful tool results. Repeated
no-progress rounds, context-length errors and malformed tool calls trigger bounded
fresh-session recovery. Workspace artifacts and the original deadline persist.
Between CLI invocations, the loop checks for the last 30 minutes or an early
`out/DONE.md` and enters bounded finalization. These checks do not interrupt a
continuing invocation; recovery transitions also wait for it to return.
Only a valid `./submit` request completes the submission.

A detached monitor handles deadline expiry, researcher exit and broker loss.
The shared stop path attempts to archive Docker logs and Claude/Codex session
directories, records submission status and reaps owned resources.
`phase=done` means shutdown finished; inspect
`audit/submission-status.json` for submission success.

TTL cleanup exceptions are caught and retried with backoff. Docker concurrent
auto-removal is awaited; `eval-resources/cleanup-events.jsonl` retains failures and
release history after recovery. `ceb run stop RUN_ID` retries unfinished cleanup.

| Evidence | Location under the run directory |
|---|---|
| Budget and lifecycle | `manifest.json`, `lifecycle.json`, `control/lifecycle.json` |
| Claude recovery | `workspace/out/researcher-state.json`, `workspace/out/researcher-loop.jsonl` |
| OpenCode tool events | `workspace/out/opencode-tools.jsonl` |
| Available Docker/session archives | `audit/researcher/` (local/OpenCode HOME archival is not implemented) |
| Visible result records | `workspace/out/eval/*.jsonl` |
| Selection records | `select_runs/*.jsonl` (operator-side) |
| Serving logs | `audit/serve/` |
| Broker and monitor logs | `audit/broker.log`, `audit/monitor.log` |
| Cleanup | `eval-resources/cleanup-status`, `eval-resources/cleanup-events.jsonl` |
| Final stop and submission state | `audit/stop.json`, `audit/submission-status.json` |

GPU utilization, a zero process exit code, `Submitted`, and a small training loss
are not task-solving scores. Use actual verifier results and complete evaluation
records when comparing candidates.

## Operator tools

```bash
uv run ceb run status --run-id RUN_ID --json
uv run ceb runtime list
uv run ceb web --port 8765
uv run ceb mcp
uv run ceb audit /path/to/run
```

The web console binds to localhost by default; use an SSH tunnel for remote access.
CLI, web and MCP share operation/collection functions, but expose different subsets
of them. [MCP documentation](docs/mcp.md) lists its current tools and parameters.
Registered cloud runtimes and their integration limits are described in
[architecture](docs/architecture.md#runtime-components).

`final-eval` and `comparison` are experimental operator interfaces, not a verified
end-to-end scoring workflow. The CLI registers only SWE replay, which also has a
serving-lifetime defect.

## Development and documentation

```bash
uv sync --locked --extra mcp --extra daytona
uv run pytest
uv run ruff check src tests scripts envelope/build.py
```

Runtime tests skip when their provider, credentials, GPU or shared-filesystem
requirements are absent. A passing skipped suite is not cloud validation.
See [AGENTS.md](AGENTS.md) for module ownership and contribution conventions,
[operations](docs/operations.md) for offline source synchronization and
[troubleshooting](docs/incidents.md) for known failure patterns.

## License

[MIT](LICENSE). Third-party components retain their licenses and attribution;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
