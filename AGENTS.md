# AGENTS.md — CoEvoBench contributor map

Read this file before changing the repository. Operator workflows live in
`.agents/skills/coevobench/SKILL.md` and `docs/operations.md`. English and Chinese
user guides describe the same supported interfaces.

## Project and terminology

CoEvoBench evaluates an LLM researcher that improves a supplied 9B base model and
its initial harness within a bounded **run**. The platform evaluates the nominated
**candidate** on held-out tasks. Use: researcher, run, harness, Visible, Selection,
Test, nomination, candidate freezing, final evaluation and weight reversion.
`harness` means the target agent's scaffold, not the researcher driver.

Protocol constants and scoring follow the reference paper. The reference budget
is 24 hours, four GPUs and a $100 teacher allowance; replication is three
independent runs per researcher–benchmark pair. Engineering smoke runs may use
smaller budgets, but must not be presented as reference-protocol results.

## Architecture and ownership

```text
cli → orchestrate → bench → core → runtime → config
```

`tests/status/test_layering.py` defines the actual allowed import graph. Core
protocol/scoring logic remains bench-independent; core evaluation orchestration
may use runtime interfaces. `research/`, `status/` and `mcp/` are service consumers.
SPI definitions depend only on the standard library and configuration.

| Package | Responsibility |
|---|---|
| `cli/` | Parser and printing wrappers; operation functions are shared with other frontends |
| `orchestrate/` | Workspace provisioning, broker startup, lifecycle store, detached monitor and ordered shutdown |
| `core/protocol/` | RunSpec, frozen split sizes, protocol-asset readers and common prompts |
| `core/evaluation/` | Candidate serving, episode orchestration and evaluation aggregation |
| `core/grading/` | Scores, submission inspection, freeze and provenance |
| `core/lifecycle/` | Run registry, timing, final replay, comparison controls, results and audit data |
| `core/contamination/` | Overlap screening and persistent corpus receipts |
| `bench/` | Dataset pipelines, task adapters, episode bindings, official-grader integration and policy-sandbox drivers |
| `runtime/` | Sandbox, serving, RunHost, researcher, GPU-job and proxy implementations; resource receipts and runtime registry |
| `research/` | Broker context, op handlers, candidate judges, client forwarding and Claude recovery driver |
| `status/` | Shared status collector and web console |
| `mcp/` | Operator-side MCP tools over shared operations |
| `config/` | Typed TOML configuration and environment overrides |
| `templates/` | Initial harnesses, task briefs and frozen researcher command entries |
| `envelope/` | Allowlisted researcher-client distribution, built separately from the operator package |
| `vendor/gorilla` | Pinned BFCL upstream Git submodule |

There is one implementation per operation. CLI printers, web handlers and MCP
tools delegate; they do not reimplement launch, stop, scoring or collection.
Frontends expose different subsets of the same operations.

## Run and workspace contracts

`~/.ceb/runs/<run_id>/` contains `manifest.json`, `lifecycle.json`, `workspace/`,
`control/`, `audit/`, `select_runs/` and `eval-resources/`. Docker mounts the
workspace read-write, base weights read-only at `/base`, the broker socket and
read-only lifecycle control at `/run/ceb`. The local-process RunHost is a
cooperative mode with weaker isolation than Docker.

Workspace entries are `eval`, `sandbox`, `timer.sh`, `submit`, `track` and
`contamination`. Environment construction is `./sandbox create --from-dir DIR`;
`--image REF` starts an existing image. Build and run use the same sandbox backend.

Serving defaults are copied into editable `serve_config.json` at launch. Keep
weights, training data, custom templates and build contexts under the workspace
when they must be read by the broker. Container-local `/tmp` is not a shared mount.
`ROUND.md` and `official_protocol.json` are not automatically provisioned.

The Claude driver records successful tool results, bounded recovery and
finalization between CLI invocations. Codex/OpenCode use separate entrypoints.
DONE.md is a note, not a submission. Only a valid nomination with an
explicit `submission.json` request counts as submitted; cleanup never fabricates it.
Its digest covers nomination files and DONE.md, not all externally referenced weights.

The monitor stops on deadline, researcher exit or broker loss. Cleanup is
run-scoped and retryable; periodic failures must not terminate the broker.
Resource errors and releases append to `eval-resources/cleanup-events.jsonl`.
Preserve historical diagnostics when updating current cleanup status.

## Evaluation contracts

- Command sandboxes execute actions; they do not call model APIs. SWE/TB harnesses
  receive the platform-created `SandboxEnvironment` and frozen model transport.
- AppWorld/BFCL candidate policy code uses a separate sandbox and injected
  completion channel; official execution and judging stay operator-side.
- Selection exposes aggregate feedback and anonymous Selection IDs. Test is
  refused by development evaluation and replayed only by operator final evaluation.
- Scores come from real artifacts/verifiers. `Submitted`, GPU utilization, loss
  and an exit code of zero do not prove task success.
- Record actual limits, decoding, serving recipe, repetitions and hardware with
  comparisons. Never present incomplete evaluations as complete measurements.
- Final-eval CLI currently registers only SWE, with known serving/report defects;
  comparison and whole-bundle verification are incomplete. BFCL generic `score` is a verdict mean, not
  the protocol's weighted helper result; Telecom trials are not auto-configured.
- Corpus screening is a provenance signal, not proof that no contamination exists.
  The final audit must check the declared corpus and training recipe.
- Keep capability descriptions aligned with implementation; do not claim isolation,
  trajectory retention or provider coverage beyond what the code and tests prove.

## Development

```bash
uv sync --locked --extra mcp --extra daytona
uv run pytest
uv run ruff check src tests scripts envelope/build.py
```

Use Python 3.11+, surrounding code style and 100-column lines. Runtime/integration
tests may skip when infrastructure is absent; report skips accurately. Add
meaningful regression tests for behavior changes. Keep optional SDK imports lazy.
Provider adapters must satisfy the sandbox/runtime contracts.

Use patch-based edits for source changes.
Do not change third-party source, frozen splits or transport bytes as part of
routine documentation cleanup. New functionality belongs in the responsible
module, not a duplicate service or an empty compatibility package.

MIT covers original project code. Preserve upstream licenses, copyright and
modification notices; see `THIRD_PARTY_NOTICES.md`.

Commit messages use an imperative English summary and explain why. Append commits;
do not rewrite shared history. Public documentation should be self-contained and
use generic example hosts/paths rather than private experiment records.

The operator skill has two maintained copies: `.agents/skills/coevobench/SKILL.md`
and `~/.agents/skills/coevobench/SKILL.md`. Keep them synchronized when updating
the installed operator workflow.

## Common change locations

- Benchmark: `bench/<name>/`, frozen assets, replay binding and benchmark tests.
- Sandbox or RunHost: `runtime/<family>/`, then `runtime/registry.py`.
- Researcher CLI: `runtime/researcher/` and launch composition in `cli/run.py`.
- Broker operation: handler module in `research/`, registration and client mapping.
- Candidate final replay: `core/lifecycle/final_eval.py` plus bench-specific binding.
- Stop/recovery failure: `research/daemon.py`, `research/loop.py`,
  `runtime/run_resources.py`, `orchestrate/monitor.py` and `orchestrate/stop.py`.
- Offline deployment: `scripts/sync-server.py` and `scripts/install-source.py`.
  Rebuild the run image after changes to templates or envelope inputs; existing
  runs retain the environment and workspace provisioned at launch.
