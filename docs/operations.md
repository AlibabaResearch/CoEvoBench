# Operations

[中文](operations_zh.md) · [README](../README.md)

## Dependencies

Use the repository checkout and the tracked lockfile. The **operator environment**
executes the broker and evaluation drivers; the run image is separate. The `benches`
extra installs dataset tooling, not every benchmark engine or the GPU stack.

```bash
uv sync --locked --extra mcp --extra benches
uv run ceb doctor
```

- SWE/TB require Docker task images and their task/verifier dependencies.
- AppWorld needs the upstream AppWorld package and matching data/environment
  version; its candidate policy also uses a sandbox.
- BFCL uses `vendor/gorilla/berkeley-function-call-leaderboard`. Its
  `scripts/build-bfcl-image.py` reads installed package versions from the invoking
  environment; install the pinned upstream dependencies plus Transformers,
  soundfile and Torch there first. Keep this environment separate from vLLM when
  dependency constraints conflict, and retain source/image inventory checks.
- τ²-bench is the [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
  project. The unrelated PyPI package named `tau2` is not this benchmark. Use the
  Python version required by the upstream revision (commonly Python 3.12+).
- Provider SDKs and credentials are required only for the providers in use.

For local SWE/TB serving, install `litellm`, `pyyaml`, `platformdirs`, `rich` and
`tenacity` in the broker environment. vLLM can live in an isolated serving venv:
the default runtime looks for a `vllm` executable beside the broker's Python, so
place a symlink to the isolated executable there, or pass `venv_bin` when composing
`VllmServeRuntime` in Python. Installing vLLM only in the researcher image is insufficient.
Record the installed versions in each environment. Manually added benchmark/ML
dependencies are outside `uv.lock`; `uv sync` may remove them. Use the prepared
environment directly or `uv run --no-sync` for such deployments.

`Dockerfile.run` installs the researcher CLI and ML dependencies separately.
The image pins vLLM/Transformers and researcher CLI versions, while some training
dependencies remain ranged. Archive its identity and `/opt/researcher-requirements.txt`.
The default Dockerfile installs Claude Code, Codex and OpenCode V2. The tracked
`uv.lock` locks the operator/dev dependency resolution; CI uses `uv sync --locked`.
As reviewed on 2026-11-05, the image declares vLLM 0.32.0 and Transformers 5.19.0.
Dependency resolution and offline checks passed, but the updated Docker/GPU images
have not been built and accepted on a live GPU deployment.

## Data and images

```bash
uv run ceb data fetch tb
uv run ceb data materialize tb
uv run ceb data build-images tb
```

`fetch` caches the frozen split definition. `materialize` fetches/loads task content
and creates public views plus operator judge records. Definitions are stored under
`src/coevobench/bench/<bench>/`; acquired content lives under `~/.ceb/cache/`.

SWE images use `swebench/sweb.eval.x86_64.<instance>` references. `build-images tb`
builds each public task's `environment/` on local Docker, records its image ID and
context hash in the task cache, and resumes previously successful builds. Use
`--tasks ID,ID --timeout 1800` for a subset. Logs live in `~/.ceb/cache/tb-image-builds/`.
Alternatively, configure a task-keyed mirror and skip local building.
An image manifest is not proof that an image is reachable from your deployment.

`[[eval_images.mirrors]]` supports prefix rewrites, `tag_map` and separate
`env:VARIABLE` credential references. `tag_source="instance_id"` or `"task_id"`
selects a registry keyed by instance/task ID. See `ceb.toml.example`.

TB materialization and workspace provisioning filter public views against Visible
IDs and remove generated non-Visible JSON files left by older versions. Existing
active workspaces are not changed until explicitly re-provisioned.

For AppWorld, install the pinned public upstream source and build its policy image:

```bash
uv run python scripts/setup-appworld.py
docker build -f deploy/appworld-policy.Dockerfile -t coevobench-appworld-policy:latest deploy
```

The setup reads dependencies from that pinned upstream revision and records the
installed versions. Modal component preparation uses `scripts/prepare-modal-appworld.py`;
`--check --base-dir PATH` validates local image inputs and weights without allocation.
This prepares provider assets, not a complete remote researcher deployment.

## Serving

The operator's `[serve.defaults]` becomes the editable workspace recipe at launch.
`max_model_len` in TOML is written as `max_len` in JSON. Existing workspace files
are preserved. A candidate's submitted recipe travels into final replay.

For text-only Qwen3.5-9B on two A10s, the following was validated with **vLLM 0.31.0**:
managed serving, four concurrent approximately 24k-input-token requests and a bounded
Visible episode. This does not validate the newer image's 0.32.0/5.19.0 stack.

```toml
[sandbox]
runtime = "docker"

[serve.defaults]
tensor_parallel = 2
data_parallel = 1
gpu_mem_util = 0.9
max_model_len = 32768
max_num_seqs = 32
enable_prefix_caching = true
enforce_eager = true
extra_args = ["--limit-mm-per-prompt", '{"image":0,"video":0}']
```

The JSON equivalent is `examples/serve/qwen35-9b-2xa10.json`. Eager mode avoids
the observed torch.compile startup error; disabling image/video inputs avoids
multimodal encoder allocation. Preserve the checkpoint's architecture and required
processor/tokenizer files. Keep merged weights under `/workspace/work/weights/`;
operator serving cannot read files stored only in the container's `/tmp`.

Inside the researcher workspace:

```bash
./eval --split visible --model ./base --workers 4
./eval --split visible --model work/weights/v1 --serve-config serve_config.json
./eval --split selection --model work/weights/v1 --tasks A-xxxxxxxx --k 2
./track -- python work/train.py
```

`--serve-config` replaces the default file; explicit parser/context flags override
the recipe. Custom template paths are recipe-relative. `eval.plan` computes the
client wait budget from the actual workload. A disconnected client can leave an
evaluation running: inspect records before retrying. Serving startup failures
include a diagnosis, effective parameters and the private log basename.

## Run monitoring and completion

Use `ceb run status --run-id RUN_ID --json`. The run directory is
`~/.ceb/runs/RUN_ID/`; inspect `workspace/out/PLAN.md` if the researcher maintains
it, training logs/checkpoints, Visible records and Selection summaries.

Distinguish these states:

- A training script exists: training has not necessarily run.
- A checkpoint and completed training log exist: training completed, but improvement
  still requires matched evaluation.
- `Submitted` is an episode exit category, not a verifier success.
- `phase=done` is a completed shutdown, not a confirmed nomination.

Claude recovery is bounded; `workspace/out/researcher-state.json` and
`workspace/out/researcher-loop.jsonl` record sessions and transitions. Finalization
and recovery decisions occur between CLI invocations. Codex/OpenCode do not use
this Claude continuation loop. A valid final nomination
contains `harness/`, `model_path.txt`, `meta.json`, `serve_config.json`, provenance
and any training recipe, followed by `out/DONE.md` and `./submit --bench BENCH`.
Submission hashes the nomination files and DONE.md, not all bytes of a checkpoint
outside that directory. Operator final evaluation is a separate, currently incomplete
workflow.

Shutdown uses the same path for deadline, researcher exit, broker loss and operator
stop. Archival attempts to save Docker container logs and `.claude/projects` /
`.codex/sessions` in `audit/researcher/`. Local and OpenCode HOME sessions are not
copied by this helper. OpenCode tool events remain in `workspace/out/opencode-tools.jsonl`.
Cleanup only targets
persisted ownership receipts and exact owner labels. Periodic cleanup errors are
caught and retried. Historical details remain in `eval-resources/cleanup-events.jsonl`.

```bash
uv run ceb run stop RUN_ID
```

Repeat this command if cleanup remains pending. Do not use global Docker pruning
or broad process-name killing for a single run's cleanup.

The status collector's `container` field still uses Docker inspection, including
for local runs. Its audit/guard counts read only `audit/bash.log`; zero counts do
not summarize JSON tool events or establish absence of violations.

## Offline source deployment

When a server cannot access the repository, use the checksum-manifest synchronizer:

```bash
git submodule update --init
# Stage new source files before packaging.
python3 scripts/sync-server.py operator@server.example /srv/CoEvoBench
ssh operator@server.example \
  'cd /srv/CoEvoBench && docker build -f Dockerfile.run -t coevobench-run:latest .'
```

It packs Git-indexed paths and pinned submodules as tar.zst, transfers via SCP and
verifies installed hashes. It removes only stale files managed by the prior source
manifest; local configuration, virtual environments and run artifacts are preserved.
The server needs Python 3 and zstd. Avoid overlay extraction that leaves deleted
modules behind. Rebuild after changes to templates or envelope inputs. Existing
runs retain their provisioned workspace and image; deploy to an idle checkout when
changing code that an active broker may import.

## Remote components

Use `ceb runtime list` and the [component map](architecture.md#runtime-components).
RunHost, sandbox and serving are independent interfaces, but some composition
paths are provider-specific. Remote workspace synchronization, base weights,
broker reachability, persistent service entrypoints and archival must all be
configured for a complete run. Validate a short run before assigning a long budget.
