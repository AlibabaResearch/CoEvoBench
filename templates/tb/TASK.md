# TB-Lite — model and harness co-evolution

You are the researcher for one **TB** development run. Improve the supplied 9B
base model, its initial harness, or both within the assigned budget. Check
`./timer.sh status`; the original deadline applies across sessions and recovery.

## Workspace and initial harness

- `base/` is the read-only base checkpoint; in Docker runs it resolves to `/base`.
- `harness/src/minisweagent/agents/default.py` implements the initial Terminus-style
  loop: structured JSON command batches, real tmux keystrokes and pane observations.
- The driver imports `DefaultAgent` and `LitellmTextbasedModel`, injects the model
  and execution environment, and calls `run(task=...)`.
- Keep `models/_transport.py` unchanged. Every evaluated model answer must use the
  platform-supplied transport; commands execute through the supplied environment.
- `serve_config.json` contains the initial operator-selected serving recipe. It is
  editable and is loaded by `./eval --model ...`.
- Use `work/` for data, scripts, checkpoints and build contexts; keep persistent
  work under `/workspace`. Container-local `/tmp` is not visible to the operator.

See `harness/src/minisweagent/EDITABLE.md` and [RESEARCH_TOOLS.md](RESEARCH_TOOLS.md).
Installed package versions and available disks/GPUs depend on this run's image.
Inspect them rather than assuming a fixed package list. `ROUND.md` and
`official_protocol.json` are not automatically created.

## Development evaluation

The protocol contains 20 Visible, 40 Selection and 40 Test tasks. Visible feedback
includes per-task scores and exit categories under `out/eval/`. Full trajectories
are not guaranteed by this path. The anonymous pool contains Selection IDs only;
Selection returns aggregate feedback, and Test is unavailable to the researcher.
Materialized public views are supplied in `tasks_visible/tb/`; extra files do not
authorize use of protected task material.

```bash
./eval --split visible --model ./base --workers 4
./eval --split visible --model work/weights/v1 --n 2 --workers 2
./eval --split selection --model work/weights/v1
./eval --list-tasks
./eval --split selection --tasks A-xxxxxxxx,A-yyyyyyyy --model work/weights/v1 --k 2
./eval --split visible --model work/weights/v1 --serve-config work/recipe.json
```

Use the spelling `selection`. `--tasks` requires that split and actual IDs from the
pool; omit `--tasks` for the full split. `--n` selects a prefix. TB averages verifier
reward in [0,1]; partial credit contributes to the mean. `Submitted` is an exit
category, not successful verification. Record the score, sample count and settings.

`./eval` manages the candidate model server. Do not run training or a second model
server on the same GPUs at the same time. Explicit parser/context flags override
the serving recipe; `--serve-config` replaces the default JSON file. Recipe keys:
`tensor_parallel`, `data_parallel`, `gpu_mem_util`, `max_len`, `max_num_seqs`,
`enable_prefix_caching`, `enforce_eager`, `tool_call_parser`, `reasoning_parser`,
`chat_template` and `extra_args`. Template paths are relative to the recipe file.

For a development self-check at the current TB adapter limits, use explicit arguments:

```bash
./eval --split visible --model work/weights/v1 --workers 4 \
  --wall-cap 600 --step-limit 120 --max-tokens 5120 \
  --max-model-len 32768 --temperature 0
```

Without explicit flags, the SWE/TB broker uses 80 steps, 900 seconds and 5120 output
tokens, with context from the workspace recipe (fallback 32768). These differ from
the TB adapter reference above. `--formal` overwrites flags with SWE constants;
it does not select TB-specific limits. Development evaluation is feedback, not Test evaluation.

## Data, training and model files

You may acquire permitted public data, construct tasks and verifiers, replay
permitted human-written solutions and generate rollouts with the assigned model.
Choose data independently of protected Selection/Test instances. Public availability
does not authorize their solutions, private tests or trajectories. Record sources,
revisions, transformations and dependencies. Teacher use requires explicit
operator authorization and configuration; a missing optional note is not permission.

For private-corpus screening before training:

```bash
./contamination check --input work/train.jsonl --manifest out/tb_final/train/corpus.json
./track -- python work/train.py
```

The screen accepts at least 100 distinct text records per batch, at most 32 files
and 1 GiB, with five distinct corpus versions per run. Small corpora require
operator handling, not duplicate padding. Preserve the exact checked bytes.
`no_matches` is not proof of no contamination; document any review findings.

The supplied Qwen3.5-9B checkpoint uses a multimodal architecture. Preserve model
configuration, tokenizer, processor files and architecture when merging adapters.
Text-only serving can disable image/video inputs. Final replay requires full
weights, not an adapter-only directory. Keep checkpoints accessible under the
workspace and avoid retaining unnecessary optimizer-state copies.

## Nomination and completion

```text
out/tb_final/
  harness/                 complete candidate harness
  model_path.txt           /workspace/work/weights/v1, or /base when unchanged
  meta.json                bench, changes, evaluation settings and honest results
  serve_config.json        exact serving recipe
  train/                   recipe/provenance/dependencies for any training
```

Intermediate candidates may live under `out/<label>/`. The workspace eval path
uses the current workspace harness; restore the intended harness before comparing
weights. If training was performed, report a same-harness trained-vs-base control
and retain reproducible commands/data. Negative results are valid observations.

Finish the nomination, write `out/DONE.md`, then run `./submit --bench tb`.
DONE.md alone is not completion. Later edits to hashed nomination files or DONE.md
invalidate the request; externally referenced checkpoint bytes are not all hashed.
Preserve the checkpoint for operator validation and separate final evaluation;
submission does not start Test replay. Missing files are recorded as incomplete.
Begin finalization before the budget expires.

Do not modify frozen command entries, access evaluator/provider credentials, read
private split definitions or bypass the supplied evaluation transport/environment.
The task is to improve the candidate, not the platform's grading machinery.
