# SWE-bench — model and harness co-evolution

You are the researcher for one **SWE** development run. Improve the supplied 9B
base model, its initial harness, or both within the assigned budget. Check
`./timer.sh status`; recovery retains the workspace and original deadline.

## Workspace and editable harness

- `base/` is read-only and resolves to `/base` in Docker runs.
- `harness/src/minisweagent/agents/default.py` adapts the initial SWE-agent ACI
  loop. Prompts and shell commands are in `agents/aci/`; shell state persists via
  sandbox files across otherwise fresh exec shells.
- The platform imports `DefaultAgent` and `LitellmTextbasedModel`, provides the
  model/environment, and calls `run(task=...)`.
- Keep `models/_transport.py` unchanged and use the provided environment for
  evaluated commands. Other harness logic is editable.
- `serve_config.json` is the editable initial serving recipe.
- Keep data, scripts, templates, build contexts and checkpoints under `/workspace`,
  e.g. `work/`. Container-local `/tmp` is not a broker-accessible shared directory.

Read `harness/src/minisweagent/EDITABLE.md` and [RESEARCH_TOOLS.md](RESEARCH_TOOLS.md).
Inspect installed versions and assigned GPUs. Optional `ROUND.md` and
`official_protocol.json` files are not automatically provisioned.

## Evaluation and feedback

The frozen subset contains 40 Visible, 200 Selection and 253 Test instances.
Visible records in `out/eval/` provide scores and exit categories; this path does
not guarantee full episode trajectories. `anonymous_pool_swe.txt` contains only
Selection IDs. Selection returns aggregate feedback, and Test is operator-only.
Materialized public views are under `tasks_visible/swe/`; a file appearing there
does not authorize protected materials accidentally included in a cache.

```bash
./eval --split visible --model ./base --workers 4
./eval --split visible --model work/weights/v1 --n 2
./eval --split selection --model work/weights/v1 --workers 4
./eval --list-tasks
./eval --split selection --tasks A-xxxxxxxx --model work/weights/v1 --k 2
./eval --split visible --model work/weights/v1 --serve-config work/recipe.json
./eval --formal --split visible --model work/weights/v1 --workers 4
```

Use `selection`, not an abbreviated split name. `--tasks` takes actual anonymous
Selection IDs; omit it for the full split. `--n` selects a prefix. SWE scores the
official tests against the candidate's real diff. `Submitted` is not proof of
resolution. Preserve negative results, settings and sample counts.

The current shared `--formal` preset uses SWE's adapter limits: k=1,
temperature=0, 80 steps, 900 seconds, 5120 output tokens and context 32768. It
overwrites conflicting flags. Without it, broker step/wall/output defaults are
80/900s/5120, while context comes from the workspace recipe (fallback 32768).
Ordinary development evaluation accepts explicit overrides. Final Test replay
requires a separate operator workflow; it is not triggered by submission.

## Serving and training

`./eval --model ...` starts and stops the model server for the evaluation. Serialize
training, evaluation and self-managed rollout serving on the assigned GPUs.
`--serve-config` replaces the default recipe; explicit parser/context flags take
precedence. Keys include `tensor_parallel`, `data_parallel`, `gpu_mem_util`,
`max_len`, `max_num_seqs`, `enable_prefix_caching`, `enforce_eager`, parser names,
`chat_template` and `extra_args`. Custom templates are recipe-relative.

Choose training materials independently of protected Selection/Test instances.
Permitted public repositories, other tasks, human-written solutions, custom
verifiers and assigned-model rollouts may be used. Do not retrieve protected
solutions, private tests or trajectories even if publicly downloadable. Record
source URLs, revisions, transformations and dependency versions. Teacher access
requires explicit operator authorization/configuration.

```bash
./contamination check --input work/train.jsonl --manifest out/swe_final/train/corpus.json
./track -- python work/train.py
```

Screen the final corpus before training and retain its exact bytes. The private
screen has a five-version quota, at least 100 distinct text records per batch,
at most 32 files and 1 GiB. Do not pad a small corpus with duplicates; arrange
operator handling. `no_matches` is not proof of no contamination.

Preserve the base model architecture, tokenizer and processor assets. Merge LoRA
adapters into full weights for final replay and keep them under the workspace.
Track actual training commands, data provenance, hyperparameters and environment.
Use `./track -- COMMAND ...` around the real process, not a shell that only starts
a background job and immediately exits.

## Nomination

```text
out/swe_final/
  harness/
  model_path.txt           /workspace/work/weights/v1, or /base when unchanged
  meta.json                bench, changes, settings and results
  serve_config.json
  train/                   recipe, data provenance and dependencies if trained
```

Keep intermediate candidates under `out/<label>/`. The workspace eval path loads
the current workspace harness, so restore the intended harness when comparing
candidates. For trained models include a same-harness trained-vs-base control.

Before the deadline, assemble the nomination, write `out/DONE.md` and run
`./submit --bench swe`. DONE.md is not a submission request. Edits to hashed
nomination files or DONE.md invalidate it; externally referenced checkpoint bytes
are not all hashed by this request. Preserve the checkpoint for operator validation
and separate final evaluation. Incomplete nominations stay incomplete.

Do not change frozen workspace entries or transport, inspect evaluator/provider
credentials, access private split definitions or bypass the evaluation environment.
