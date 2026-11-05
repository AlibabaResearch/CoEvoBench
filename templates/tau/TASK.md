# Telecom — model and harness co-evolution

This run evaluates **τ²-bench Telecom**, benchmark key `tau`. Improve the supplied
9B base model, the initial text ReAct harness, or both. Check `./timer.sh status`;
the standard provisioner does not generate `ROUND.md`.

## Workspace and interfaces

Edit `harness/src/tauharness/`: prompts, textual action parsing, history, retries
and agent logic. Keep `agent_factory(tools, domain_policy, **kwargs)`, returning an
agent with `get_init_state()` and
`generate_next_message(message, state) -> (assistant_message, new_state)`.
Use the platform-supplied candidate model endpoint. The telecom environment,
simulator configuration, official tasks and reward computation belong to the operator.

`base/` is read-only. Keep persistent data, templates and checkpoints under the
workspace; container-local `/tmp` is not visible to operator serving. Initial
serving settings are in editable `serve_config.json`. Harness provenance is in
`harness/SOURCE.md`.

## Evaluation

The protocol uses **23 Visible / 46 Selection / 45 Test** tasks. The anonymous pool
contains only 46 Selection IDs. Visible results provide diagnostic scores and exit
categories; full trajectory retention depends on the path. Selection returns
aggregate feedback, and Test is operator-only.

```bash
./eval --split visible --n 2 --model ./base
./eval --split selection --model work/weights/v1 --k 4
./eval --list-tasks
./eval --split selection --tasks A-xxxxxxxx --model work/weights/v1 --k 4
```

The domain score is mean single-trial success; the reference Test protocol requires
four trials per task. The operator final-eval path does not currently select these
trials or register Telecom replay automatically.
Match repetitions, decoding and serving parameters when comparing candidates.
The initial text agent uses a 1024-token output cap, but the broker candidate path
passes 5120 unless `--max-tokens` is explicit. Environment/driver limits also apply.
A simulator model is fixed by the operator, not chosen by the
researcher. The shared `--formal` helper applies SWE constants rather than a
Telecom-specific final preset.

## Data and nomination

Permitted public data and assigned-model rollouts may be used, but never target
protected Selection/Test tasks, solutions, private tests or trajectories. Teacher
use requires explicit operator authorization/configuration. Keep sources,
transformations, training commands, hyperparameters and dependency versions.

Save intermediate candidates under `out/<label>/`. Final `out/tau_final/` contains
the full `harness/`, `model_path.txt`, `meta.json`, `serve_config.json` and any
training recipe/provenance. Use `/base` if weights are unchanged or a full merged
checkpoint under `/workspace/...`. If trained, record a same-harness trained-vs-base
control and negative results as well as improvements.

Before the deadline, write `out/DONE.md` and run `./submit --bench tau`. A note alone
does not complete submission. Subsequent edits to hashed nomination files or DONE.md
invalidate the request; checkpoint bytes outside the nomination are not fully hashed.
Preserve the checkpoint for operator verification.
Do not modify frozen entries, evaluator state, private split definitions or other
runs. Development tools are described in [RESEARCH_TOOLS.md](RESEARCH_TOOLS.md).
