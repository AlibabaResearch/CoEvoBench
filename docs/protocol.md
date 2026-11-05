# Protocol and code map

[中文](protocol_zh.md) · [README](../README.md)

The reference paper defines terminology, scoring and frozen constants. Code paths
below are relative to `src/coevobench/`. Changing a frozen split or evaluation
constant changes the protocol; record the version rather than mixing results.

| Protocol element | Implementation |
|---|---|
| Run tuple and researcher configuration | `core/protocol/spec.py` |
| Frozen split sizes and validation | `core/protocol/splits.py` |
| Split and image assets | `bench/<bench>/splits.json`, `image-list.json` where applicable |
| Protocol-asset readers | `core/protocol/protocol_assets.py` |
| Reported score, improvement and normalized weights | `core/grading/scoring.py` |
| Exit-status classification | `core/protocol/exit_status.py` |
| Candidate evaluation and aggregation | `core/evaluation/candidate.py`, `evaluate.py` |
| Final replay and matched controls | `core/lifecycle/final_eval.py`, `comparison.py`, bench replay bindings |
| Workspace and lifecycle | `orchestrate/provision.py`, `supervisor.py`, `monitor.py`, `stop.py` |
| Explicit submission and candidate freezing | `core/grading/submission.py`, `freeze.py` |
| Corpus screening and receipts | `core/contamination/` |

## Reference budgets and splits

A reference development run has 24 hours, four GPUs and a $100 teacher allowance.
Teacher access and spending controls must be configured by the operator. The broker
teacher forwarder does not meter USD or enforce the $100 allowance. Three independent
runs are required per researcher–benchmark pair.
Single-run and reduced-budget engineering results must be identified as such.

| Benchmark | Visible | Selection | Test |
|---|---:|---:|---:|
| SWE | 40 | 200 | 253 |
| TB | 20 | 40 | 40 |
| AppWorld | 27 | 51 | 51 |
| Telecom | 23 | 46 | 45 |
| BFCL | 235 | 470 | 560 |

BFCL counts scoring records, including memory questions. Telecom's reference Test
protocol requires four trials per task; final-eval currently defaults to `k=1` and
does not register Telecom replay. The anonymous workspace pool contains Selection IDs only.
Publicly downloadable material is still subject to the protected-task boundary.

## Development feedback

Use `--split visible` or `--split selection`. `--tasks` requires Selection and a
comma-separated list of IDs from `./eval --list-tasks`; omit it for the whole split.
`--n` selects a prefix, `--k` controls repetitions, and `--workers` controls concurrency.
The client rejects Test. Visible records include scores and exit categories;
Selection returns aggregate feedback while its detailed records stay operator-side.

SWE scores resolved instances; TB averages verifier rewards in [0,1]. AppWorld
uses its official task-goal evaluator. Telecom uses mean single-trial success.
BFCL's protocol metric and `aggregate_score` helper combine multi-turn and memory
with weights 0.6/0.4. The generic evaluation engine currently averages verdicts
without calling that helper, so its `score` is not the weighted domain metric.
Small subsets are diagnostic and do not replace complete domain scoring.

`Submitted` describes an episode's termination, not successful verification.
Infrastructure failures and incomplete measurements must remain distinguishable
from measured task failure. Final reports use the scoring and reliability rules,
including weight reversion to the base model with the evolved harness when required.

## Formal presets and final evaluation

The workspace flag is `--formal`. The shared broker helper **overwrites** supplied
values with SWE constants: k=1, temperature 0, 80 steps, 900 seconds, 5120 output
tokens and 32768 context. It does not reject conflicting flags. AppWorld subsequently
uses its own formal settings (50 steps, 3000 tokens, temperature 0, seed 100, no
task wall cap, 600s request timeout, 65536 context). BFCL rejects the helper's
nonzero wall cap; its output cap also cannot exceed 4096.

Without `--formal`, defaults depend on the path:

| Candidate path | Step / wall / output defaults | Context |
|---|---|---|
| SWE/TB broker | 80 / 900s / 5120 | Workspace recipe; fallback 32768 |
| TB adapter reference | 120 / 600s / 5120 | 32768 |
| AppWorld | Harness config, then explicit request overrides | Recipe; fallback 65536 |
| BFCL | Harness config: initially 20 / no wall cap / 4096 | Recipe; fallback 65536 |
| Telecom | Broker passes output cap 5120, overriding harness initial 1024 | Recipe-dependent |

Use explicit limits for a TB reference-style development self-check:

```bash
./eval --split visible --model ./base --workers 4 \
  --step-limit 120 --wall-cap 600 --max-tokens 5120 \
  --max-model-len 32768 --temperature 0
```

This is development feedback. Operator `final-eval` and `comparison` are experimental:
CLI replay registration, serving lifetime, bundle integrity and control staging have
open gaps. `--revert-weights` expresses
base-weight replay intent but does not make those paths complete. Keep actual
serving/evaluation settings with every measurement.

## Submission

The researcher writes `out/<bench>_final/` with the harness, `model_path.txt`,
`meta.json`, `serve_config.json` and reproducibility material. `out/DONE.md` records
the conclusion; `./submit --bench BENCH` creates the explicit request fingerprint.
Edits to those hashed files invalidate the request; checkpoint files outside the
nomination are not fully covered. Shutdown verifies the actual request and
reports incomplete/invalid submissions without creating missing artifacts.

`ceb calibrate swe` checks that gold patches pass and empty patches fail. Calibration
supports grader reliability; it is separate from candidate capability measurement.
