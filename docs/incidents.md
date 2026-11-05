# Troubleshooting patterns

[中文](incidents_zh.md) · [Operations](operations.md)

## TTL cleanup races with Docker auto-removal

**Symptom:** `docker rm -f` reports `removal ... already in progress` during TTL
expiry. Treating this as a fatal maintenance error can terminate the broker.

**Handling:** the shared Docker removal helper waits with a bounded retry. Failed
receipts remain pending; broker maintenance catches errors and backs off. Consult
`eval-resources/cleanup-events.jsonl` for the resource, timestamp and error.
`cleanup-status` is current state, not historical evidence. Retry `ceb run stop`
if final shutdown still has pending resources.

## vLLM fails before readiness

Inspect `audit/serve/` and the effective recipe in the client error. Common causes:

- Insufficient KV cache: reduce context/concurrency, disable unused multimodal
  inputs, or adjust the memory fraction within the available GPU budget.
- Multimodal encoder OOM: text-only tasks can set image/video limits to zero.
- torch.compile `NoneType.size` failure: try `enforce_eager=true` for the affected
  model/runtime combination.
- Model path not found: keep checkpoints and custom templates in the workspace,
  not container-local `/tmp`; preserve materialized weight files and processor assets.
- Missing compiler/runtime dependency: install it in the operator serving
  environment, not only in the researcher image.

The [two-A10 recipe](../examples/serve/qwen35-9b-2xa10.json) was tested with vLLM
0.31.0, not the current image's 0.32.0 stack. A process allocating GPU memory is not
proof that readiness or evaluation succeeded. Check actual completion and verifier records.

## Researcher repeats without useful work

Inspect `out/researcher-state.json` and `out/researcher-loop.jsonl`. The Claude
driver counts novel successful tool results, not log volume or exit code alone.
Three consecutive no-progress rounds trigger a new session; context-length errors
and malformed calls trigger recovery after the invocation returns. Recovery and
finalization decisions occur between invocations; the 30-minute/DONE.md check does
not interrupt a continuing invocation. Both mechanisms are bounded.
Workspace artifacts and the original deadline survive session changes.

## Submission is incomplete after shutdown

DONE.md does not replace `./submit`. Check `audit/submission-status.json` for missing
files or a stale request fingerprint. Shutdown records the actual state and does
not manufacture a nomination. `phase=done` only means resources were stopped.

## Stale source or run image

Overlay copies can leave deleted source files behind. Use the checksum-manifest
synchronizer, then rebuild the envelope after client/template changes. A directory
bind for `/run/ceb` makes atomically replaced lifecycle files visible; a single-file
bind can leave the container reading an old inode.

## Sandbox timeout kills the sandbox itself

With an init process, the keepalive is its child. A process sweep must preserve both
init and the keepalive, and must not reuse a stale process list from an earlier exec.
The Docker sandbox adapter handles process-tree timeout cleanup explicitly.

## Frozen bundle or score looks inconsistent

`freeze.json` is excluded from the tree hash to avoid hashing the manifest into
itself. `evaluate` checks only the submission tree before replay, not all copied
weights or post-replay state. Preserve transport and independently check complete
artifact hashes before/after replay.
Scoring reliability exempts a healthy all-Submitted distribution, but Submitted
still requires a verifier score. Incomplete evaluations and task failure are
different outcomes; review both score and exit-status distribution.
