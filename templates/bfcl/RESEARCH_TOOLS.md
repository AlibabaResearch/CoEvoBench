# Research tools

You may independently acquire, generate and transform training data, use other
training tasks, write verifiers and construct task environments. Provided datasets
and images are starting resources, not a whitelist. Selection/Test restrictions still
apply. These development tools are separate from official evaluation.

## Downloading data and code

Use `hf`, `curl`, Python and Git freely for permitted training materials. Record
source URLs, revisions, checksums and data transformations in your training recipe.
Downloading task descriptions does not supply their executable environments. Selection/Test
solutions remain excluded even when publicly downloadable; data freedom does not
authorize using other checkpoints or external model APIs outside the separate
model policy.

Use your workspace, e.g. `work/data/`, for downloads and build contexts. The broker
only reads and writes files under this workspace, including resolved symlinks.
Host connectivity varies by run; judge sandboxes are offline by default — download
materials into your workspace (or a network-enabled sandbox, see `--network` below)
and transfer from there. Connectivity is not guaranteed.

## Task environments and interactive sandboxes

`./sandbox` is the platform client. It uses the experiment's broker, which holds
the sandbox provider credentials. Do not modify the client.

Start an environment from a build context — a directory containing a `Dockerfile`
at its root plus exactly the files the build needs:

```dockerfile
FROM python:3.11-slim
WORKDIR /task
COPY . /task/
RUN python -m pip install --no-cache-dir pytest
CMD ["sleep", "infinity"]
```

```bash
./sandbox create --from-dir work/my-task --cpu 2 --memory 4 --ttl 1800
./sandbox exec SESSION_ID 'python /task/verify.py' --cwd /task --timeout 120
./sandbox upload SESSION_ID work/input.bin /task/input.bin
./sandbox download SESSION_ID work/reward.json /task/reward.json
./sandbox status SESSION_ID
./sandbox export SESSION_ID work/environment.jsonl
./sandbox close SESSION_ID
```

The create call builds and runs on the same sandbox provider; there is no separate
researcher build job or push step. Put local `COPY`/`ADD` inputs under the context
root. Docker/Modal use directory contexts; Daytona's SDK collects the referenced
inputs. Ignore-file handling and cache reuse depend on the backend. Keep credentials,
checkpoints and unrelated files out. The context must be inside your workspace;
escaping symlinks are rejected. Docker build errors include a log tail; other
providers expose their SDK's diagnostics.

`--image REF` uses an existing image reachable by the provider. `--network` requests
network access (off by default); connectivity still depends on the deployment.
`--build-timeout` defaults to 1800 seconds: Docker enforces it and Daytona passes
it to SDK creation, but Modal currently does not enforce this argument. Use separate
sessions for independent environments; cache reuse is provider-dependent.

For both transfers, argument order is `SESSION_ID LOCAL_PATH REMOTE_PATH`.
Upload preserves binary bytes; broker download currently reads/writes text. Encode
binary downloads as base64 text in the sandbox, then decode locally. Remote paths
must be absolute. Each exec is a fresh shell:
use `--cwd`, explicit environment exports or tmux for persistent shell state.

CLI resource defaults are 2 CPU, 4 GiB, TTL 1800 seconds, readiness timeout 120
seconds; raw Python requests without `mem_gib` default to 8 GiB. These are development
defaults, not final evaluation constraints. Service quotas and your declared resource allocation
still apply. Close environments promptly; TTL provides fallback cleanup. Resource
requests, commands, exit codes and transfer metadata are recorded per experiment.
`export` requires a live registered session, so export before closing. Command
output text, model messages and rewards must be recorded by your rollout code.

You can write/run your own verifier, upload a third-party training task or build
synthetic tasks this way. Use Visible `./eval` for evaluation feedback; do not
retrieve protected evaluation environments through development tools. The broker cannot attach to
arbitrary sandbox IDs or inspect another experiment's environments. Do not use
builds or downloads to retrieve Selection/Test materials. This is a cooperative
benchmark boundary, not a substitute for OS/container isolation from other host
users.

## Rollout and training

A Python client is available:

```python
from coevobench.research.client import Client
client = Client()
session = client.call("sandbox.create", image="python:3.11-slim", ttl_s=300)
try:
    result = client.call("sandbox.exec", sandbox_id=session["sandbox_id"],
                         command="python -V", timeout_s=30)
    print(result["text"])
finally:
    client.call("sandbox.close", sandbox_id=session["sandbox_id"])
```

For custom training tasks, write a workspace-side rollout driver using the Python
client above. Call the permitted candidate model from that driver, send commands
to the sandbox, and record messages, rewards and cleanup yourself. The BFCL
workspace does not include a generic rollout script. You control prompts, steps,
sampling, retries, reward design and training algorithm.

Visible evaluation writes scores and exit categories under `out/eval/`; complete
episode trajectories are not guaranteed. Selection returns aggregate feedback.
Use `--split selection --tasks A-...` for a subset; the pool contains Selection
only, never Test. Self-evaluation and final evaluation are separate operations.

Use `./track -- python work/train.py` around the actual training process. Keep
broker-readable artifacts under `/workspace`, not container-local `/tmp`.
Finish with `out/DONE.md` and `./submit --bench bfcl`.

## Corpus provenance

The broker's private corpus-screening service is currently wired for SWE/TB, not
BFCL. Preserve data sources, revisions, transformations and exact training inputs
for operator review; an installed client does not imply this service is configured
for the benchmark. Protected Selection/Test materials remain excluded.
