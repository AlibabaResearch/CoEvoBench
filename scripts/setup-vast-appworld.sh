#!/usr/bin/env bash
# Run on the rented Vast Ubuntu/Debian instance, as the operator account.
# CUDA comes from the provider. No Docker daemon or systemd is needed here.
set -euo pipefail
test "$(id -u)" = 0 || { echo 'Run this setup as root.' >&2; exit 1; }
CEB_ROOT="${CEB_VAST_ROOT:-/workspace/coevobench}"
mkdir -p "$CEB_ROOT"
if ! id cebtrain >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash cebtrain
fi
mkdir -p "$CEB_ROOT/jobs" "$CEB_ROOT/base" "$CEB_ROOT/.operator"
chmod 755 "$CEB_ROOT" "$CEB_ROOT/jobs" "$CEB_ROOT/base"
chmod 700 "$CEB_ROOT/.operator"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install uv==0.13.3
fi
apt-get update
apt-get install -y rsync build-essential git curl procps util-linux
uv venv --python 3.11 --allow-existing "$CEB_ROOT/.venv-serve"
uv pip install --python "$CEB_ROOT/.venv-serve/bin/python" 'vllm==0.32.0'
uv venv --python 3.11 --allow-existing "$CEB_ROOT/.venv-train"
uv pip install --python "$CEB_ROOT/.venv-train/bin/python" \
  'torch==2.13.0' 'transformers==5.19.0' 'peft>=0.12' 'trl>=0.9' \
  'accelerate>=0.34' 'datasets>=2.20'
uv pip check --python "$CEB_ROOT/.venv-serve/bin/python"
uv pip check --python "$CEB_ROOT/.venv-train/bin/python"
uv pip freeze --python "$CEB_ROOT/.venv-train/bin/python" > "$CEB_ROOT/train-installed.txt"
nvidia-smi
"$CEB_ROOT/.venv-serve/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() >= 4, 'The configured experiment requires four GPUs'
for i in range(4):
    assert torch.ones(1, device=f'cuda:{i}').cpu().item() == 1
print('CUDA operations passed on four GPUs; model-load and collective tests still required.')
PY
