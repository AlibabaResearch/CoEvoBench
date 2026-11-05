#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
command -v uv >/dev/null
test -f vendor/gorilla/berkeley-function-call-leaderboard/pyproject.toml

# BFCL pins NumPy 1.x; current vLLM requires NumPy 2.x through OpenCV.
uv sync --extra mcp --extra benches
uv pip install --python .venv/bin/python --torch-backend cpu \
  -e vendor/gorilla/berkeley-function-call-leaderboard soundfile
if [ ! -x .venv-serve/bin/python ]; then
  uv venv .venv-serve
fi
uv pip install --python .venv-serve/bin/python 'vllm==0.32.0'
ln -sfn ../../.venv-serve/bin/vllm .venv/bin/vllm
uv pip check --python .venv/bin/python
uv pip check --python .venv-serve/bin/python
