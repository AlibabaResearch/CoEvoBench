#!/usr/bin/env bash
# Tested host family: Alibaba Cloud Linux 4, x86_64, systemd.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo "Run this host installation script as root." >&2
  exit 1
fi
. /etc/os-release
if [ "$ID" != alinux ] || [ "$(uname -m)" != x86_64 ]; then
  echo "This script targets Alibaba Cloud Linux x86_64." >&2
  exit 1
fi
dnf install -y moby-engine docker-compose-plugin
groupadd -f --system docker
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  -o /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/etc/docker/daemon.json')
config = json.loads(path.read_text())
config.setdefault('log-driver', 'local')
config.setdefault('log-opts', {'max-size': '20m', 'max-file': '5'})
config.setdefault('live-restore', True)
path.write_text(json.dumps(config, indent=2) + '\n')
PY
setup_tmp=$(mktemp -d)
trap 'rm -rf "$setup_tmp"' EXIT
curl -fsSL https://github.com/docker/buildx/releases/download/v0.38.0/buildx-v0.38.0.linux-amd64 \
  -o "$setup_tmp/docker-buildx"
printf '%s  %s\n' \
  4fe4cc38adf48169132749b6ca22a990928db0118e3407584ee553723115d287 \
  "$setup_tmp/docker-buildx" | sha256sum -c -
install -D -m 0755 "$setup_tmp/docker-buildx" /usr/local/lib/docker/cli-plugins/docker-buildx
if ! command -v uv >/dev/null; then
  python3 -m venv "$setup_tmp/uv"
  "$setup_tmp/uv/bin/python" -m pip install uv==0.13.3
  install -m 0755 "$setup_tmp/uv/bin/uv" /usr/local/bin/uv
fi
systemctl reset-failed docker.socket docker.service
systemctl enable --now containerd docker.socket docker
systemctl restart docker
docker version
docker buildx version
