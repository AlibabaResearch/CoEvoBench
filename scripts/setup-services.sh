#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ "$(id -u)" -ne 0 ]; then
  echo "Run service installation as root." >&2
  exit 1
fi
test -x "$project_dir/.venv/bin/python"
# Other benchmarks can opt into the existing local registry service.
if [ "${1:-}" = "--with-docker-registry" ]; then
  if ! docker container inspect coevobench-registry >/dev/null 2>&1; then
    docker run -d --restart unless-stopped --name coevobench-registry \
      -p 127.0.0.1:5000:5000 -v coevobench-registry-data:/var/lib/registry registry:2.8.3
  else
    docker start coevobench-registry >/dev/null
  fi
fi
cat > /etc/systemd/system/coevobench-supervisor.service <<UNIT
[Unit]
Description=CoEvoBench experiment deadline supervisor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$project_dir
ExecStart="$project_dir/.venv/bin/python" "$project_dir/scripts/supervise_runs.py"
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now coevobench-supervisor
systemctl restart coevobench-supervisor
