#!/usr/bin/env bash
# Start Paper Reader in the background inside Colab: both llama-servers + the backend on 127.0.0.1:8000.
# The notebook then opens it through Colab's authenticated port proxy.
#   bash scripts/colab/start_colab.sh        # start (idempotent)
#   bash scripts/colab/start_colab.sh stop   # stop everything
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
L=data/logs; mkdir -p "$L"
PORT=${PR_PORT:-8000}

if [ "${1:-}" = stop ]; then
  [ -f "$L/backend.pid" ] && kill "$(cat "$L/backend.pid")" 2>/dev/null || true
  rm -f "$L/backend.pid"
  ./scripts/stop_llama.sh
  echo "Stopped."; exit 0
fi

[ -f .gpu.env ] || { echo "Run scripts/colab/setup_colab.sh first."; exit 1; }
# shellcheck disable=SC1091
source .gpu.env
[ -d web/dist ] || { echo "web/dist missing - run scripts/colab/setup_colab.sh"; exit 1; }

# setsid: detach the servers from the notebook cell's shell so they outlive it.
setsid -w ./scripts/start_llama.sh

if curl -sf "http://127.0.0.1:$PORT/api/papers" >/dev/null; then
  echo "  backend already running on :$PORT"
else
  # Colab's proxy sends its own Host header, so accept any Host; the socket still binds to loopback.
  (cd backend && PR_ALLOWED_HOSTS="${PR_ALLOWED_HOSTS:-*}" setsid nohup .venv/bin/python app.py \
     > "$ROOT/$L/backend.log" 2>&1 & echo $! > "$ROOT/$L/backend.pid")
  for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/api/papers" >/dev/null && break; sleep 1; done
  curl -sf "http://127.0.0.1:$PORT/api/papers" >/dev/null || { echo "Backend failed - see $L/backend.log"; tail -20 "$L/backend.log"; exit 1; }
  echo "  backend ready on :$PORT"
fi
