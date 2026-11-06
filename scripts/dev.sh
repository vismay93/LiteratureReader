#!/usr/bin/env bash
# One-shot dev launcher: llama-servers + backend (:8000) + Vite (:5173), all on 127.0.0.1.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/start_llama.sh
trap './scripts/stop_llama.sh; kill 0' EXIT
(cd backend && uv run python app.py) &
(cd web && npm run dev -- --host 127.0.0.1) &
wait
