#!/usr/bin/env bash
# Run Paper Reader: both llama-servers + backend serving the built UI on http://127.0.0.1:8000.
# Everything binds to loopback only. Ctrl-C stops all of it.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d web/dist ] || { echo "web/dist missing - run ./scripts/install.sh first"; exit 1; }
[ -s models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf ] || { echo "Models missing - run ./scripts/download_models.sh"; exit 1; }

./scripts/start_llama.sh
trap './scripts/stop_llama.sh' EXIT
echo "Paper Reader: http://127.0.0.1:8000"
cd backend && uv run python app.py
