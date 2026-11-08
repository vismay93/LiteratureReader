#!/usr/bin/env bash
# Run Paper Reader on http://127.0.0.1:8000. Everything binds to loopback only. Ctrl-C stops all of it.
#
#   ./scripts/start.sh              # full: both llama-servers + backend (process, read, regenerate)
#   ./scripts/start.sh --read-only  # no models: read papers already processed or imported
#                                   # (~150 MB of RAM; upload and "Regenerate" are turned off)
set -euo pipefail
cd "$(dirname "$0")/.."

READ_ONLY=0
for a in "$@"; do
  case "$a" in
    --read-only) READ_ONLY=1 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a (try --help)" >&2; exit 2 ;;
  esac
done

[ -d web/dist ] || { echo "web/dist missing - run ./scripts/install.sh first"; exit 1; }

if [ "$READ_ONLY" = 1 ]; then
  echo "Paper Reader (read-only, no models): http://127.0.0.1:8000"
  cd backend && PR_READ_ONLY=1 exec uv run python app.py
fi

[ -s models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf ] || {
  echo "Models missing - run ./scripts/download_models.sh,"
  echo "or ./scripts/start.sh --read-only to read imported papers without them."
  exit 1
}

./scripts/start_llama.sh
trap './scripts/stop_llama.sh' EXIT
echo "Paper Reader: http://127.0.0.1:8000"
cd backend && PR_READ_ONLY=0 uv run python app.py
