#!/usr/bin/env bash
# Batch-process papers on the GPU box, then write transferable bundles to exports/.
#   ./scripts/gpu/batch.sh paper1.pdf papers_dir/ 1706.03762 @reading-list.txt [--parallel 3] [--audio wav]
# Starts the llama-servers (tuned from .gpu.env), runs batch.py, stops the servers on exit
# unless KEEP_SERVERS=1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$HERE/.gpu.env" ] || { echo "Run scripts/gpu/setup_gpu.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$HERE/.gpu.env"
[ $# -gt 0 ] || { sed -n '2,5p' "$0"; exit 2; }

"$HERE/scripts/start_llama.sh"
[ "${KEEP_SERVERS:-0}" = 1 ] || trap '"$HERE/scripts/stop_llama.sh"' EXIT

# Run from the caller's cwd so relative input paths work; batch.py resolves its own paths.
"$HERE/backend/.venv/bin/python" "$HERE/backend/batch.py" process "$@" \
  --export "${PR_EXPORT_DIR:-$HERE/exports}" --report "$HERE/data/logs/last-batch.json"
