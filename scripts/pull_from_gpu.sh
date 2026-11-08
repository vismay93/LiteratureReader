#!/usr/bin/env bash
# Run on your laptop: export bundles on the GPU box, copy them over, import into the local library.
#   ./scripts/pull_from_gpu.sh user@gpu-host [--remote-dir paper-reader] [--audio mp3|wav] [--no-export]
# Uses ssh + rsync (resumable). Only papers missing from the local library are transferred.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${1:?usage: pull_from_gpu.sh user@gpu-host [--remote-dir DIR] [--audio mp3|wav] [--no-export]}; shift
RDIR=paper-reader; AUDIO=mp3; EXPORT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --remote-dir) RDIR=$2; shift 2 ;;
    --audio) AUDIO=$2; shift 2 ;;
    --no-export) EXPORT=0; shift ;;
    *) echo "Unknown option $1"; exit 2 ;;
  esac
done
[[ "$AUDIO" =~ ^(mp3|wav)$ ]] || { echo "--audio must be mp3 or wav"; exit 2; }
Q=$(printf '%q' "$RDIR")

if [ "$EXPORT" = 1 ]; then
  echo "==> Exporting on $HOST:$RDIR"
  ssh "$HOST" "cd $Q && backend/.venv/bin/python backend/batch.py export --all --out exports --audio $AUDIO"
fi

echo "==> Copying new bundles"
mkdir -p data/imports
# Skip papers already in the local library, so bundles can be deleted after import.
FILTER=$(mktemp); trap 'rm -f "$FILTER"' EXIT
for d in "${PR_DATA_DIR:-data/papers}"/*/; do
  if [ -f "$d/doc.json" ]; then echo "- $(basename "$d").tar.gz" >> "$FILTER"; fi
done
rsync -av --partial --filter="merge $FILTER" --include='*.tar.gz' --exclude='*' \
  "$HOST:$RDIR/exports/" data/imports/

echo "==> Importing (bundles are deleted once installed)"
(cd backend && uv run --quiet python batch.py import --delete ../data/imports)
echo "Done. Start the reader with ./scripts/start.sh (or --read-only, without models)"
