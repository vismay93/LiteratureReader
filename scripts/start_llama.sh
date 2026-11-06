#!/usr/bin/env bash
# Starts both llama-servers on loopback only (skips any already running). Logs go to data/logs/.
#
# Tunables (env; scripts/gpu/* set these from .gpu.env):
#   LLAMA_SERVER      llama-server binary            (default: llama-server on PATH)
#   PR_NP             parallel slots per server       (default 2; backend reads the same var)
#   PR_QWEN_QUANT     Q4_K_M | Q8_0 | F16             (default Q4_K_M)
#   PR_MMPROJ_QUANT   Q8_0 | F16                      (default Q8_0)
#   PR_QWEN_SLOT_CTX  context tokens per Qwen slot    (default 8192)
#   PR_NGL            layers offloaded to GPU         (default 99 = all)
set -euo pipefail
cd "$(dirname "$0")/.."
M=models; L=data/logs; mkdir -p "$L"
BIN=${LLAMA_SERVER:-llama-server}
NP=${PR_NP:-2}
QQ=${PR_QWEN_QUANT:-Q4_K_M}
MQ=${PR_MMPROJ_QUANT:-Q8_0}
QCTX=${PR_QWEN_SLOT_CTX:-8192}
NGL=${PR_NGL:-99}
up() { curl -sf "http://127.0.0.1:$1/health" >/dev/null; }

QWEN="$M/qwen3-vl-8b/Qwen3VL-8B-Instruct-$QQ.gguf"
QMM="$M/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-$MQ.gguf"
for f in "$QWEN" "$QMM"; do
  [ -s "$f" ] || { echo "Missing $f - run PR_QWEN_QUANT=$QQ PR_MMPROJ_QUANT=$MQ ./scripts/download_models.sh"; exit 1; }
done

if up 8081; then echo "  :8081 already running"; else
  "$BIN" --host 127.0.0.1 --port 8081 \
    -m "$M/granite-docling/granite-docling-258M-f16.gguf" \
    --mmproj "$M/granite-docling/mmproj-granite-docling-258M-f16.gguf" \
    -c $((16384 * NP)) -np "$NP" -ngl "$NGL" --jinja --special > "$L/llama-docling.log" 2>&1 &
  echo $! > "$L/llama-docling.pid"
fi

if up 8082; then echo "  :8082 already running"; else
  "$BIN" --host 127.0.0.1 --port 8082 -m "$QWEN" --mmproj "$QMM" \
    -c $((QCTX * NP)) -np "$NP" -ngl "$NGL" -fa on --jinja > "$L/llama-qwen.log" 2>&1 &
  echo $! > "$L/llama-qwen.pid"
fi

echo "Waiting for llama-servers ($NP slots each, Qwen $QQ)..."
for port in 8081 8082; do
  for _ in $(seq 300); do up $port && break; sleep 1; done
  up $port || { echo "  :$port failed to start - see $L/"; exit 1; }
  echo "  :$port ready"
done
