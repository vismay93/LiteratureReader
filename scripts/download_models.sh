#!/usr/bin/env bash
# Downloads all local models into ./models (idempotent and resumable).
set -euo pipefail
cd "$(dirname "$0")/.."
M=models
mkdir -p "$M/kokoro"

# Use the hf CLI if installed, otherwise run it ephemerally through uv.
if command -v hf >/dev/null 2>&1; then HF=(hf); else HF=(uvx --from "huggingface_hub[cli]" hf); fi

echo "Layout/OCR: granite-docling-258M (~0.5 GB)"
"${HF[@]}" download ggml-org/granite-docling-258M-GGUF \
  granite-docling-258M-f16.gguf mmproj-granite-docling-258M-f16.gguf --local-dir "$M/granite-docling"

# PR_QWEN_QUANT: Q4_K_M (5 GB, default) | Q8_0 (8.7 GB) | F16 (16 GB); PR_MMPROJ_QUANT: Q8_0 | F16
QQ=${PR_QWEN_QUANT:-Q4_K_M}; MQ=${PR_MMPROJ_QUANT:-Q8_0}
echo "Narration/captions: Qwen3-VL-8B-Instruct $QQ (+ mmproj $MQ)"
"${HF[@]}" download Qwen/Qwen3-VL-8B-Instruct-GGUF \
  "Qwen3VL-8B-Instruct-$QQ.gguf" "mmproj-Qwen3VL-8B-Instruct-$MQ.gguf" --local-dir "$M/qwen3-vl-8b"

# Kokoro-82M (ONNX export used by the kokoro-onnx package)
echo "Speech: Kokoro-82M (~0.35 GB)"
K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -s "$M/kokoro/$f" ] || curl -fL --retry 3 -C - -o "$M/kokoro/$f" "$K/$f"
done
echo "Models ready in $M/"
