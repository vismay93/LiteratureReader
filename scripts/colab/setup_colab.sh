#!/usr/bin/env bash
# One-time setup inside a Google Colab GPU runtime (Runtime > Change runtime type > T4 GPU or better).
# Called by colab/paper_reader.ipynb; you can also run it from a Colab terminal:
#   bash scripts/colab/setup_colab.sh
# It adds the pieces Colab lacks (Node 20+, the CUDA toolkit on PATH, the right CUDA_ARCH)
# and then runs the normal GPU setup (scripts/gpu/setup_gpu.sh -y).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

bold() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

command -v nvidia-smi >/dev/null || die "No GPU. In Colab: Runtime > Change runtime type > T4 GPU (or better)."

# CUDA toolkit ships in /usr/local/cuda on Colab but is not always on PATH.
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

# Build llama.cpp for this card only (much faster than all architectures): "7.5" -> 75.
if [ -z "${CUDA_ARCH:-}" ]; then
  CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  export CUDA_ARCH
fi
ok "GPU compute capability: sm_$CUDA_ARCH"

# ------------------------------------------------------------------ Node 20+ (for the web UI build)
bold "Node.js"
NODE_DIR="$ROOT/third_party/node"
[ -x "$NODE_DIR/bin/node" ] && export PATH="$NODE_DIR/bin:$PATH"
if ! have node || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 20 ]; then
  V=v22.12.0
  mkdir -p "$NODE_DIR"
  curl -fsSL "https://nodejs.org/dist/$V/node-$V-linux-x64.tar.xz" | tar -xJ -C "$NODE_DIR" --strip-components=1
  export PATH="$NODE_DIR/bin:$PATH"
fi
ok "node $(node -v)"

# ------------------------------------------------------------------ web UI
bold "Building the web UI"
(cd web && npm ci --silent && npm run build --silent >/dev/null)
ok "web/dist built"

# ------------------------------------------------------------------ GPU setup (llama.cpp CUDA build, Python env, models)
bold "GPU setup (first run builds llama.cpp with CUDA: ~10 min)"
./scripts/gpu/setup_gpu.sh -y
