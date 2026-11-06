#!/usr/bin/env bash
# One-time setup on a Linux + NVIDIA GPU machine (headless; no web UI needed).
#   ./scripts/gpu/setup_gpu.sh            # auto-size for the detected GPU
#   ./scripts/gpu/setup_gpu.sh -y         # no prompts
# Env overrides: PR_QWEN_QUANT, PR_MMPROJ_QUANT, PR_NP, LLAMA_SERVER, CUDA_ARCH (e.g. 86;89)
# Writes .gpu.env, which scripts/gpu/batch.sh sources.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
YES=0; [ "${1:-}" = "-y" ] && YES=1

bold() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
confirm() { [ "$YES" = 1 ] && return 0; read -r -p "  $1 [Y/n] " r; [[ -z "$r" || "$r" =~ ^[Yy] ]]; }

# ------------------------------------------------------------------ GPU
bold "Detecting GPU"
have nvidia-smi || die "nvidia-smi not found - install the NVIDIA driver first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
# Size the deployment to VRAM: Q8_0 weights ~10 GB with mmproj; each slot ~1.2 GB KV at 8k ctx.
if   [ "$VRAM_MB" -ge 70000 ]; then DEF_Q=Q8_0;   DEF_MM=F16;  DEF_NP=16
elif [ "$VRAM_MB" -ge 40000 ]; then DEF_Q=Q8_0;   DEF_MM=F16;  DEF_NP=8
elif [ "$VRAM_MB" -ge 20000 ]; then DEF_Q=Q8_0;   DEF_MM=Q8_0; DEF_NP=4
else                                 DEF_Q=Q4_K_M; DEF_MM=Q8_0; DEF_NP=2; fi
QQ=${PR_QWEN_QUANT:-$DEF_Q}; MQ=${PR_MMPROJ_QUANT:-$DEF_MM}; NP=${PR_NP:-$DEF_NP}
ok "VRAM ${VRAM_MB} MB -> Qwen $QQ, mmproj $MQ, $NP parallel slots"

# ------------------------------------------------------------------ uv
bold "uv"
if ! have uv; then
  confirm "uv not found. Install it to ~/.local/bin with the official installer?" || die "uv is required."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | cut -d' ' -f2)"

# ------------------------------------------------------------------ llama.cpp (CUDA)
bold "llama.cpp with CUDA"
cuda_ok() { "$1" --list-devices 2>&1 | grep -qi cuda; }
BIN=${LLAMA_SERVER:-}
if [ -z "$BIN" ] && have llama-server && cuda_ok "$(command -v llama-server)"; then BIN=$(command -v llama-server); fi
if [ -z "$BIN" ] && [ -x third_party/llama.cpp/build/bin/llama-server ]; then BIN=$ROOT/third_party/llama.cpp/build/bin/llama-server; fi
if [ -z "$BIN" ]; then
  for t in git cmake nvcc; do have $t || die "$t is required to build llama.cpp with CUDA (install CUDA toolkit / cmake / git)."; done
  confirm "Build llama.cpp with CUDA into third_party/llama.cpp (~5-10 min)?" || die "llama.cpp is required."
  mkdir -p third_party
  [ -d third_party/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
  ARCH_FLAG=(); [ -n "${CUDA_ARCH:-}" ] && ARCH_FLAG=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  # Fast build: only llama-server, only this GPU's architecture (CUDA_ARCH), no tests/examples,
  # static libs (one self-contained binary, easy to cache), Ninja when available.
  GEN=(); have ninja && GEN=(-G Ninja)
  cmake -S third_party/llama.cpp -B third_party/llama.cpp/build "${GEN[@]}" \
    -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release "${ARCH_FLAG[@]}"
  cmake --build third_party/llama.cpp/build --target llama-server -j "$(nproc)"
  BIN=$ROOT/third_party/llama.cpp/build/bin/llama-server
fi
cuda_ok "$BIN" || die "$BIN does not see a CUDA device."
ok "$BIN"

# ------------------------------------------------------------------ Python (+ CUDA onnxruntime for Kokoro)
bold "Python environment"
(cd backend && uv sync --quiet)
# kokoro-onnx pins the CPU onnxruntime; swap in the CUDA build (with pip-provided CUDA/cuDNN libs).
# NOTE: `uv run`/`uv sync` would revert this, so GPU scripts call backend/.venv/bin/python directly.
(cd backend && uv pip uninstall --quiet onnxruntime && uv pip install --quiet "onnxruntime-gpu[cuda,cudnn]")
PROV=$(backend/.venv/bin/python -c "import onnxruntime as o; print(','.join(o.get_available_providers()))")
echo "$PROV" | grep -q CUDAExecutionProvider || die "onnxruntime-gpu has no CUDA provider ($PROV)"
ok "onnxruntime providers: $PROV"

# ------------------------------------------------------------------ models
bold "Models"
PR_QWEN_QUANT=$QQ PR_MMPROJ_QUANT=$MQ ./scripts/download_models.sh

cat > .gpu.env <<ENV
# Written by scripts/gpu/setup_gpu.sh - edit to tune.
export LLAMA_SERVER="$BIN"
export PR_QWEN_QUANT=$QQ
export PR_MMPROJ_QUANT=$MQ
export PR_NP=$NP
export PR_TTS_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
ENV
ok "wrote .gpu.env"

# ------------------------------------------------------------------ smoke test
bold "Smoke test"
(cd backend && .venv/bin/python -m pytest -q >/dev/null) && ok "tests pass"
backend/.venv/bin/python - <<'PY'
import sys, time, tempfile, pathlib
sys.path.insert(0, "backend")
import os; os.environ.setdefault("PR_TTS_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
from pipeline import tts
out = pathlib.Path(tempfile.mkdtemp()) / "t.wav"
tts.synthesize("Warm up.", out)
t = time.time(); m = tts.synthesize("The Transformer relies entirely on attention to draw global dependencies between input and output. " * 3, out)
print(f"  ✓ Kokoro on {tts._engine().sess.get_providers()[0]}: {m['duration']/(time.time()-t):.0f}x realtime")
PY

bold "Done"
cat <<'EOF'
  Process:  ./scripts/gpu/batch.sh ~/papers/*.pdf 1706.03762 @list.txt
  Bundles land in exports/. From your laptop:  ./scripts/pull_from_gpu.sh user@gpu-host
EOF
