#!/usr/bin/env bash
# One-time setup on a Linux + NVIDIA GPU machine (headless; no web UI needed).
#   ./scripts/gpu/setup_gpu.sh            # auto-size for the detected GPU
#   ./scripts/gpu/setup_gpu.sh -y         # no prompts
# Env overrides: PR_QWEN_QUANT, PR_MMPROJ_QUANT, PR_NP, LLAMA_SERVER, LLAMA_BUILD (e.g. b12144),
#                LLAMA_FROM_SOURCE=1, CUDA_ARCH (e.g. 86;89), SKIP_SMOKE=1
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
# Order: $LLAMA_SERVER > CUDA llama-server on PATH > prebuilt release binary (default, ~30 s)
#        > source build (LLAMA_FROM_SOURCE=1, or if the prebuilt one can't see the GPU).
bold "llama.cpp with CUDA"
cuda_ok() { "$1" --list-devices 2>&1 | grep -qi cuda; }
PRE=third_party/llama-prebuilt
WRAP=$ROOT/third_party/bin/llama-server

install_prebuilt() {
  local tag cuda drv url
  # Pinned to a tested build; LLAMA_BUILD=latest picks the newest bNNNN release.
  tag=${LLAMA_BUILD:-b12144}
  if [ "$tag" = latest ]; then
    tag=$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" | python3 -c \
      "import sys,json,re; print(next(r['tag_name'] for r in json.load(sys.stdin) if re.fullmatch(r'b\d+', r['tag_name'])))")
  fi
  [[ "$tag" =~ ^b[0-9]+$ ]] || { echo "  could not resolve llama.cpp release tag ($tag)"; return 1; }
  # Driver's max CUDA version decides 12.x vs 13.x binaries.
  drv=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
  if [ "${drv:-12}" -ge 13 ]; then cuda=13.4; else cuda=12.8; fi
  url=https://github.com/ggml-org/llama.cpp/releases/download/$tag
  echo "  downloading llama.cpp $tag (CUDA $cuda)"
  rm -rf "$PRE" && mkdir -p "$PRE" third_party/bin
  curl -fsSL "$url/llama-$tag-bin-ubuntu-cuda-$cuda-x64.tar.gz" | tar -xz -C "$PRE" --strip-components=1
  # The CUDA backend needs cudart/cublas; fetch them only if the host doesn't provide them.
  if LD_LIBRARY_PATH="$PRE:${LD_LIBRARY_PATH:-}" ldd "$PRE/libggml-cuda.so" | grep -q "not found"; then
    echo "  host lacks CUDA runtime libs; downloading cudart bundle"
    curl -fsSL "$url/cudart-llama-$tag-bin-ubuntu-cuda-$cuda-x64.tar.gz" | tar -xz -C "$PRE" --strip-components=1
  fi
  cat > "$WRAP" <<WRAPPER
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$ROOT/$PRE:\${LD_LIBRARY_PATH:-}"
exec "$ROOT/$PRE/llama-server" "\$@"
WRAPPER
  chmod +x "$WRAP"
  echo "$tag cuda-$cuda" > "$PRE/VERSION"
}

build_from_source() {
  for t in git cmake nvcc; do have $t || die "$t is required to build llama.cpp with CUDA (install CUDA toolkit / cmake / git)."; done
  confirm "Build llama.cpp with CUDA into third_party/llama.cpp (~5-10 min)?" || die "llama.cpp is required."
  mkdir -p third_party
  [ -d third_party/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
  local arch=(); [ -n "${CUDA_ARCH:-}" ] && arch=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release "${arch[@]}"
  cmake --build third_party/llama.cpp/build --target llama-server -j "$(nproc)"
}

BIN=${LLAMA_SERVER:-}
if [ -z "$BIN" ] && have llama-server && cuda_ok "$(command -v llama-server)"; then BIN=$(command -v llama-server); fi
if [ -z "$BIN" ] && [ "${LLAMA_FROM_SOURCE:-0}" != 1 ]; then
  if [ -x "$WRAP" ] && cuda_ok "$WRAP"; then BIN=$WRAP
  elif install_prebuilt && cuda_ok "$WRAP"; then BIN=$WRAP
  else echo "  prebuilt binary unusable here; falling back to a source build"; fi
fi
if [ -z "$BIN" ]; then
  [ -x third_party/llama.cpp/build/bin/llama-server ] || build_from_source
  BIN=$ROOT/third_party/llama.cpp/build/bin/llama-server
fi
cuda_ok "$BIN" || die "$BIN does not see a CUDA device."
ok "$BIN $( [ -f "$PRE/VERSION" ] && [ "$BIN" = "$WRAP" ] && cat "$PRE/VERSION")"

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
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
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
fi

bold "Done"
cat <<'EOF'
  Process:  ./scripts/gpu/batch.sh ~/papers/*.pdf 1706.03762 @list.txt
  Bundles land in exports/. From your laptop:  ./scripts/pull_from_gpu.sh user@gpu-host
EOF
