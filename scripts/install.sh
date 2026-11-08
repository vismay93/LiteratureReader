#!/usr/bin/env bash
# Paper Reader installer: checks/installs prerequisites, Python + web deps, models, and builds the UI.
# Safe to re-run (every step is idempotent).
#
#   ./scripts/install.sh               # full install
#   ./scripts/install.sh --skip-models # reader only: no ~6.3 GB model download and no llama.cpp
#                                      # (then ./scripts/start.sh --read-only)
#   ./scripts/install.sh -y            # don't ask before installing missing tools with Homebrew
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_MODELS=0; YES=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    -y|--yes) YES=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

bold() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

confirm() {
  [ "$YES" = 1 ] && return 0
  read -r -p "  $1 [Y/n] " r; [[ -z "$r" || "$r" =~ ^[Yy] ]]
}

brew_install() {  # brew_install <command> <formula>
  if have "$1"; then ok "$1 ($(command -v "$1"))"; return; fi
  have brew || die "$1 is missing and Homebrew isn't available. Install $2 manually, then re-run."
  confirm "$1 not found. Install '$2' with Homebrew?" || die "$1 is required."
  brew install "$2"
  ok "$1 installed"
}

# ------------------------------------------------------------------ prerequisites
bold "Checking prerequisites"
OS=$(uname -s)
[ "$OS" = Darwin ] || warn "Tested on macOS (Apple Silicon). On $OS install llama.cpp, uv and Node 20+ yourself."

if [ "$SKIP_MODELS" = 0 ]; then
  brew_install llama-server llama.cpp
elif have llama-server; then
  ok "llama-server ($(command -v llama-server))"
else
  warn "llama.cpp not installed (only needed to process papers; read-only mode works without it)"
fi
brew_install uv uv
brew_install node node
brew_install curl curl

NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
[ "$NODE_MAJOR" -ge 20 ] || die "Node 20+ required (found $(node -v))."
ok "node $(node -v)"
have llama-server && llama-server --version 2>&1 | grep -q version && ok "llama.cpp $(llama-server --version 2>&1 | grep -o 'version: [0-9]*')"

# Disk space check for models (~6.3 GB) when downloading.
if [ "$SKIP_MODELS" = 0 ]; then
  FREE_GB=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
  [ "$FREE_GB" -ge 8 ] || die "Need ~8 GB free for models, only ${FREE_GB} GB available."
  ok "${FREE_GB} GB free disk"
fi

# ------------------------------------------------------------------ backend
bold "Installing backend (Python, via uv)"
(cd backend && uv sync --quiet)
ok "backend/.venv ready"

# ------------------------------------------------------------------ frontend
bold "Installing and building frontend"
if [ -f web/package-lock.json ]; then (cd web && npm ci --silent); else (cd web && npm install --silent); fi
(cd web && npm run build --silent >/dev/null)
ok "web/dist built"

# ------------------------------------------------------------------ models
if [ "$SKIP_MODELS" = 0 ]; then
  bold "Downloading models (~6.3 GB, resumable)"
  ./scripts/download_models.sh
else
  warn "Skipped models. Run ./scripts/download_models.sh before first use."
fi

# ------------------------------------------------------------------ verify
bold "Verifying"
mkdir -p data/papers data/logs
(cd backend && uv run --quiet pytest -q >/dev/null) && ok "backend tests pass" || die "backend tests failed (cd backend && uv run pytest)"
missing=0
for f in models/granite-docling/granite-docling-258M-f16.gguf \
         models/granite-docling/mmproj-granite-docling-258M-f16.gguf \
         models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
         models/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf \
         models/kokoro/kokoro-v1.0.onnx models/kokoro/voices-v1.0.bin; do
  if [ -s "$f" ]; then ok "$f"; else warn "missing $f"; missing=1; fi
done

bold "Done"
if [ "$missing" = 1 ]; then
  echo "  Models are missing: run ./scripts/download_models.sh to process papers here,"
  echo "  or read imported papers without them: ./scripts/start.sh --read-only"
fi
cat <<'EOF'
  Start:  ./scripts/start.sh        then open http://127.0.0.1:8000
  Read:   ./scripts/start.sh --read-only   (no models loaded; import papers with batch.py import)
  Dev:    ./scripts/dev.sh          (hot-reload UI at http://127.0.0.1:5173)
  Stop:   Ctrl-C (or ./scripts/stop_llama.sh if the model servers are left running)
EOF
