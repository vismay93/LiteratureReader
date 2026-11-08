#!/usr/bin/env bash
# Build the web UI into web/dist. Uses system Node >= 20.19 if present, otherwise downloads a
# portable Node 22 into third_party/node (no root needed; ~30 MB). Skips if dist is up to date.
set -euo pipefail
cd "$(dirname "$0")/.."
NODE_VER=${NODE_VERSION:-22.21.1}

ok_node() {
  command -v node >/dev/null 2>&1 || return 1
  node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>20||(a===20&&b>=19)?0:1)'
}

if [ -d web/dist ] && [ -z "$(find web/src web/index.html web/vite.config.ts -newer web/dist -print -quit)" ] \
   && [ "${FORCE:-0}" != 1 ]; then
  echo "web/dist is up to date"; exit 0
fi

if ! ok_node; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) plat=linux-x64 ;; Linux-aarch64) plat=linux-arm64 ;;
    Darwin-arm64) plat=darwin-arm64 ;; Darwin-x86_64) plat=darwin-x64 ;;
    *) echo "Unsupported platform for portable Node; install Node 20.19+"; exit 1 ;;
  esac
  dir=third_party/node-v$NODE_VER-$plat
  if [ ! -x "$dir/bin/node" ]; then
    echo "Downloading Node $NODE_VER ($plat)"
    mkdir -p third_party
    curl -fsSL "https://nodejs.org/dist/v$NODE_VER/node-v$NODE_VER-$plat.tar.gz" | tar -xz -C third_party
  fi
  export PATH="$PWD/$dir/bin:$PATH"
fi
echo "Using node $(node -v)"
cd web
if [ -f package-lock.json ]; then npm ci --no-audit --no-fund --loglevel=error; else npm install --no-audit --no-fund; fi
npm run build --silent
echo "Built web/dist"
