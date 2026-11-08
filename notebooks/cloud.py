"""Helpers for running Paper Reader from a Colab / SageMaker / Jupyter notebook.

Inference engines (models come straight from Hugging Face in the first two):
  * "vllm" (default) - vLLM OpenAI servers (continuous batching, CUDA graphs), installed from
    prebuilt wheels into an isolated venv (.venv-vllm) so the kernel's torch is untouched.
  * "sglang" - SGLang servers (RadixAttention, CUDA graphs) in an isolated venv (.venv-sglang).
    Needs an NVIDIA driver with CUDA 13 (>= 580) and glibc >= 2.34 (Ubuntu 22.04+, AL2023).
  * "transformers" - inference/hf_server.py on the kernel's own torch. Slowest, but most portable.
  * "llama.cpp" - prebuilt CUDA llama-server + GGUF models (same stack as the desktop app).

Everything listens on 127.0.0.1 inside the VM. The UI is reached only through the notebook
platform's own authenticated proxy (Colab's kernel port proxy, or jupyter-server-proxy on
SageMaker / JupyterHub) - no public tunnels are opened.

    import cloud
    cloud.setup()                      # install deps + prefetch models + build UI (idempotent)
    cloud.start()                      # start inference + backend, print a link to the UI
    cloud.process(["1706.03762"])      # optional: batch-process without the UI
    cloud.download_bundles()           # optional: take processed papers home
    cloud.stop()
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "logs"
STATE = ROOT / ".cloud.json"
VENV_PY = ROOT / "backend" / ".venv" / "bin" / "python"
VLLM_PY = ROOT / ".venv-vllm" / "bin" / "python"
SGLANG_PY = ROOT / ".venv-sglang" / "bin" / "python"
# Pinned: CLI flags and the dots.ocr processor were checked against this release's source.
SGLANG_REQ = ["sglang==0.5.23", "flash-attn-4>=4.0.0b18"]  # its flash-attn-4 dep is a pre-release
DOTS_TEMPLATE = ROOT / "data" / "dots_chat_template.jinja"
HF_PORT = 8081
LAYOUT_PORT, VLM_PORT = 8081, 8082
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
LAYOUT_MODEL = "ibm-granite/granite-docling-258M"
DOTS_MODEL = "dots-studio/dots.ocr"            # official repo (rednote-hilab/dots.ocr redirects here)
# dots.ocr ships custom code (trust_remote_code); pin the reviewed commit so it can't change under us.
DOTS_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"
_backend: subprocess.Popen | None = None
_port = 8000


# ------------------------------------------------------------------ helpers

def platform() -> str:
    """'colab' | 'sagemaker' | 'jupyter'. PR_PLATFORM forces one (the Colab notebook sets 'colab')."""
    if os.environ.get("PR_PLATFORM"):
        return os.environ["PR_PLATFORM"]
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if Path("/opt/ml/metadata/resource-metadata.json").exists() or os.environ.get("SAGEMAKER_INTERNAL_IMAGE_URI"):
        return "sagemaker"
    return "jupyter"


def _sh(cmd: list[str] | str, env: dict | None = None, cwd: Path = ROOT, check: bool = True) -> int:
    """Run a command, streaming its output into the notebook cell."""
    p = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})}, shell=isinstance(cmd, str),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if check and rc:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def gpu_env() -> dict[str, str]:
    """Settings written by setup_gpu.sh (.gpu.env), used by the llama.cpp engine."""
    env: dict[str, str] = {}
    f = ROOT / ".gpu.env"
    if f.exists():
        for line in f.read_text().splitlines():
            m = re.match(r'^\s*export\s+([A-Z_][A-Z0-9_]*)=(.*)$', line)
            if m:
                env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def _engine_env(st: dict | None = None) -> dict[str, str]:
    """Environment for the backend / batch CLI matching the configured engine."""
    st = st or _state()
    if st.get("engine") in ("vllm", "sglang"):
        dots = st.get("layout_model") == DOTS_MODEL
        env = {"PR_LAYOUT_URL": f"http://127.0.0.1:{LAYOUT_PORT}",
               "PR_VLM_URL": f"http://127.0.0.1:{VLM_PORT}",
               "PR_LAYOUT_MODEL_NAME": "layout", "PR_VLM_MODEL_NAME": "vlm",
               "PR_LAYOUT_MODEL": "dots" if dots else "docling",
               # dots.ocr: ~200 dpi pages, long JSON output; docling: 8192-token context incl. image
               "PR_LAYOUT_LONG_SIDE": "1600" if dots else "1024",
               "PR_LAYOUT_MAX_TOKENS": str(st.get("layout_max_tokens", 16000 if dots else 6800)),
               "PR_MAX_IMAGE_PIXELS": "1300000",
               # SGLang renders dots' processor template, which inserts the image placeholder
               "PR_DOTS_IMAGE_PLACEHOLDER": "0" if st["engine"] == "sglang" else "1",
               "PR_NP": str(st.get("concurrency", 16))}
        if st.get("hf_home"):
            env["HF_HOME"] = st["hf_home"]
        return env
    if st.get("engine") == "transformers":
        env = {"PR_LAYOUT_URL": f"http://127.0.0.1:{HF_PORT}/layout",
               "PR_VLM_URL": f"http://127.0.0.1:{HF_PORT}/vlm",
               "PR_NP": str(st.get("max_batch", 4))}
        if st.get("hf_home"):
            env["HF_HOME"] = st["hf_home"]
        return env
    return gpu_env()


def _py(st: dict | None = None) -> str:
    """Python that runs the backend: the kernel's (vllm/transformers) or backend/.venv (llama.cpp)."""
    return str(VENV_PY) if (st or _state()).get("engine") == "llama.cpp" else sys.executable


def _gpu() -> tuple[str, int, float]:
    """(name, VRAM MB, compute capability) of the largest GPU, via nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "none", 0, 0.0
    rows = [[c.strip() for c in ln.split(",")] for ln in out.strip().splitlines() if ln.strip()]
    name, mem, cc = max(rows, key=lambda r: int(r[1]))
    return name, int(mem), float(cc)


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def _uv() -> str | None:
    """uv binary (installed with pip on first use); much faster than pip for large wheels."""
    found = shutil.which("uv") or (str(Path(sys.executable).parent / "uv")
                                   if (Path(sys.executable).parent / "uv").exists() else None)
    if found is None:
        _sh([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
        cand = Path(sys.executable).parent / "uv"
        found = shutil.which("uv") or (str(cand) if cand.exists() else None)
    return found


def _pip(*args: str):
    """Install into the kernel's Python (uv if available, else pip)."""
    uv = _uv()
    if uv:
        _sh([uv, "pip", "install", "--python", sys.executable, *args])
    else:
        _sh([sys.executable, "-m", "pip", "install", "-q", *args])


def _uninstall(pkg: str):
    uv = _uv()
    if uv:
        _sh([uv, "pip", "uninstall", "--python", sys.executable, pkg], check=False)
    else:
        _sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg], check=False)


def _has_module(mod: str) -> bool:
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0


# ------------------------------------------------------------------ setup

def setup(engine: str = "vllm", layout: str = "auto", vlm_model: str | None = None, quant: str | None = None,
          max_batch: int | None = None, hf_home: str | None = None, models_dir: str | None = None,
          # llama.cpp engine options
          qwen_quant: str | None = None, parallel: int | None = None, llama_build: str | None = None,
          from_source: bool = False, smoke_test: bool = False):
    """Install everything (idempotent) and remember the choices for start().

    engine="vllm" (default): vLLM servers, models from Hugging Face (see vllm_plan for sizing).
        layout:     "auto" (dots.ocr on >=20 GB GPUs, else granite-docling) | "dots" | "docling"
        vlm_model:  narration model; default Qwen/Qwen3.8-27B-FP8 on >=70 GB GPUs,
                    else Qwen3-VL 8B / 4B(-FP8) / 2B by VRAM
        max_batch:  max concurrent sequences for the narration model (default sized to VRAM)
        hf_home:    Hugging Face cache dir (e.g. on Google Drive to avoid re-downloading)
    engine="sglang": same models and sizing as vllm, served by SGLang (CUDA 13 drivers only).
    engine="transformers": inference/hf_server.py on the kernel's torch (vlm_model, quant, max_batch).
        quant:      None | "4bit" | "8bit" (bitsandbytes)
    engine="llama.cpp": prebuilt CUDA llama-server + GGUF (qwen_quant, parallel, llama_build, from_source)
    """
    t0 = time.time()
    if engine not in ("vllm", "sglang", "transformers", "llama.cpp"):
        raise ValueError("engine must be 'vllm', 'sglang', 'transformers' or 'llama.cpp'")
    plat = platform()
    if plat == "sagemaker" and models_dir is None:
        models_dir = "/tmp/paper-reader-models"  # SageMaker home volumes are often only 5 GB
    if plat == "sagemaker" and hf_home is None and engine != "llama.cpp":
        hf_home = "/tmp/paper-reader-hf"
    if models_dir:
        _link_models(Path(models_dir))
    name, vram, cc = _gpu()
    free = shutil.disk_usage((ROOT / "models").resolve() if (ROOT / "models").exists() else ROOT).free / 1e9
    print(f"Platform: {plat} | GPU: {name} ({vram} MB, sm {cc}) | free disk: {free:.0f} GB | engine: {engine}")
    if vram == 0:
        print("WARNING: no NVIDIA GPU detected - processing will be very slow on CPU.")
    if free < 12:
        print("WARNING: <12 GB free; pass models_dir=/hf_home= pointing at a larger volume.")

    if engine == "llama.cpp":
        env = {"PATH": f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"}
        for k, v in (("PR_QWEN_QUANT", qwen_quant), ("PR_NP", parallel), ("LLAMA_BUILD", llama_build)):
            if v:
                env[k] = str(v)
        if from_source:
            env["LLAMA_FROM_SOURCE"] = "1"
        if not smoke_test:
            env["SKIP_SMOKE"] = "1"
        _sh(["bash", "scripts/gpu/setup_gpu.sh", "-y"], env=env)
        st = {"engine": "llama.cpp"}
    elif engine in ("vllm", "sglang"):
        if vram == 0:
            raise RuntimeError(f"{engine} needs an NVIDIA GPU; use engine='transformers' on CPU/MPS.")
        if engine == "sglang":
            st = _setup_sglang(vram, cc, layout, vlm_model, max_batch, hf_home)
        else:
            st = _setup_vllm(vram, cc, layout, vlm_model, max_batch, hf_home)
    else:
        st = _setup_transformers(vram, vlm_model, quant, max_batch, hf_home)

    _sh(["bash", "scripts/build_web.sh"])
    STATE.write_text(json.dumps(st, indent=1))
    print(f"\nSetup finished in {time.time() - t0:.0f}s ({json.dumps(st)}). Next: cloud.start()")


def _setup_transformers(vram: int, vlm_model, quant, max_batch, hf_home) -> dict:
    if vram >= 38000:
        d_model, d_batch = "Qwen/Qwen3-VL-8B-Instruct", 8
    elif vram >= 20000:
        d_model, d_batch = "Qwen/Qwen3-VL-4B-Instruct", 6
    elif vram >= 12000:
        d_model, d_batch = "Qwen/Qwen3-VL-4B-Instruct", 4
    else:
        d_model, d_batch = "Qwen/Qwen3-VL-2B-Instruct", 2
    st = {"engine": "transformers", "layout_model": LAYOUT_MODEL, "vlm_model": vlm_model or d_model,
          "quant": quant or "none", "max_batch": max_batch or d_batch, "hf_home": hf_home}
    print(f"Models: layout {LAYOUT_MODEL}, narration {st['vlm_model']} (quant {st['quant']}), "
          f"batch {st['max_batch']}")

    print("\n==> Python packages (into this kernel's environment)")
    if not _has_module("torch"):
        print("torch not found - installing torch + torchvision")
        _pip("torch", "torchvision")
    elif not _has_module("torchvision"):
        _pip("torchvision")
    _pip("transformers>=5.0", "accelerate>=1.0")
    if st["quant"] != "none":
        _pip("bitsandbytes")
    _setup_backend(vram, torch_in_kernel=True)
    _prefetch([LAYOUT_MODEL, st["vlm_model"]], hf_home)
    _backend_tests()
    return st


# vLLM on PyPI is built for CUDA 13 (driver >= 580). Older drivers get the release's CUDA 12.9 wheel
# with matching torch from the PyTorch index. Both resolve to prebuilt wheels (no compiling).
VLLM_CU129_WHEEL = ("https://github.com/vllm-project/vllm/releases/download/v0.32.0/"
                    "vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl")


def _driver_cuda() -> float:
    """Highest CUDA version the installed NVIDIA driver supports (from nvidia-smi), or 0."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0.0
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return float(m.group(1)) if m else 0.0


def vllm_install_args(driver_cuda: float) -> list[str]:
    if driver_cuda >= 13.0:
        return ["vllm"]
    return [f"vllm @ {VLLM_CU129_WHEEL}", "--extra-index-url", "https://download.pytorch.org/whl/cu129",
            "--index-strategy", "unsafe-best-match"]


# Narration models from largest to smallest; used for the plan and for automatic step-down.
VLM_LADDER = ["Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct-FP8",
              "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct-FP8", "Qwen/Qwen3-VL-2B-Instruct"]
# Largest images we send (profiling hints, see vllm_command): layout pages are rendered at
# 1600 px (dots) / 1024 px (docling) on the long side; VLM crops are capped at 1.3 MP.
IMAGE_HINT = {"dots": (1232, 1596), "docling": (792, 1024), "vlm": (1140, 1140)}


def vllm_plan(vram: int, cc: float, layout: str = "auto", vlm_model: str | None = None,
              max_batch: int | None = None) -> dict:
    """Pick models and GPU memory split for two servers on one GPU (pure; unit-tested).

    Each server's share must cover weights + profiling activations + KV cache:
      dots.ocr ~6 GB weights, granite-docling ~0.5 GB; narration gets the rest of a 0.90 budget.
    Narration prompts are short (image + ~1k tokens, <=2k out), so 8k context is plenty below
    38 GB; dots pages need ~2.5k image tokens + up to 12k output tokens.
    """
    gb = vram / 1000
    if layout == "auto":
        layout = "dots" if vram >= 20000 else "docling"
    if layout not in ("dots", "docling"):
        raise ValueError("layout must be 'auto', 'dots' or 'docling'")
    big = vram >= 38000
    # FP8 weights run natively on Ada/Hopper (sm >= 8.9) and via weight-only Marlin on Ampere.
    fp8 = cc >= 8.0
    if vram >= 70000:
        d_model, seqs, max_len, eager = "Qwen/Qwen3.8-27B-FP8", 32, 16384, False
    elif big:
        d_model, seqs, max_len, eager = "Qwen/Qwen3-VL-8B-Instruct", 32, 16384, False
    elif vram >= 20000:   # L4 / A10G (24 GB): two servers must share ~21 GB
        d_model = "Qwen/Qwen3-VL-4B-Instruct-FP8" if fp8 else "Qwen/Qwen3-VL-4B-Instruct"
        seqs, max_len, eager = 12, 8192, False
    elif vram >= 12000:
        d_model, seqs, max_len, eager = "Qwen/Qwen3-VL-4B-Instruct", 8, 8192, True
    else:
        d_model, seqs, max_len, eager = "Qwen/Qwen3-VL-2B-Instruct", 8, 8192, True
    need = 9.5 if layout == "dots" else 2.5      # GB: weights + activations + KV
    layout_util = round(min(0.45, max(0.08, need / gb)), 2) if gb else 0.2
    dots = layout == "dots"
    plan = {
        "engine": "vllm",
        "layout_model": DOTS_MODEL if dots else LAYOUT_MODEL,
        "layout_max_len": (32768 if big else 16384) if dots else 8192,
        "layout_max_tokens": (16000 if big else 12000) if dots else 6800,
        "layout_seqs": 16 if big else 8,
        "layout_gpu_util": layout_util,
        "vlm_model": vlm_model or d_model,
        "max_num_seqs": max_batch or seqs,
        "max_model_len": max_len,
        "vlm_gpu_util": round(0.90 - layout_util, 2),
        "enforce_eager": eager,
        # pre-Ampere GPUs (T4, V100) have no bfloat16
        "dtype": "half" if cc < 8.0 else "auto",
        "fp8_ok": fp8,
    }
    plan["concurrency"] = plan["max_num_seqs"]
    return plan


OOM_MARKERS = ("No available memory for the cache blocks", "CUDA out of memory",
               "OutOfMemoryError", "Not enough memory", "is greater than the available",
               "Insufficient memory", "less than desired GPU memory utilization")


def is_oom(log_text: str) -> bool:
    return any(m.lower() in log_text.lower() for m in OOM_MARKERS)


def shrink_plan(st: dict, which: str) -> dict | None:
    """A smaller configuration after `which` ("layout" | "vlm") ran out of memory, or None.

    vlm:    halve concurrency once, then step down the model ladder (skipping FP8 where it
            can't run), finally shorten the context.
    layout: dots.ocr -> granite-docling, handing the freed memory to the narration server.
    """
    st = dict(st)
    if which == "layout":
        if st["layout_model"] != DOTS_MODEL:
            return None
        st.update(layout_model=LAYOUT_MODEL, layout_max_len=8192, layout_max_tokens=6800,
                  layout_seqs=8)
        freed = st["layout_gpu_util"] - 0.12
        st["layout_gpu_util"] = 0.12
        st["vlm_gpu_util"] = round(st["vlm_gpu_util"] + max(0.0, freed), 2)
        return st
    if st["max_num_seqs"] > 8:
        st["max_num_seqs"] = st["concurrency"] = max(8, st["max_num_seqs"] // 2)
        return st
    ladder = [m for m in VLM_LADDER if st.get("fp8_ok", True) or not m.endswith("-FP8")]
    if st["vlm_model"] in ladder:
        i = ladder.index(st["vlm_model"])
        if i + 1 < len(ladder):
            st["vlm_model"] = ladder[i + 1]
            return st
    if st["max_model_len"] > 4096:
        st["max_model_len"] = 4096
        return st
    return None


def _setup_vllm(vram: int, cc: float, layout: str, vlm_model, max_batch, hf_home) -> dict:
    st = {**vllm_plan(vram, cc, layout, vlm_model, max_batch), "hf_home": hf_home}
    print(f"Models: layout {st['layout_model']}, narration {st['vlm_model']} | dtype {st['dtype']}, "
          f"{st['max_num_seqs']} seqs, ctx {st['max_model_len']}, "
          f"GPU share {st['layout_gpu_util']}+{st['vlm_gpu_util']}")

    print(f"\n==> vLLM (prebuilt wheels, isolated venv .venv-vllm; driver CUDA {_driver_cuda()})")
    uv = _uv()
    if not uv:
        raise RuntimeError("could not install uv (needed to install vLLM quickly)")
    if not VLLM_PY.exists():
        _sh([uv, "venv", "--python", "3.12", str(VLLM_PY.parent.parent)])
    if subprocess.run([str(VLLM_PY), "-c", "import vllm"], capture_output=True).returncode != 0:
        _sh([uv, "pip", "install", "--python", str(VLLM_PY), *vllm_install_args(_driver_cuda())])
    _sh([uv, "pip", "install", "--python", str(VLLM_PY), "ninja"])  # for any FlashInfer JIT fallback
    _sh([str(VLLM_PY), "-c", "import vllm, torch; print('vllm', vllm.__version__, '| torch', "
         "torch.__version__, '| cuda', torch.cuda.is_available())"])

    _setup_backend(vram, torch_in_kernel=_has_module("torch"))
    _prefetch([st["layout_model"], st["vlm_model"]], hf_home,
              revisions={DOTS_MODEL: DOTS_REVISION})
    _backend_tests()
    return st


def sglang_requirements_ok(driver_cuda: float, glibc: tuple[int, int]) -> str | None:
    """None if SGLang's prebuilt wheels can run here, else the reason they can't."""
    if driver_cuda < 13.0:
        return (f"SGLang 0.5.x ships CUDA 13 builds only; this driver supports CUDA {driver_cuda or '?'} "
                "(needs driver >= 580). Use engine='vllm', which has a CUDA 12.9 build.")
    if glibc < (2, 34):
        return (f"SGLang wheels need glibc >= 2.34 (found {glibc[0]}.{glibc[1]}; e.g. Amazon Linux 2). "
                "Use engine='vllm'.")
    return None


def _glibc() -> tuple[int, int]:
    try:
        v = os.confstr("CS_GNU_LIBC_VERSION").split()[-1]
        major, minor = v.split(".")[:2]
        return int(major), int(minor)
    except (AttributeError, ValueError, OSError, IndexError):
        return (0, 0)


def _setup_sglang(vram: int, cc: float, layout: str, vlm_model, max_batch, hf_home) -> dict:
    why = sglang_requirements_ok(_driver_cuda(), _glibc())
    if why:
        raise RuntimeError(why)
    st = {**vllm_plan(vram, cc, layout, vlm_model, max_batch), "engine": "sglang", "hf_home": hf_home}
    print(f"Models: layout {st['layout_model']}, narration {st['vlm_model']} | dtype {st['dtype']}, "
          f"{st['max_num_seqs']} seqs, ctx {st['max_model_len']}")

    print("\n==> SGLang (prebuilt wheels, isolated venv .venv-sglang)")
    uv = _uv()
    if not uv:
        raise RuntimeError("could not install uv (needed to install SGLang quickly)")
    if not SGLANG_PY.exists():
        _sh([uv, "venv", "--python", "3.12", str(SGLANG_PY.parent.parent)])
    if subprocess.run([str(SGLANG_PY), "-c", "import sglang"], capture_output=True).returncode != 0:
        _sh([uv, "pip", "install", "--python", str(SGLANG_PY), *SGLANG_REQ])
    _sh([uv, "pip", "install", "--python", str(SGLANG_PY), "ninja"])
    _sh([str(SGLANG_PY), "-c", "import sglang, torch; print('sglang', sglang.__version__, '| torch', "
         "torch.__version__, '| cuda', torch.cuda.is_available())"])

    _setup_backend(vram, torch_in_kernel=_has_module("torch"))
    _prefetch([st["layout_model"], st["vlm_model"]], hf_home, revisions={DOTS_MODEL: DOTS_REVISION})
    if st["layout_model"] == DOTS_MODEL:
        _write_dots_template(hf_home)
        st["dots_chat_template"] = str(DOTS_TEMPLATE)
    _backend_tests()
    return st


def _write_dots_template(hf_home: str | None):
    """dots.ocr's processor chat template (emits the image placeholder per image part)."""
    env = {"HF_HOME": hf_home} if hf_home else {}
    DOTS_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    _sh([sys.executable, "-c",
         "import json, sys; from huggingface_hub import hf_hub_download\n"
         "p = hf_hub_download(sys.argv[1], 'chat_template.json', revision=sys.argv[2])\n"
         "open(sys.argv[3], 'w').write(json.load(open(p))['chat_template'])\n"
         "print('dots chat template ->', sys.argv[3])",
         DOTS_MODEL, DOTS_REVISION, str(DOTS_TEMPLATE)], env=env)


def _setup_backend(vram: int, torch_in_kernel: bool):
    """Backend deps into the kernel, Kokoro on GPU, and the Kokoro model files."""
    _pip("-r", str(ROOT / "notebooks" / "requirements-cloud.txt"))
    if vram:
        # Kokoro on GPU: swap the CPU onnxruntime pulled in by kokoro-onnx for the CUDA build.
        # Without torch in the kernel, pull CUDA/cuDNN runtime libs from pip as well.
        probe = "import onnxruntime as o; print('CUDAExecutionProvider' in o.get_available_providers())"
        has_cuda = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.strip()
        if has_cuda != "True":
            _uninstall("onnxruntime")
            _pip("onnxruntime-gpu" if torch_in_kernel else "onnxruntime-gpu[cuda,cudnn]")

    print("\n==> Kokoro-82M voice model")
    kdir = ROOT / "models" / "kokoro"
    kdir.mkdir(parents=True, exist_ok=True)
    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not (kdir / f).exists() or (kdir / f).stat().st_size == 0:
            print(f"downloading {f}")
            urllib.request.urlretrieve(f"{KOKORO_URL}/{f}", kdir / f)


def _prefetch(repos: list[str], hf_home: str | None, revisions: dict | None = None):
    print("\n==> Prefetching Hugging Face models")
    env = {"HF_HOME": hf_home} if hf_home else {}
    specs = [f"{r}@{(revisions or {}).get(r, '')}" for r in repos]
    _sh([sys.executable, "-c",
         "import sys; from huggingface_hub import snapshot_download\n"
         "for spec in sys.argv[1:]:\n"
         "    r, _, rev = spec.partition('@')\n"
         "    print(r, '->', snapshot_download(r, revision=rev or None, allow_patterns=["
         "'*.json','*.jinja','*.safetensors','*.txt','*.model','*.tiktoken','*.py']))",
         *specs], env=env)


def _backend_tests():
    print("\n==> Backend tests")
    _sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings"],
        cwd=ROOT / "backend", check=False)


def _link_models(target: Path):
    target.mkdir(parents=True, exist_ok=True)
    link = ROOT / "models"
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if any(link.iterdir()):
            print(f"models/ already has files; leaving it in place (not linking to {target})")
            return
        link.rmdir()
    link.symlink_to(target)
    print(f"models/ -> {target}")


# ------------------------------------------------------------------ run

def _proxy(port: int) -> tuple[str | None, list[str]]:
    """(UI URL or None, extra Host headers to accept) for this platform's authenticated proxy."""
    plat = platform()
    if plat == "colab":
        from google.colab.output import eval_js  # type: ignore
        url = eval_js(f"google.colab.kernel.proxyPort({port}, {{'cache': false}})")
        host = re.sub(r"^https?://([^/:]+).*$", r"\1", url)
        return url, [host]
    base = _jupyter_base_url()
    hosts = ["*.sagemaker.aws", "*.on.aws"] if plat == "sagemaker" else []
    if base is not None:
        return f"{base}proxy/{port}/", hosts
    return None, hosts


def _jupyter_base_url() -> str | None:
    """Jupyter server base_url (e.g. '/' or '/jupyterlab/default/'), used for jupyter-server-proxy."""
    try:
        from jupyter_server import serverapp  # type: ignore
        servers = list(serverapp.list_running_servers())
    except Exception:
        try:
            from notebook import notebookapp  # type: ignore
            servers = list(notebookapp.list_running_servers())
        except Exception:
            servers = []
    if servers:
        b = servers[0].get("base_url", "/")
        return b if b.endswith("/") else b + "/"
    return os.environ.get("JUPYTERHUB_SERVICE_PREFIX")


def start_inference(timeout: int = 1200):
    """Start the model server(s) for the configured engine and wait until models are loaded."""
    st = _state()
    if not st:
        raise RuntimeError("Run cloud.setup() first.")
    if st["engine"] == "llama.cpp":
        _sh(["bash", "scripts/start_llama.sh"], env=gpu_env())
        return
    if st["engine"] == "sglang":
        # SGLang sizes each server from the memory free *when it starts*, so the second server's
        # fraction is relative to what the first left over.
        f_layout = st["layout_gpu_util"]
        f_vlm = sglang_vlm_fraction(st["layout_gpu_util"], st["vlm_gpu_util"])
        _start_server("sglang", "layout", LAYOUT_PORT, st, timeout, sglang_command(
            "layout", st["layout_model"], LAYOUT_PORT, st, mem_fraction=f_layout,
            max_len=st.get("layout_max_len", 8192), seqs=st.get("layout_seqs", 16), eager=True))
        _start_server("sglang", "vlm", VLM_PORT, st, timeout, sglang_command(
            "vlm", st["vlm_model"], VLM_PORT, st, mem_fraction=f_vlm, max_len=st["max_model_len"],
            seqs=st["max_num_seqs"], eager=st["enforce_eager"]))
        return
    if st["engine"] == "vllm":
        _start_vllm_pair(st, timeout)
        return
    if _healthy(f"http://127.0.0.1:{HF_PORT}/health"):
        print(f"inference server already running on :{HF_PORT}")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "inference" / "hf_server.py"), "--port", str(HF_PORT),
           "--layout-model", st["layout_model"], "--vlm-model", st["vlm_model"],
           "--quant", st["quant"], "--max-batch", str(st["max_batch"])]
    env = {**os.environ, **_engine_env(st)}
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(LOGS / "hf-server.log", "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    (LOGS / "hf-server.pid").write_text(str(p.pid))
    print(f"Loading {st['layout_model']} + {st['vlm_model']} (log: data/logs/hf-server.log)", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _healthy(f"http://127.0.0.1:{HF_PORT}/health"):
            print(f" ready in {time.time() - t0:.0f}s")
            return
        if p.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{HF_PORT}/health", timeout=2) as r:
                pass
        except urllib.error.HTTPError as e:  # 503 while loading; error details in body
            body = json.loads(e.read() or b"{}")
            errs = {k: v for k, v in body.get("engines", {}).items() if v != "loading"}
            if errs:
                print()
                logs("hf-server")
                raise RuntimeError(f"model failed to load: {errs}")
        except OSError:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print()
    logs("hf-server")
    raise RuntimeError("inference server did not become ready")


def _image_hint(name: str, model: str) -> tuple[int, int]:
    if name == "layout":
        return IMAGE_HINT["dots" if model == DOTS_MODEL else "docling"]
    return IMAGE_HINT["vlm"]


def vllm_command(name: str, model: str, port: int, st: dict, *, util: float, max_len: int,
                 seqs: int, eager: bool) -> list[str]:
    """`vllm serve` command line for one model (loopback only)."""
    cmd = [str(VLLM_PY.parent / "vllm"), "serve", model,
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name,
           "--gpu-memory-utilization", str(util), "--max-model-len", str(max_len),
           "--max-num-seqs", str(seqs),
           # One image per request, and profile with the largest image we actually send rather
           # than the model's maximum (dots allows 11 MP = ~14k vision tokens), which otherwise
           # eats the KV-cache budget at startup ("No available memory for the cache blocks").
           "--limit-mm-per-prompt", json.dumps({"image": dict(zip(("count", "width", "height"),
                                                                 (1, *_image_hint(name, model))))}),
           "--dtype", st.get("dtype", "auto")]
    if eager:
        cmd.append("--enforce-eager")  # skip CUDA-graph capture: faster start, less memory
    if model == DOTS_MODEL:
        # Official dots.ocr serving flags; remote code pinned to a reviewed commit.
        cmd += ["--trust-remote-code", "--revision", DOTS_REVISION,
                "--chat-template-content-format", "string"]
    return cmd


def sglang_vlm_fraction(layout_util: float, vlm_util: float) -> float:
    """SGLang --mem-fraction-static for the 2nd server, relative to memory left by the 1st."""
    return round(min(0.88, vlm_util / max(0.05, 1.0 - layout_util)), 2)


def sglang_command(name: str, model: str, port: int, st: dict, *, mem_fraction: float, max_len: int,
                   seqs: int, eager: bool) -> list[str]:
    """`sglang.launch_server` command line for one model (loopback only)."""
    cmd = [str(SGLANG_PY), "-m", "sglang.launch_server", "--model-path", model,
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name,
           "--mem-fraction-static", str(mem_fraction), "--context-length", str(max_len),
           "--max-running-requests", str(seqs), "--dtype", st.get("dtype", "auto"),
           # FlashInfer JIT-compiles its sampler and attention kernels with nvcc on first use.
           # Notebook images often have no nvcc, or a conda nvcc without CUDA headers
           # ("cuda_runtime.h: No such file"), so never use FlashInfer: Triton kernels compile
           # with Triton's bundled toolchain, and SDPA is plain PyTorch.
           "--sampling-backend", "pytorch",
           "--attention-backend", "triton",
           "--mm-attention-backend", "sdpa"]
    if eager:
        cmd.append("--disable-cuda-graph")  # faster start, less memory
    if model == DOTS_MODEL:
        cmd += ["--trust-remote-code", "--revision", DOTS_REVISION,
                "--chat-template", st.get("dots_chat_template", str(DOTS_TEMPLATE))]
    return cmd


def _read_from(path: Path, offset: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def server_env(kind: str, st: dict, venv_bin: Path) -> dict[str, str]:
    """Environment for an inference server process.

    The venv isn't "activated", so put its bin/ (ninja, etc.) on PATH for any JIT toolchain
    lookups, and switch vLLM to its PyTorch/Triton sampler: FlashInfer's sampler JIT-compiles
    CUDA at startup, which fails without ninja + nvcc ("No such file or directory: 'ninja'").
    """
    env = {**os.environ, "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    if st.get("hf_home"):
        env["HF_HOME"] = st["hf_home"]
    if kind == "vllm":
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def _start_vllm_pair(st: dict, timeout: int, attempts: int = 6):
    """Start layout then narration; on out-of-memory, shrink the plan, save it, and retry."""
    for _ in range(attempts):
        which = None
        try:
            which = "layout"
            # Layout first, so the narration server's memory check sees it.
            _start_vllm("layout", st["layout_model"], LAYOUT_PORT, st, timeout,
                        util=st["layout_gpu_util"], max_len=st.get("layout_max_len", 8192),
                        seqs=st.get("layout_seqs", 16), eager=True)
            which = "vlm"
            _start_vllm("vlm", st["vlm_model"], VLM_PORT, st, timeout, util=st["vlm_gpu_util"],
                        max_len=st["max_model_len"], seqs=st["max_num_seqs"],
                        eager=st["enforce_eager"])
            return
        except ServerFailed as e:
            if not is_oom(e.log):
                raise
            smaller = shrink_plan(st, which)
            if smaller is None and which == "vlm":
                smaller = shrink_plan(st, "layout")   # last resort: free memory from layout
                which = "layout" if smaller else which
            if smaller is None:
                raise RuntimeError(f"{which} server ran out of GPU memory even at the smallest "
                                   "configuration; try a bigger GPU or engine='llama.cpp'") from e
            changes = {k: (st.get(k), v) for k, v in smaller.items() if st.get(k) != v}
            print(f"\n{which} server ran out of GPU memory - retrying with: "
                  + ", ".join(f"{k}={new}" for k, (_, new) in changes.items()))
            _kill_pidfile(f"vllm-{which}")
            if which == "layout":
                _kill_pidfile("vllm-vlm")
            if smaller["layout_model"] != st["layout_model"] or smaller["vlm_model"] != st["vlm_model"]:
                _prefetch([smaller["layout_model"], smaller["vlm_model"]], st.get("hf_home"),
                          revisions={DOTS_MODEL: DOTS_REVISION})
            st.clear()
            st.update(smaller)
            STATE.write_text(json.dumps(st, indent=1))   # remembered for the next start()
            time.sleep(5)   # let the GPU memory of the dead process be released
    raise RuntimeError("could not start vLLM within the retry budget")


def _start_vllm(name: str, model: str, port: int, st: dict, timeout: int, **kw):
    _start_server("vllm", name, port, st, timeout, vllm_command(name, model, port, st, **kw))


class ServerFailed(RuntimeError):
    def __init__(self, msg: str, log: str):
        super().__init__(msg)
        self.log = log


def _start_server(kind: str, name: str, port: int, st: dict, timeout: int, cmd: list[str]):
    label = f"{kind} {name}"
    if _healthy(f"http://127.0.0.1:{port}/health"):
        print(f"{label} already running on :{port}")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{kind}-{name}.log"
    offset = log_path.stat().st_size if log_path.exists() else 0
    env = server_env(kind, st, Path(cmd[0]).parent)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(LOGS / f"{kind}-{name}.log", "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    (LOGS / f"{kind}-{name}.pid").write_text(str(p.pid))
    model = cmd[cmd.index("--model-path") + 1] if "--model-path" in cmd else cmd[2]
    print(f"Starting {label}: {model} (log: data/logs/{kind}-{name}.log)", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _healthy(f"http://127.0.0.1:{port}/health"):
            print(f" ready in {time.time() - t0:.0f}s")
            return
        if p.poll() is not None:
            print()
            new_log = _read_from(log_path, offset)
            if not is_oom(new_log):
                logs(f"{kind}-{name}", 60)
            raise ServerFailed(f"{label} exited (code {p.returncode}); see data/logs/{kind}-{name}.log",
                               new_log)
        print(".", end="", flush=True)
        time.sleep(5)
    print()
    logs(f"{kind}-{name}", 60)
    raise RuntimeError(f"{label} did not become ready in {timeout}s")


def start(port: int = 8000, data_dir: str | None = None, extra_hosts: list[str] | None = None,
          voice: str | None = None):
    """Start inference + the backend (all on 127.0.0.1), then show a link to the UI.

    data_dir:    where processed papers live (e.g. a mounted Google Drive folder to persist them)
    extra_hosts: additional Host headers to accept, if your proxy uses another hostname
    """
    global _backend, _port
    _port = port
    st = _state()
    start_inference()

    url, hosts = _proxy(port)
    env = {**_engine_env(st), "PR_PORT": str(port), "PR_HOST": "127.0.0.1",
           "PR_ALLOWED_HOSTS": ",".join(["127.0.0.1", "localhost", *hosts, *(extra_hosts or [])])}
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        env["PR_DATA_DIR"] = str(Path(data_dir).resolve())
    if voice:
        env["PR_VOICE"] = voice

    _stop_backend()
    LOGS.mkdir(parents=True, exist_ok=True)
    _backend = subprocess.Popen([_py(st), "app.py"], cwd=ROOT / "backend", env={**os.environ, **env},
                                stdout=open(LOGS / "backend.log", "a"), stderr=subprocess.STDOUT,
                                start_new_session=True)
    (LOGS / "backend.pid").write_text(str(_backend.pid))
    for _ in range(60):
        if _healthy(f"http://127.0.0.1:{port}/api/papers"):
            break
        if _backend.poll() is not None:
            logs("backend")
            raise RuntimeError("backend exited")
        time.sleep(1)
    else:
        raise RuntimeError(f"backend did not come up; see {LOGS / 'backend.log'}")
    print(f"Backend ready on 127.0.0.1:{port} (data: {env.get('PR_DATA_DIR', 'data/papers')})")
    show(port, url)


def show(port: int | None = None, url: str | None = None):
    """Display a link to the UI (a clickable link in notebooks, plain text in a terminal)."""
    try:
        from IPython.display import HTML, display
        in_nb = "ipykernel" in sys.modules
    except ImportError:
        in_nb = False
    port = port or _port
    if platform() == "colab":
        # Print the proxy URL as plain text: Colab's JavaScript link widget
        # (serve_kernel_port_as_window) doesn't render in every browser/setup.
        if url is None:
            try:
                url, _ = _proxy(port)
            except Exception as e:  # eval_js needs the Colab frontend (e.g. fails from a thread)
                print(f"Could not get the Colab proxy URL ({e}). Re-run cloud.show() in its own cell.")
                return
        print(f"Paper Reader: {url}")
        print("Open it in a new tab of the browser you're using for Colab (the link is tied to your\n"
              "Google login and this runtime; it stops working when the runtime ends).")
        if in_nb:
            display(HTML(f'<p><a href="{html.escape(url)}" target="_blank" rel="noopener">'
                         f'▶ Open Paper Reader</a></p>'))
        return
    if url is None:
        url, _ = _proxy(port)
    if url and not in_nb:
        print(f"Paper Reader: {url}")
    elif url:
        display(HTML(f'<p><b>Paper Reader:</b> <a href="{html.escape(url)}" target="_blank">'
                     f'{html.escape(url)}</a> (opens through the authenticated Jupyter proxy; keep the '
                     f'trailing slash)</p>'))
    else:
        print(f"UI is on 127.0.0.1:{port} inside this machine. Reach it through your platform's "
              f"proxy (…/proxy/{port}/) or an SSH tunnel: ssh -L {port}:127.0.0.1:{port} <host>")


def status():
    st = _state()
    env = _engine_env(st)
    checks = [("layout", (env.get("PR_LAYOUT_URL") or "http://127.0.0.1:8081") + "/health"),
              ("narration", (env.get("PR_VLM_URL") or "http://127.0.0.1:8082") + "/health"),
              ("backend", f"http://127.0.0.1:{_port}/api/papers")]
    for name, url in checks:
        print(f"{'✓' if _healthy(url) else '✗'} {name:10s} {url}")
    print("settings:", json.dumps(st or {"engine": "not set up"}))


def _kill_pidfile(name: str):
    f = LOGS / f"{name}.pid"
    if f.exists():
        try:
            os.killpg(int(f.read_text()), 15)
        except (ProcessLookupError, ValueError, PermissionError):
            try:
                os.kill(int(f.read_text()), 15)
            except (ProcessLookupError, ValueError):
                pass
        f.unlink(missing_ok=True)


def _stop_backend():
    global _backend
    if _backend and _backend.poll() is None:
        _backend.terminate()
        try:
            _backend.wait(10)
        except subprocess.TimeoutExpired:
            _backend.kill()
    else:
        _kill_pidfile("backend")
    _backend = None


def stop():
    """Stop the backend and inference servers (frees the GPU)."""
    _stop_backend()
    for name in ("hf-server", "vllm-vlm", "vllm-layout", "sglang-vlm", "sglang-layout"):
        _kill_pidfile(name)
    _sh(["bash", "scripts/stop_llama.sh"], check=False)
    print("Stopped.")


# ------------------------------------------------------------------ batch + transfer

def process(inputs: list[str], parallel: int = 2, export: bool = True, audio: str = "mp3",
            data_dir: str | None = None):
    """Batch-process PDFs / dirs / arXiv refs / @lists headlessly (starts inference if needed)."""
    st = _state()
    env = _engine_env(st)
    if data_dir:
        env["PR_DATA_DIR"] = str(Path(data_dir).resolve())
    start_inference()
    cmd = [_py(st), "backend/batch.py", "process", *inputs, "--parallel", str(parallel)]
    if export:
        cmd += ["--export", str(ROOT / "exports"), "--audio", audio]
    _sh(cmd, env=env, check=False)


def download_bundles(audio: str = "mp3", data_dir: str | None = None) -> Path:
    """Export every processed paper and (on Colab) download one .tar of bundles to your computer.
    Import at home with: cd backend && uv run python batch.py import <file-or-dir>"""
    st = _state()
    env = _engine_env(st)
    if data_dir:
        env["PR_DATA_DIR"] = str(Path(data_dir).resolve())
    out = ROOT / "exports"
    _sh([_py(st), "backend/batch.py", "export", "--all", "--out", str(out), "--audio", audio], env=env)
    bundles = sorted(out.glob("*.tar.gz"))
    if not bundles:
        print("No processed papers yet.")
        return out
    pack = ROOT / "paper-reader-bundles.tar"
    subprocess.run(["tar", "-cf", str(pack), "-C", str(out), *[b.name for b in bundles]], check=True)
    print(f"{len(bundles)} bundle(s) -> {pack} ({pack.stat().st_size / 1e6:.0f} MB)")
    if platform() == "colab":
        from google.colab import files  # type: ignore
        files.download(str(pack))
    else:
        print("Download it from the Jupyter file browser, then at home:\n"
              "  tar -xf paper-reader-bundles.tar -C some_dir && "
              "(cd backend && uv run python batch.py import ../some_dir)")
    return pack


def logs(name: str = "backend", lines: int = 40):
    """Tail a log: backend | vllm-vlm | vllm-layout | sglang-vlm | sglang-layout | hf-server | llama-*."""
    f = LOGS / f"{name}.log"
    print("".join(f.read_text(errors="replace").splitlines(True)[-lines:]) if f.exists() else f"no {f}")


if __name__ == "__main__":  # minimal CLI for terminals on the same machines
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"setup": setup, "start": start, "stop": stop, "status": status}[cmd]()
    if cmd == "start":
        print("Servers run in the background. Stop them with: python notebooks/cloud.py stop")
