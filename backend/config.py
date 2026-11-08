"""Runtime configuration (env-overridable). All listeners default to loopback."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PR_DATA_DIR", ROOT / "data" / "papers"))
MODELS_DIR = Path(os.environ.get("PR_MODELS_DIR", ROOT / "models"))

HOST = os.environ.get("PR_HOST", "127.0.0.1")
PORT = int(os.environ.get("PR_PORT", "8000"))
# Host headers the backend accepts (DNS-rebinding guard). Loopback by default; notebook launchers
# add their authenticated proxy host (e.g. Colab / SageMaker Jupyter proxy). "*.domain" wildcards ok.
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("PR_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
                 if h.strip()]
if "*" in ALLOWED_HOSTS:
    raise SystemExit("PR_ALLOWED_HOSTS='*' disables the Host check; list the proxy host explicitly.")

# llama-server endpoints (OpenAI-compatible)
LAYOUT_URL = os.environ.get("PR_LAYOUT_URL", "http://127.0.0.1:8081")
VLM_URL = os.environ.get("PR_VLM_URL", "http://127.0.0.1:8082")
# Model names sent in requests (needed by vLLM/SGLang-style servers; llama-server ignores them).
LAYOUT_MODEL_NAME = os.environ.get("PR_LAYOUT_MODEL_NAME") or None
VLM_MODEL_NAME = os.environ.get("PR_VLM_MODEL_NAME") or None
# Max new tokens for one page of DocTags. granite-docling's context is 8192 (prompt included), so
# servers that enforce it (vLLM) need this lowered to ~6500.
LAYOUT_MAX_TOKENS = int(os.environ.get("PR_LAYOUT_MAX_TOKENS", "8192"))
# Downscale images above this many pixels before sending (0 = never). Bounds vision tokens.
MAX_IMAGE_PIXELS = int(os.environ.get("PR_MAX_IMAGE_PIXELS", "0"))
# Qwen3.x "thinking" models reason before answering; narration doesn't need it. Sends
# chat_template_kwargs={"enable_thinking": false} (ignored by templates without the switch).
VLM_DISABLE_THINKING = os.environ.get("PR_VLM_DISABLE_THINKING", "1") != "0"
# dots.ocr: put the image placeholder in the prompt text ourselves (vLLM recipe, text-only chat
# template). SGLang runs with the processor chat template, which emits it per image part: set 0.
DOTS_IMAGE_PLACEHOLDER = os.environ.get("PR_DOTS_IMAGE_PLACEHOLDER", "1") != "0"
# "docling" (granite-docling DocTags), "dots" (dots.ocr JSON layout) or "qwen" (VLM grounding)
LAYOUT_MODEL = os.environ.get("PR_LAYOUT_MODEL", "docling")

KOKORO_MODEL = Path(os.environ.get("PR_KOKORO_MODEL", MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx"))
KOKORO_VOICES = Path(os.environ.get("PR_KOKORO_VOICES", MODELS_DIR / "kokoro" / "voices-v1.0.bin"))
KOKORO_VOICE = os.environ.get("PR_VOICE", "af_heart")

MAX_PDF_BYTES = int(os.environ.get("PR_MAX_PDF_MB", "60")) * 1024 * 1024
# px of page image sent to the layout model (dots.ocr is trained on ~200 dpi pages: use ~1600)
LAYOUT_LONG_SIDE = int(os.environ.get("PR_LAYOUT_LONG_SIDE", "1024"))
CROP_DPI = 200            # DPI for figure/table/equation crops sent to the VLM
LLM_TIMEOUT = 600.0

# Concurrent LLM requests; keep equal to llama-server --parallel (scripts read the same PR_NP).
CONCURRENCY = int(os.environ.get("PR_NP", "2"))
# onnxruntime providers for Kokoro, comma-separated. "auto" = CUDA if available, else CPU.
TTS_PROVIDERS = os.environ.get("PR_TTS_PROVIDERS", "auto")
