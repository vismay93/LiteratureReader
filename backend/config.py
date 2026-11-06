"""Runtime configuration (env-overridable). All listeners default to loopback."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PR_DATA_DIR", ROOT / "data" / "papers"))
MODELS_DIR = Path(os.environ.get("PR_MODELS_DIR", ROOT / "models"))

HOST = os.environ.get("PR_HOST", "127.0.0.1")
PORT = int(os.environ.get("PR_PORT", "8000"))

# llama-server endpoints (OpenAI-compatible)
LAYOUT_URL = os.environ.get("PR_LAYOUT_URL", "http://127.0.0.1:8081")
VLM_URL = os.environ.get("PR_VLM_URL", "http://127.0.0.1:8082")
# "docling" (granite-docling DocTags) or "qwen" (Qwen3-VL JSON grounding fallback)
LAYOUT_MODEL = os.environ.get("PR_LAYOUT_MODEL", "docling")

KOKORO_MODEL = Path(os.environ.get("PR_KOKORO_MODEL", MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx"))
KOKORO_VOICES = Path(os.environ.get("PR_KOKORO_VOICES", MODELS_DIR / "kokoro" / "voices-v1.0.bin"))
KOKORO_VOICE = os.environ.get("PR_VOICE", "af_heart")

MAX_PDF_BYTES = int(os.environ.get("PR_MAX_PDF_MB", "60")) * 1024 * 1024
LAYOUT_LONG_SIDE = 1024   # px of page image sent to the layout model
CROP_DPI = 200            # DPI for figure/table/equation crops sent to the VLM
LLM_TIMEOUT = 600.0

# Concurrent LLM requests; keep equal to llama-server --parallel (scripts read the same PR_NP).
CONCURRENCY = int(os.environ.get("PR_NP", "2"))
# onnxruntime providers for Kokoro, comma-separated. "auto" = CUDA if available, else CPU.
TTS_PROVIDERS = os.environ.get("PR_TTS_PROVIDERS", "auto")
