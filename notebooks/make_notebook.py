"""Regenerates paper_reader_cloud.ipynb: python notebooks/make_notebook.py notebooks/paper_reader_cloud.ipynb"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# 📄🔊 Paper Reader — Colab / SageMaker

> **On Google Colab?** Use the Colab notebook instead: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vismay93/LiteratureReader/blob/master/notebooks/paper_reader_colab.ipynb)

Runs the full Paper Reader UI on a cloud GPU. You upload or fetch a paper, and it is read aloud while the
paragraph, figure, table or equation being read is highlighted.

| | |
|---|---|
| **Inference** | **vLLM**, installed from prebuilt wheels into an isolated venv with uv, so there's nothing to compile. Models load straight from Hugging Face. **Layout/OCR:** `dots-studio/dots.ocr` on GPUs with ≥ 20 GB; granite-docling on smaller ones. **Narration:** `Qwen/Qwen3.8-27B-FP8` on 80 GB GPUs; `Qwen3-VL` 8B/4B/2B on smaller ones |
| **Speech** | Kokoro-82M on the GPU (onnxruntime-gpu) |
| **Install time** | About 3–6 min on first run: vLLM wheels plus model downloads. Re-running is idempotent |
| **GPU** | Any NVIDIA GPU with 12 GB or more: Colab T4/L4/A100; SageMaker `ml.g4dn` / `ml.g5` / `ml.g6` / `ml.p4d` |

**Colab:** *Runtime → Change runtime type → GPU*.
**SageMaker:** use a GPU notebook instance or a JupyterLab space with **≥ 30 GB** of storage. Models go to `/tmp` by default.

> 🔒 **Access:** every server binds to `127.0.0.1` inside the VM. You reach the UI only through the platform's
> authenticated proxy: Colab's kernel-port proxy (your Google account) or SageMaker's Jupyter proxy (your AWS
> session). No public tunnel such as ngrok is opened. Don't add one: the app has no login of its own.
""")

md("## 1. Get the code")
code('''
import os, sys, subprocess, importlib
from pathlib import Path

REPO_URL = "https://github.com/vismay93/LiteratureReader.git"   # your fork, or "" to use an uploaded archive
ARCHIVE  = ""   # path to an uploaded paper-reader-*.tar.gz; on Colab, leave both empty to upload now

IN_COLAB = "google.colab" in sys.modules
if (Path.cwd() / "cloud.py").exists():                 # notebook opened from inside the repo
    APP = Path.cwd().parent
else:
    BASE = Path("/content") if IN_COLAB else (Path.home() / "SageMaker" if (Path.home() / "SageMaker").is_dir() else Path.home())
    APP = BASE / "paper-reader"
    if not (APP / "notebooks" / "cloud.py").exists():
        if REPO_URL:
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(APP)], check=True)
        else:
            if not ARCHIVE:
                if not IN_COLAB:
                    raise SystemExit("Set REPO_URL, or upload paper-reader-*.tar.gz and set ARCHIVE.")
                from google.colab import files
                ARCHIVE = next(iter(files.upload()))
            subprocess.run(["tar", "-xzf", ARCHIVE, "-C", str(BASE)], check=True)

sys.path.insert(0, str(APP / "notebooks"))
import cloud; importlib.reload(cloud)
print("App:", APP, "| platform:", cloud.platform())
''')

md("""
### (Optional) Keep processed papers across sessions
Colab VMs are wiped when the runtime ends. To keep papers, store them in Google Drive.
On SageMaker, `~/SageMaker` persists on notebook instances.
""")
code('''
DATA_DIR = None                     # None = <app>/data/papers (ephemeral on Colab)
PERSIST_TO_DRIVE = False
if IN_COLAB and PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = "/content/drive/MyDrive/paper-reader/papers"
''')

md("""
## 2. Install (vLLM + models from Hugging Face + UI)
vLLM goes into its own venv (`.venv-vllm`), so the kernel's torch is untouched. The build is chosen to match your NVIDIA driver:
CUDA 13, or the CUDA 12.9 build for older drivers. The backend's small dependencies go into this kernel. Models are prefetched into the Hugging Face cache.

| VRAM | Layout / OCR | Narration |
|---|---|---|
| ≥ 70 GB (A100-80GB, H100) | dots.ocr | Qwen3.8-27B-FP8 |
| ≥ 38 GB (A100-40GB, L40S) | dots.ocr | Qwen3-VL-8B-Instruct |
| ≥ 20 GB (L4 / A10G) | dots.ocr | Qwen3-VL-4B-Instruct-FP8 |
| ≥ 12 GB (T4) | granite-docling | Qwen3-VL-4B-Instruct (fp16) |
| smaller | granite-docling | Qwen3-VL-2B-Instruct |

| Option | Purpose |
|---|---|
| `layout="dots"` / `layout="docling"` | Force a layout model (default `"auto"`) |
| `vlm_model="Qwen/Qwen3-VL-2B-Instruct"` | Smaller or faster narration model, e.g. if a T4 runs out of memory |
| `max_batch=16` | Max concurrent sequences for the narration model |
| `hf_home="/content/drive/MyDrive/hf-cache"` | Keep downloaded models on Drive between Colab sessions |
| `engine="sglang"` | Serve the same models with SGLang (needs an NVIDIA driver ≥ 580 / CUDA 13 and glibc ≥ 2.34; Colab qualifies if its driver is new enough) |
| `engine="transformers"` or `engine="llama.cpp"` | Alternative engines (slower, or GGUF) |
""")
code('''
cloud.setup(
    # engine="sglang",                     # "vllm" (default) | "sglang" | "transformers" | "llama.cpp"
    # layout="dots",                       # "auto" | "dots" | "docling"
    # vlm_model="Qwen/Qwen3-VL-2B-Instruct",
    # max_batch=16,
    # hf_home="/content/drive/MyDrive/hf-cache",
)
''')

md("""
## 3. Start the reader
Starts two vLLM servers, one for layout and one for narration (1–4 min the first time), plus the backend, then shows a link. If a server runs out of GPU memory, it automatically retries with a smaller configuration and remembers it. Open it in a new tab, then drop a PDF or paste
an arXiv ID. Speed depends on the GPU; A100/L4 are several times faster than a T4.
""")
code('''
cloud.start(data_dir=DATA_DIR)      # voice="am_michael" for a different Kokoro voice
''')

md("""
## 4. (Optional) Batch-process a reading list
This runs without the UI. Inputs can be arXiv IDs or URLs, PDF paths, folders, or `@list.txt`. Results show up in the
reader's library, and bundles are written to `exports/`.
""")
code('''
cloud.process(["1706.03762", "2010.11929"], parallel=2, data_dir=DATA_DIR)
''')

md("""
## 5. (Optional) Take papers home
Exports every processed paper as a compact bundle (MP3 audio) and downloads one `.tar` to your computer.
To import it on your laptop:
```bash
mkdir -p data/imports && tar -xf paper-reader-bundles.tar -C data/imports
cd backend && uv run python batch.py import ../data/imports
```
""")
code('''
cloud.download_bundles(data_dir=DATA_DIR)
''')

md("## Troubleshooting")
code('''
cloud.status()
cloud.logs("vllm-vlm")         # also "vllm-layout", "sglang-vlm", "sglang-layout", "backend"
''')
code('''
cloud.show()                   # show the UI link again
''')
code('''
cloud.stop()                   # stop all servers (frees the GPU)
''')

nb = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
json.dump(nb, open(sys.argv[1], "w"), indent=1)
