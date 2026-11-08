"""Regenerates the Colab-only notebook:
    python notebooks/make_colab_notebook.py notebooks/paper_reader_colab.ipynb

Every code cell is a Colab form: the code is hidden ("cellView": "form") and the inputs are
#@param fields. The notebook assumes Google Colab on a T4 GPU (16 GB) and sizes models for it.
"""
import json, sys

REPO = "vismay93/LiteratureReader"
BRANCH = "master"
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def form(s):
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(True)})

md(f"""
# 📄🔊 Paper Reader on Google Colab (T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/paper_reader_colab.ipynb)

Listen to research papers while the paragraph, figure, table or equation being read is highlighted on the PDF.

1. **Runtime → Change runtime type → T4 GPU.**
2. Fill in the forms and press ▶ on each cell, top to bottom. To see a cell's code, double-click it.
3. Step 4 prints a **Paper Reader** link. Open it in a new tab of this browser.

Tuned for a **T4 (16 GB)**: granite-docling for layout, Qwen3-VL-4B for narration, served by vLLM in fp16.
First run: about 5–10 min. Colab wipes the VM when the runtime ends; tick **USE_DRIVE** to keep your papers.

> 🔒 Every server listens on `127.0.0.1` inside the VM. The link goes through Colab's port proxy, which works only for your Google account while this runtime is running. Don't add a public tunnel (ngrok etc.): the app has no login.
""")

form('''
#@title 1. Settings { display-mode: "form" }
#@markdown **Storage**
USE_DRIVE = False  #@param {type:"boolean"}
#@markdown Keep processed papers in Google Drive across sessions (asks for Drive access).
DRIVE_DIR = "/content/drive/MyDrive/paper-reader"  #@param {type:"string"}
CACHE_MODELS_ON_DRIVE = False  #@param {type:"boolean"}
#@markdown Also keep the downloaded models (~10 GB) on Drive as one archive, so a new session skips the download. Models are always served from local disk.

#@markdown ---
#@markdown **Models (T4)**
ENGINE = "vllm"  #@param ["vllm", "transformers", "llama.cpp"]
#@markdown `vllm` is fastest. Use `transformers` if vLLM won't start; `llama.cpp` uses the desktop app's GGUF models.
NARRATION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  #@param ["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"]
#@markdown Pick 2B if the T4 runs out of memory.
MAX_BATCH = 8  #@param {type:"slider", min:1, max:16, step:1}

#@markdown ---
#@markdown **Voice**
VOICE = "af_heart"  #@param ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

#@markdown ---
#@markdown **Code** (change only if you use a fork)
REPO_URL = "https://github.com/''' + REPO + '''.git"  #@param {type:"string"}
BRANCH = "''' + BRANCH + '''"  #@param {type:"string"}

import os, subprocess
os.environ["PR_PLATFORM"] = "colab"
PORT = 8000

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True)
if gpu.returncode != 0:
    raise SystemExit("No GPU. Runtime -> Change runtime type -> T4 GPU, then run this cell again.")
print("GPU:", gpu.stdout.strip())
if "T4" not in gpu.stdout:
    print("Note: this notebook is tuned for a T4; it will still run with the T4 settings.")
print(f"Engine: {ENGINE} | narration: {NARRATION_MODEL} | batch: {MAX_BATCH} | voice: {VOICE} | Drive: {USE_DRIVE}")
''')

form('''
#@title 2. Get the code { display-mode: "form" }
#@markdown Clones the repo into `/content/paper-reader` (or updates it) and mounts Google Drive if enabled.
import sys, importlib
from pathlib import Path

APP = Path("/content/paper-reader")
if (APP / ".git").exists():
    subprocess.run(["git", "-C", str(APP), "pull", "--ff-only", "-q"], check=False)
else:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO_URL, str(APP)], check=True)
os.chdir(APP)

DATA_DIR = None
HF_HOME = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = f"{DRIVE_DIR}/papers"
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    if CACHE_MODELS_ON_DRIVE:
        HF_HOME = f"{DRIVE_DIR}/hf-cache"

sys.path.insert(0, str(APP / "notebooks"))
import cloud; importlib.reload(cloud)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"Code: {APP} @ {commit}")
print("Papers stored in:", DATA_DIR or f"{APP}/data/papers (deleted when the runtime ends)")
''')

form('''
#@title 3. Install { display-mode: "form" }
#@markdown Installs the inference engine, backend, models and web UI. About 5–10 min the first time; re-running skips finished steps.
cloud.setup(engine=ENGINE, layout="docling", vlm_model=NARRATION_MODEL,
            max_batch=MAX_BATCH, hf_home=HF_HOME)
''')

form('''
#@title 4. Start the reader { display-mode: "form" }
#@markdown Starts the model servers (1–4 min) and the backend, then prints the link.
#@markdown Drop a PDF on the page, or paste an arXiv ID such as `1706.03762`.
cloud.start(port=PORT, data_dir=DATA_DIR, voice=VOICE)
''')

form('''
#@title Show the link again { display-mode: "form" }
#@markdown Run this if the link didn't appear or you lost it.
EMBED_IN_NOTEBOOK = False  #@param {type:"boolean"}
#@markdown Tick to show the reader inside this notebook instead (a new tab works better for audio and keyboard shortcuts).
if EMBED_IN_NOTEBOOK:
    from google.colab import output
    output.serve_kernel_port_as_iframe(PORT, path="/", height=900)
else:
    cloud.show(PORT)
''')

form('''
#@title 5. (Optional) Batch-process papers { display-mode: "form" }
#@markdown Processes papers without the UI; they then appear in the reader's library.
PAPERS = "1706.03762 2010.11929"  #@param {type:"string"}
#@markdown Space-separated arXiv IDs or URLs, or paths to PDFs/folders (e.g. on Drive).
PARALLEL = 2  #@param {type:"slider", min:1, max:4, step:1}
cloud.process(PAPERS.split(), parallel=PARALLEL, data_dir=DATA_DIR)
''')

form('''
#@title 6. (Optional) Download papers to your computer { display-mode: "form" }
#@markdown Downloads one `paper-reader-bundles.tar`. To import it on your Mac:
#@markdown `mkdir -p data/imports && tar -xf ~/Downloads/paper-reader-bundles.tar -C data/imports && (cd backend && uv run python batch.py import ../data/imports)`
cloud.download_bundles(data_dir=DATA_DIR)
''')

form('''
#@title Troubleshooting: status and logs { display-mode: "form" }
LOG = "backend"  #@param ["backend", "vllm-vlm", "vllm-layout", "hf-server", "llama-docling", "llama-qwen"]
LINES = 40  #@param {type:"integer"}
cloud.status()
print()
cloud.logs(LOG, LINES)
''')

md("""
| Problem | Fix |
|---|---|
| vLLM won't start | In step 1 set **ENGINE** to `transformers`, then run steps 1, 3 and 4 again |
| `SafetensorError: ... header too large` | A model file was corrupt. Run step 3 again: it checks every weight file and re-downloads bad ones |
| Out of GPU memory | In step 1 set **NARRATION_MODEL** to the 2B model or lower **MAX_BATCH**, then run steps 1, 3 and 4 again |
| Link shows "Invalid host header" | Run step 4 again (it re-reads Colab's proxy address) |
| Link stopped working | The runtime restarted or timed out. Run every step again; with **USE_DRIVE** your papers are kept |
""")

form('''
#@title Stop everything { display-mode: "form" }
#@markdown Stops all servers and frees the GPU.
cloud.stop()
''')

nb = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
