"""Regenerates the Colab-only notebook:
    python notebooks/make_colab_notebook.py notebooks/paper_reader_colab.ipynb

Plain code cells only: no #@param forms, no ipywidgets, no JavaScript output widgets.
Everything (including the UI link) is printed as text, so it shows up in every browser.
"""
import json, sys

REPO = "vismay93/LiteratureReader"
BRANCH = "master"
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md(f"""
# 📄🔊 Paper Reader on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/paper_reader_colab.ipynb)

Listen to research papers while the paragraph, figure, table or equation being read is highlighted on the PDF, on a Colab GPU.

1. **Runtime → Change runtime type → T4 GPU** (or L4 / A100).
2. Run the cells **top to bottom**, one at a time (Shift+Enter). Each prints what it did. Nothing here depends on forms or widgets.
3. Step 4 prints a **Paper Reader link**. Open it in a new tab of this same browser.

First run: about 5–10 min (vLLM wheels plus model downloads). Colab wipes the VM when the runtime ends; turn on `USE_DRIVE` in step 1 to keep your papers.

> 🔒 Every server listens on `127.0.0.1` inside the VM. The link goes through Colab's port proxy, which only works for your Google account while this runtime is running. Don't add a public tunnel (ngrok etc.): the app has no login.
""")

md("## 0. Check the GPU")
code('''
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("No GPU. Runtime -> Change runtime type -> T4 GPU, then run this cell again.")
print("GPU:", r.stdout.strip())
''')

md("""
## 1. Settings
Edit the values below if you want, then run the cell.
""")
code(f'''
REPO_URL = "https://github.com/{REPO}.git"   # your fork, if you have one
BRANCH   = "{BRANCH}"

USE_DRIVE  = False     # True: keep processed papers in Google Drive across sessions (asks for Drive access)
DRIVE_DIR  = "/content/drive/MyDrive/paper-reader"
CACHE_MODELS_ON_DRIVE = False   # True: also keep the Hugging Face model cache on Drive (~10-20 GB; faster restarts)

ENGINE = "vllm"        # "vllm" (fastest) | "transformers" (most compatible; use if vLLM fails) | "llama.cpp"
VOICE  = "af_heart"    # Kokoro voice: af_heart, am_michael, bf_emma, bm_george, ...
PORT   = 8000
print("Settings saved. Engine:", ENGINE, "| Drive:", USE_DRIVE)
''')

md("## 2. Get the code (and mount Drive if enabled)")
code('''
import os, sys, importlib
from pathlib import Path

APP = Path("/content/paper-reader")
if (APP / ".git").exists():
    subprocess.run(["git", "-C", str(APP), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(APP)], check=True)
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
print("Code in:", APP, "| commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
      capture_output=True, text=True).stdout.strip())
print("Papers stored in:", DATA_DIR or f"{APP}/data/papers (temporary)")
''')

md("""
## 3. Install
Installs the inference engine, the backend, the models and the web UI. Re-running skips finished steps.
If it runs out of GPU memory later, come back and pass `vlm_model="Qwen/Qwen3-VL-2B-Instruct"`.
""")
code('''
cloud.setup(engine=ENGINE, hf_home=HF_HOME)
''')

md("""
## 4. Start the reader and get the link
Starts the model servers (1–4 min the first time) and the backend, then prints the link.
Drop a PDF on the page, or paste an arXiv ID such as `1706.03762`.
""")
code('''
cloud.start(port=PORT, data_dir=DATA_DIR, voice=VOICE)
''')

md("""
**Link didn't appear, or stopped working?** Run the next cell to print it again.

**Prefer it inside the notebook?** Run the embed cell. A new tab works better for audio and keyboard shortcuts, though.
""")
code('''
cloud.show(PORT)
''')
code('''
from google.colab import output
output.serve_kernel_port_as_iframe(PORT, path="/", height=900)
''')

md("""
## 5. (Optional) Batch-process a reading list
Runs without the UI. Inputs can be arXiv IDs or URLs, PDF paths, or folders. Results show up in the reader's library.
""")
code('''
cloud.process(["1706.03762", "2010.11929"], parallel=2, data_dir=DATA_DIR)
''')

md("""
## 6. (Optional) Download papers to your computer
Downloads one `.tar` of processed papers. To import it on your Mac:
```bash
mkdir -p data/imports && tar -xf ~/Downloads/paper-reader-bundles.tar -C data/imports
cd backend && uv run python batch.py import ../data/imports
```
""")
code('''
cloud.download_bundles(data_dir=DATA_DIR)
''')

md("## Troubleshooting")
code('''
cloud.status()            # which servers are up
cloud.logs("backend")     # also: "vllm-vlm", "vllm-layout", "hf-server"
''')
md("""
| Problem | Fix |
|---|---|
| vLLM fails to start on a T4 | Set `ENGINE = "transformers"` in step 1, then run steps 1, 3 and 4 again |
| Out of GPU memory | Run `cloud.setup(engine=ENGINE, hf_home=HF_HOME, vlm_model="Qwen/Qwen3-VL-2B-Instruct")`, then step 4 |
| Link shows "Invalid host header" | Run `cloud.start(port=PORT, data_dir=DATA_DIR, extra_hosts=["<host from the link>"])` |
| Link stopped working | The runtime restarted or timed out. Run all cells again (Drive keeps your papers if `USE_DRIVE` is on) |
""")
code('''
cloud.stop()              # stop all servers and free the GPU
''')

nb = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i}"
json.dump(nb, open(sys.argv[1], "w"), indent=1)
