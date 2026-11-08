"""Background processing: PDF -> layout -> narration -> audio -> doc.json.

Every stage caches its output under data/papers/<id>/ so a crashed or restarted job
resumes where it left off, and a re-upload of the same PDF is instant.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path

import httpx

import config
from . import doctags, layout, narrate, pdf, tts
from .ingest import paper_dir

log = logging.getLogger("paper-reader.jobs")

VISUAL = ("figure", "chart", "table", "equation")
TEXTUAL = ("text", "list_item", "footnote", "heading", "title", "caption", "code")


class JobManager:
    def __init__(self):
        self.status: dict[str, dict] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def start(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    def get(self, pid: str) -> dict:
        if pid in self.status:
            return self.status[pid]
        d = paper_dir(pid)
        if (d / "doc.json").exists():
            return {"state": "done", "stage": "done", "done": 1, "total": 1}
        f = d / "status.json"
        if f.exists():
            return json.loads(f.read_text())
        return {"state": "unknown"}

    def submit(self, pid: str, force: bool = False):
        d = paper_dir(pid)
        if (d / "doc.json").exists() and not force:
            return
        if self.status.get(pid, {}).get("state") in ("queued", "running"):
            return
        self._set(pid, state="queued", stage="queued", done=0, total=0, message="Waiting in queue")
        self.queue.put_nowait(pid)

    def _set(self, pid: str, **kw):
        st = self.status.setdefault(pid, {})
        st.update(kw, updated=time.time())
        try:
            (paper_dir(pid) / "status.json").write_text(json.dumps(st))
        except OSError:
            pass

    async def _run(self):
        while True:
            pid = await self.queue.get()
            try:
                self._set(pid, state="running")
                await process(pid, lambda **kw: self._set(pid, **kw))
                self._set(pid, state="done", stage="done", message="Ready")
            except Exception as e:  # keep worker alive
                log.exception("job %s failed", pid)
                self._set(pid, state="error", message=f"{type(e).__name__}: {e}",
                          trace=traceback.format_exc()[-2000:])


async def process(pid: str, report, sem: asyncio.Semaphore | None = None):
    """Run the full pipeline for one paper. Pass a shared `sem` to bound LLM calls across papers."""
    d = paper_dir(pid)
    src = d / "source.pdf"
    for sub in ("pages", "layout", "crops", "narration", "audio"):
        (d / sub).mkdir(exist_ok=True)

    async with httpx.AsyncClient() as client:
        await _check_servers(client)
        n = pdf.page_count(src)

        # 1. Render pages for the layout model.
        report(stage="render", done=0, total=n, message="Rendering pages")
        for i in range(n):
            img = d / "pages" / f"{i}.png"
            if not img.exists():
                await asyncio.to_thread(pdf.render_page, src, i, img)
            report(done=i + 1)

        # 2. Layout + OCR + reading order per page.
        report(stage="layout", done=0, total=n, message="Analyzing layout")
        sem = sem or asyncio.Semaphore(config.CONCURRENCY)
        page_blocks: list[list[dict]] = [[] for _ in range(n)]
        done = 0

        async def do_page(i: int):
            nonlocal done
            cache = d / "layout" / f"{i}.json"
            if cache.exists():
                page_blocks[i] = json.loads(cache.read_text())
            else:
                async with sem:
                    blocks, raw = await layout.analyze_page(client, d / "pages" / f"{i}.png", i)
                (d / "layout" / f"{i}.raw.txt").write_text(raw)
                cache.write_text(json.dumps(blocks))
                page_blocks[i] = blocks
            done += 1
            report(done=done, message=f"Analyzing layout (page {done}/{n})")

        await asyncio.gather(*(do_page(i) for i in range(n)))

        # 3. Prefer the PDF text layer over OCR for born-digital papers.
        tl = await asyncio.to_thread(pdf.TextLayer, src)
        blocks: list[dict] = []
        for i, pb in enumerate(page_blocks):
            for b in pb:
                if tl.has_text(i) and b["type"] in TEXTUAL:
                    t = tl.text_in(i, b["bbox"])
                    if t and len(t) >= 0.5 * len(b.get("text", "")):
                        b["ocr_text"] = b.get("text", "")
                        b["text"] = t
                blocks.append(b)
        blocks = doctags.postprocess(blocks)

        # 4. Crops for visual blocks.
        for b in blocks:
            if b["type"] in VISUAL:
                c = d / "crops" / f"{b['id']}.png"
                if not c.exists():
                    await asyncio.to_thread(pdf.crop_region, src, b["page"], b["bbox"], c)
                b["crop"] = f"crops/{b['id']}.png"

        # 5+6. Narration (LLM) pipelined with TTS (CPU).
        todo = [i for i, b in enumerate(blocks) if not b.get("skip")]
        report(stage="narrate", done=0, total=len(todo), message="Writing narration and speech")
        tts_q: asyncio.Queue = asyncio.Queue()
        progress = {"n": 0}

        async def narrate_one(i: int):
            b = blocks[i]
            cache = d / "narration" / f"{b['id']}.json"
            res = json.loads(cache.read_text()) if cache.exists() else None
            if res is not None and b["type"] in narrate.TEXT_TYPES \
                    and res.get("src", "llm") != narrate.text_source(b):
                # PR_REWRITE_TEXT changed since this block was narrated: redo it and its audio.
                res = None
                for f in (d / "audio" / f"{b['id']}.json", d / "audio" / f"{b['id']}.wav"):
                    f.unlink(missing_ok=True)
            if res is None:
                crop = d / b["crop"] if b.get("crop") else None
                if narrate.uses_llm(b):
                    async with sem:
                        res = await narrate.narrate_block(client, blocks, i, crop)
                else:   # headings and rule-cleaned text: no model call, don't queue for a slot
                    res = await narrate.narrate_block(client, blocks, i, crop)
                cache.write_text(json.dumps(res))
            b.update(res)
            await tts_q.put(i)

        async def tts_worker():
            while True:
                i = await tts_q.get()
                if i is None:
                    return
                b = blocks[i]
                meta_f = d / "audio" / f"{b['id']}.json"
                wav = d / "audio" / f"{b['id']}.wav"
                if meta_f.exists() and wav.exists():
                    meta = json.loads(meta_f.read_text())
                else:
                    meta = await asyncio.to_thread(tts.synthesize, b["narration"] or ".", wav)
                    meta_f.write_text(json.dumps(meta))
                b["audio"] = f"audio/{b['id']}.wav"
                b["duration"] = meta["duration"]
                b["sentences"] = meta["sentences"]
                progress["n"] += 1
                report(done=progress["n"],
                       message=f"Writing narration and speech ({progress['n']}/{len(todo)})")

        worker = asyncio.create_task(tts_worker())
        await asyncio.gather(*(narrate_one(i) for i in todo))
        await tts_q.put(None)
        await worker

        # 7. Final document.
        sizes = pdf.page_sizes(src)
        doc = {"id": pid, "pages": [{"width": w, "height": h} for w, h in sizes],
               "title": next((b["text"] for b in blocks if b["type"] == "title"), ""),
               "blocks": blocks}
        (d / "doc.json").write_text(json.dumps(doc))


async def regenerate_block(pid: str, bid: str) -> dict:
    """Re-run narration + TTS for one block (after editing prompts)."""
    d = paper_dir(pid)
    doc = json.loads((d / "doc.json").read_text())
    blocks = doc["blocks"]
    i = next(k for k, b in enumerate(blocks) if b["id"] == bid)
    b = blocks[i]
    async with httpx.AsyncClient() as client:
        crop = d / b["crop"] if b.get("crop") else None
        res = await narrate.narrate_block(client, blocks, i, crop)
    (d / "narration" / f"{bid}.json").write_text(json.dumps(res))
    b.update(res)
    meta = await asyncio.to_thread(tts.synthesize, b["narration"] or ".", d / "audio" / f"{bid}.wav")
    (d / "audio" / f"{bid}.json").write_text(json.dumps(meta))
    b["duration"], b["sentences"] = meta["duration"], meta["sentences"]
    b["audio"] = f"audio/{bid}.wav"
    (d / "doc.json").write_text(json.dumps(doc))
    return b


async def _check_servers(client: httpx.AsyncClient):
    from .llm import healthy
    need = {"VLM (Qwen3-VL)": config.VLM_URL}
    if config.LAYOUT_MODEL == "docling":
        need["layout (granite-docling)"] = config.LAYOUT_URL
    for name, url in need.items():
        if not await healthy(client, url):
            raise RuntimeError(f"{name} llama-server not reachable at {url} - run scripts/start_llama.sh")
    if not Path(config.KOKORO_MODEL).exists():
        raise RuntimeError("Kokoro model missing - run scripts/download_models.sh")
