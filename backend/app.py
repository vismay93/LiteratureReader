"""FastAPI backend. Local, single-user: binds to 127.0.0.1 only.

Run: uv run python app.py   (or scripts/dev.sh)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import ingest
from pipeline.jobs import JobManager, regenerate_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
jobs = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs.start()
    yield


app = FastAPI(title="Paper Reader", lifespan=lifespan)
# Reject requests whose Host isn't loopback (blocks DNS-rebinding from web pages).
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


def _dir(pid: str):
    try:
        d = ingest.paper_dir(pid)
    except ingest.IngestError:
        raise HTTPException(404, "not found")
    if not (d / "source.pdf").exists():
        raise HTTPException(404, "not found")
    return d


@app.post("/api/papers")
async def create_paper(file: UploadFile | None = File(None), arxiv: str | None = Form(None),
                       force: bool = Form(False)):
    try:
        if file is not None:
            data = await file.read(config.MAX_PDF_BYTES + 1)
        elif arxiv:
            aid = ingest.parse_arxiv(arxiv)
            data = await asyncio.to_thread(ingest.fetch_arxiv, aid)
        else:
            raise HTTPException(400, "Provide a PDF file or an arXiv ID/URL")
        pid = ingest.store_pdf(data)
    except ingest.IngestError as e:
        raise HTTPException(400, str(e))
    jobs.submit(pid, force=force)
    return {"paper_id": pid, "status": jobs.get(pid)}


@app.get("/api/papers")
async def list_papers():
    out = []
    if config.DATA_DIR.exists():
        for d in sorted(config.DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (d / "source.pdf").exists():
                title = ""
                if (d / "doc.json").exists():
                    title = json.loads((d / "doc.json").read_text()).get("title", "")
                out.append({"paper_id": d.name, "title": title, "status": jobs.get(d.name)})
    return out


@app.get("/api/papers/{pid}")
async def get_paper(pid: str):
    d = _dir(pid)
    f = d / "doc.json"
    if not f.exists():
        raise HTTPException(409, "still processing")
    return FileResponse(f, media_type="application/json")


@app.get("/api/papers/{pid}/status")
async def status(pid: str):
    _dir(pid)
    return jobs.get(pid)


@app.get("/api/papers/{pid}/events")
async def events(pid: str):
    _dir(pid)

    async def gen():
        last = None
        while True:
            st = jobs.get(pid)
            msg = json.dumps({k: v for k, v in st.items() if k != "trace"})
            if msg != last:
                yield f"data: {msg}\n\n"
                last = msg
            if st.get("state") in ("done", "error", "unknown"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/papers/{pid}/pdf")
async def get_pdf(pid: str):
    return FileResponse(_dir(pid) / "source.pdf", media_type="application/pdf")


_FILE_RE = re.compile(r"^p\d+b\d+\.(wav|mp3|png)$")


@app.get("/api/papers/{pid}/{kind}/{name}")
async def get_asset(pid: str, kind: str, name: str):
    if kind not in ("audio", "crops") or not _FILE_RE.match(name):
        raise HTTPException(404, "not found")
    f = _dir(pid) / kind / name
    if not f.exists():
        raise HTTPException(404, "not found")
    return FileResponse(f)


@app.post("/api/papers/{pid}/blocks/{bid}/regenerate")
async def regenerate(pid: str, bid: str):
    d = _dir(pid)
    if not re.match(r"^p\d+b\d+$", bid) or not (d / "doc.json").exists():
        raise HTTPException(404, "not found")
    try:
        return await regenerate_block(pid, bid)
    except StopIteration:
        raise HTTPException(404, "block not found")


# Serve the built frontend (web/dist) if present.
_dist = config.ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
