"""Page layout + OCR + reading order via a local VLM on llama-server."""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

import config
from . import doctags
from .llm import chat, image_part

DOCLING_PROMPT = "Convert this page to docling."

QWEN_LAYOUT_PROMPT = """You are a document layout analyzer for academic papers.
List every layout element on this page in human reading order (for two-column pages: \
left column top-to-bottom, then right column). Return ONLY a JSON array, each item:
{"type": one of ["title","heading","text","list_item","figure","chart","table","equation","caption","footnote","code","page_header","page_footer"],
 "bbox_2d": [x0, y0, x1, y1] (integers on a 0-1000 grid relative to the image),
 "text": the exact text (LaTeX for equations, empty for figures/charts/tables)}"""


async def analyze_page(client: httpx.AsyncClient, image: Path, page: int) -> tuple[list[dict], str]:
    """Return (blocks, raw_model_output) for one page image."""
    if config.LAYOUT_MODEL == "qwen":
        raw = await chat(client, config.VLM_URL, [{"role": "user", "content": [
            image_part(image), {"type": "text", "text": QWEN_LAYOUT_PROMPT}]}],
            max_tokens=8192, temperature=0.0)
        return parse_qwen_layout(raw, page), raw

    raw = await chat(client, config.LAYOUT_URL, [{"role": "user", "content": [
        image_part(image), {"type": "text", "text": DOCLING_PROMPT}]}],
        max_tokens=8192, temperature=0.0)
    return doctags.parse_page(raw, page), raw


def parse_qwen_layout(raw: str, page: int) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        bb = it.get("bbox_2d") or it.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x0, y0, x1, y1 = (float(v) / 1000.0 for v in bb)
        t = it.get("type", "text")
        blk = {"page": page, "type": t, "bbox": [round(min(x0, x1), 4), round(min(y0, y1), 4),
                                                  round(max(x0, x1), 4), round(max(y0, y1), 4)],
               "text": (it.get("text") or "").strip()}
        if t == "equation":
            blk["latex"] = blk["text"]
        if t == "table":
            blk["table"] = []
        out.append(blk)
    return out
