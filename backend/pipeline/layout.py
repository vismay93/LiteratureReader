"""Page layout + OCR + reading order via a local VLM.

Layout models (config.LAYOUT_MODEL):
  docling - ibm-granite/granite-docling-258M, DocTags output (llama.cpp / vLLM / transformers)
  dots    - dots-studio/dots.ocr, JSON [{bbox, category, text}] in reading order (vLLM)
  qwen    - the narration VLM with a JSON grounding prompt (fallback)
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import re
from html.parser import HTMLParser
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

# Official dots.ocr "prompt_layout_all_en" (rednote-hilab/dots.ocr, dots_ocr/utils/prompts.py).
DOTS_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
# vLLM needs the placeholder in the text, otherwise it inserts "\n" before the prompt.
DOTS_IMAGE_TOKENS = "<|img|><|imgpad|><|endofimg|>"
DOTS_FACTOR, DOTS_MIN_PIXELS, DOTS_MAX_PIXELS = 28, 3136, 11289600

DOTS_CATEGORY = {
    "caption": "caption", "footnote": "footnote", "formula": "equation", "list-item": "list_item",
    "page-footer": "page_footer", "page-header": "page_header", "picture": "figure",
    "section-header": "heading", "table": "table", "text": "text", "title": "title",
}


async def analyze_page(client: httpx.AsyncClient, image: Path, page: int) -> tuple[list[dict], str]:
    """Return (blocks, raw_model_output) for one page image."""
    if config.LAYOUT_MODEL == "qwen":
        raw = await chat(client, config.VLM_URL, [{"role": "user", "content": [
            image_part(image), {"type": "text", "text": QWEN_LAYOUT_PROMPT}]}],
            max_tokens=8192, temperature=0.0)
        return parse_qwen_layout(raw, page), raw

    if config.LAYOUT_MODEL == "dots":
        part, (w, h) = dots_image_part(image)
        raw = await chat(client, config.LAYOUT_URL, [{"role": "user", "content": [
            part, {"type": "text",
                   "text": (DOTS_IMAGE_TOKENS if config.DOTS_IMAGE_PLACEHOLDER else "") + DOTS_PROMPT}]}],
            max_tokens=config.LAYOUT_MAX_TOKENS, temperature=0.1, top_p=0.9)
        return parse_dots_layout(raw, page, w, h), raw

    # DocTags element tags are special tokens: ask the server to keep them in the output
    # (vLLM honours these fields; llama-server needs --special and ignores them).
    raw = await chat(client, config.LAYOUT_URL, [{"role": "user", "content": [
        image_part(image, max_pixels=0), {"type": "text", "text": DOCLING_PROMPT}]}],
        max_tokens=config.LAYOUT_MAX_TOKENS, temperature=0.0,
        skip_special_tokens=False, spaces_between_special_tokens=False)
    return doctags.parse_page(raw, page), raw


# ------------------------------------------------------------------ dots.ocr

def smart_resize(height: int, width: int, factor: int = DOTS_FACTOR,
                 min_pixels: int = DOTS_MIN_PIXELS, max_pixels: int = DOTS_MAX_PIXELS) -> tuple[int, int]:
    """dots.ocr / Qwen2-VL resize rule: sides divisible by `factor`, area within bounds."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def dots_image_part(path: Path) -> tuple[dict, tuple[int, int]]:
    """Resize to the exact grid the model sees, so returned bboxes are in known pixel space."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    h, w = smart_resize(img.height, img.width)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": uri}}, (w, h)


def _dots_items(raw: str) -> list[dict]:
    """JSON list of layout cells; salvages complete cells from truncated/odd output."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    items, dec = [], json.JSONDecoder()
    for m in re.finditer(r'\{\s*"bbox"', s):
        try:
            obj, _ = dec.raw_decode(s, m.start())
            items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def parse_dots_layout(raw: str, page: int, width: int, height: int) -> list[dict]:
    out = []
    for it in _dots_items(raw):
        bb = it.get("bbox")
        cat = DOTS_CATEGORY.get(str(it.get("category", "")).strip().lower())
        if cat is None or not isinstance(bb, list) or len(bb) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bb)
        except (TypeError, ValueError):
            continue
        box = [min(x0, x1) / width, min(y0, y1) / height, max(x0, x1) / width, max(y0, y1) / height]
        box = [round(min(1.0, max(0.0, v)), 4) for v in box]
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            continue
        text = (it.get("text") or "").strip()
        blk: dict = {"page": page, "type": cat, "bbox": box}
        if cat == "equation":
            blk["latex"] = blk["text"] = _strip_math(text)
        elif cat == "table":
            blk["table"] = html_table_rows(text)
            blk["text"] = ""
        elif cat == "figure":
            blk["text"] = ""
        else:
            blk["text"] = _markdown_to_text(text)
            if cat == "heading":
                blk["level"] = 1
        out.append(blk)
    return out


def _strip_math(s: str) -> str:
    s = s.strip()
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if s.startswith(a) and s.endswith(b) and len(s) > len(a) + len(b):
            return s[len(a):-len(b)].strip()
    return s


def _markdown_to_text(s: str) -> str:
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s)            # headings
    s = re.sub(r"(\*\*|__)(.+?)\1", r"\2", s)           # bold
    s = re.sub(r"(?<!\w)([*_])(\S.*?)\1(?!\w)", r"\2", s)  # italics
    s = re.sub(r"^\s*[-*+]\s+", "", s)                  # bullet
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_rows(s: str) -> list[list[str]]:
    p = _TableParser()
    try:
        p.feed(s)
        p.close()
    except Exception:
        return []
    return [r for r in p.rows if r]


# ------------------------------------------------------------------ qwen fallback

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
