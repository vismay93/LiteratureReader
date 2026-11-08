"""Generate the spoken narration for each block."""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

import config
from . import prompts
from .llm import chat, image_part

# Numeric citations: [12], [3, 4], [1–5], [2; 7], [12, 15–17]
_CITE_RE = re.compile(r"\s*\[\d+(?:\s*[,;–-]\s*\d+)*\]")
_YEAR = r"(?:19|20)\d{2}[a-z]?"
# One author-year citation: "Smith, 2020", "Smith et al. 2020", "Smith and Jones, 2019a", "e.g., Lee 2021"
_AUTHOR_YEAR = re.compile(
    rf"^(?:(?:e\.g\.|i\.e\.|see|cf\.)\s*,?\s*)?"
    rf"(?:[A-Z][\w'’\-]+(?:\s+(?:van|von|de|der|da|del|la|le)\s+[A-Z][\w'’\-]+)*)"
    rf"(?:\s+et\s+al\.?|(?:,?\s+(?:and|&)\s+|,\s+)[A-Z][\w'’\-]+)*"
    rf",?\s+{_YEAR}(?:\s*,\s*{_YEAR})*$")
_PAREN_RE = re.compile(r"\s*\(([^()]{3,300})\)")
_YEARS_ONLY_RE = re.compile(rf"\s*\({_YEAR}(?:\s*[;,]\s*{_YEAR})*\)")   # "Smith et al. (2020)"
_URL_RE = re.compile(r"\s*(?:https?://|www\.)\S+", re.I)
_DOI_RE = re.compile(r"\s*\bdoi:\s*\S+", re.I)
_HYPHEN_RE = re.compile(r"\b([a-z]{2,})- (?!and\b|or\b|to\b|vs\b)([a-z]{2,})\b")
TEXT_TYPES = ("text", "list_item", "footnote")
_FIG_NUM_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table)\s*(\d+)", re.I)


def heading_narration(text: str) -> str:
    t = text.strip()
    m = re.match(r"^((?:\d+\.)*\d+|[A-Z](?:\.\d+)*)\.?\s+(.*)$", t)
    if m:
        num, title = m.groups()
        return f"Section {num}. {title}."
    return t if t.endswith(".") else t + "."


def clean_markup(s: str) -> str:
    s = s.strip().strip('"').strip()
    s = re.sub(r"[*_#`]+", "", s)
    return re.sub(r"\s+", " ", s)


def _strip_paren_citation(m: re.Match) -> str:
    parts = [p.strip() for p in m.group(1).split(";")]
    return "" if parts and all(_AUTHOR_YEAR.match(p) for p in parts) else m.group(0)


def clean_text(text: str) -> str:
    """Rule-based narration for plain text: drop citations, URLs and DOIs, re-join words
    hyphenated across lines, and tidy spacing. Used instead of an LLM rewrite by default."""
    s = _CITE_RE.sub("", text)
    s = _PAREN_RE.sub(_strip_paren_citation, s)
    s = _YEARS_ONLY_RE.sub("", s)
    s = _URL_RE.sub("", s)
    s = _DOI_RE.sub("", s)
    s = _HYPHEN_RE.sub(r"\1\2", s)
    s = re.sub(r"\(\s*\)|\[\s*\]", "", s)              # brackets left empty
    s = re.sub(r"\s+([.,;:!?)])", r"\1", s)               # "word ." -> "word."
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_source(b: dict) -> str:
    """How a text-like block is narrated: 'llm' (VLM rewrite) or 'rules' (clean_text).
    The author/affiliation block always uses the LLM (one short call per paper)."""
    return "llm" if config.REWRITE_TEXT or b.get("authors") else "rules"


def uses_llm(b: dict) -> bool:
    """Whether narrating this block calls the VLM (so it must take an LLM concurrency slot)."""
    if b["type"] in ("heading", "title"):
        return False
    if b["type"] in TEXT_TYPES:
        return text_source(b) == "llm" and len(b.get("text", "").split()) >= 4
    return True


def visual_label(b: dict) -> tuple[str, str] | None:
    """('Figure', '3') from the caption, if present."""
    m = _FIG_NUM_RE.match(b.get("caption") or "")
    if not m:
        return None
    kind = "Table" if m.group(1).lower().startswith("tab") else "Figure"
    return kind, m.group(2)


def find_reference_context(blocks: list[dict], label: tuple[str, str] | None, near: int) -> str:
    """Paragraphs that mention e.g. 'Figure 3' / 'Fig. 3' / 'Table 2'; else neighbours."""
    if label:
        kind, num = label
        pat = re.compile(rf"\b({'Tab(?:le)?' if kind == 'Table' else 'Fig(?:ure)?'})\.?\s*{num}\b", re.I)
        hits = [b["text"] for b in blocks if b["type"] in ("text", "list_item") and pat.search(b.get("text", ""))]
        if hits:
            return " ... ".join(hits[:2])[:2000]
    ctx = [blocks[j]["text"] for j in (near - 1, near + 1)
           if 0 <= j < len(blocks) and blocks[j]["type"] == "text"]
    return " ... ".join(ctx)[:1500]


def table_cells(rows: list[list[str]]) -> str:
    if not rows:
        return "(none)"
    return "\n".join(" | ".join(r) for r in rows[:40])[:3000]


async def narrate_block(client: httpx.AsyncClient, blocks: list[dict], i: int, crop: Path | None) -> dict:
    """Return {'narration': str, optionally 'type': 'chart'} for blocks[i]."""
    b = blocks[i]
    t = b["type"]
    section = b.get("section") or "(unknown)"

    if t in ("heading", "title"):
        return {"narration": heading_narration(b["text"]) if t == "heading" else b["text"]}

    if t in TEXT_TYPES:
        text = b["text"]
        src = text_source(b)
        if src == "rules" or len(text.split()) < 4:
            return {"narration": clean_text(text), "src": src}
        prev = next((blocks[j]["text"] for j in range(i - 1, -1, -1)
                     if blocks[j]["type"] in ("text", "list_item")), "(start of paper)")
        out = await chat(client, config.VLM_URL, [
            {"role": "system", "content": prompts.NARRATOR_SYSTEM},
            {"role": "user", "content": prompts.TEXT_USER.format(section=section, prev=prev[-600:], text=text)},
        ], max_tokens=min(2048, 64 + int(len(text.split()) * 2.5)), temperature=0.2)
        return {"narration": clean_markup(out) or clean_text(text), "src": src}

    ctx = find_reference_context(blocks, visual_label(b), i)
    content: list[dict] = [image_part(crop)] if crop else []

    if t == "equation":
        content.append({"type": "text", "text": prompts.EQUATION_USER.format(
            latex=b.get("latex") or "(none)", section=section, context=ctx or "(none)")})
        out = await chat(client, config.VLM_URL, [{"role": "user", "content": content}],
                         max_tokens=400, temperature=0.2)
        return {"narration": clean_markup(out)}

    if t in ("figure", "chart"):
        content.append({"type": "text", "text": prompts.FIGURE_USER.format(
            caption=b.get("caption") or "(none)", section=section, context=ctx or "(none)")})
        out = await chat(client, config.VLM_URL, [{"role": "user", "content": content}],
                         max_tokens=700, temperature=0.2)
        res = _parse_json_obj(out)
        if res and res.get("narration"):
            r = {"narration": clean_markup(res["narration"])}
            if res.get("kind") == "chart":
                r["type"] = "chart"
            return r
        return {"narration": clean_markup(out)}

    if t == "table":
        content.append({"type": "text", "text": prompts.TABLE_USER.format(
            caption=b.get("caption") or "(none)", section=section,
            cells=table_cells(b.get("table") or []), context=ctx or "(none)")})
        out = await chat(client, config.VLM_URL, [{"role": "user", "content": content}],
                         max_tokens=700, temperature=0.2)
        return {"narration": clean_markup(out)}

    if t == "code":
        out = await chat(client, config.VLM_URL, [{"role": "user", "content": prompts.CODE_USER.format(
            section=section, text=b["text"][:3000])}], max_tokens=300, temperature=0.2)
        return {"narration": clean_markup(out)}

    return {"narration": b.get("text", "")}


def _parse_json_obj(s: str) -> dict | None:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
