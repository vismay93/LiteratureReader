"""Parse granite-docling DocTags output into typed, located blocks.

DocTags look like:
  <doctag><section_header_level_1><loc_40><loc_60><loc_200><loc_70>1 Introduction</section_header_level_1>
  <text><loc_..><loc_..><loc_..><loc_..>Body text ...</text>
  <picture><loc_..>x4<caption><loc_..>x4 Figure 1: ...</caption></picture>
  <otsl><loc_..>x4<ched>A<ched>B<nl><fcel>1<fcel>2<nl></otsl>
  <formula><loc_..>x4 E = mc^2</formula></doctag>
Locations are on a 0..500 grid relative to the page image, in (x0, y0, x1, y1) order.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GRID = 500.0

_TOKEN_RE = re.compile(r"<(/?)([a-zA-Z_][a-zA-Z0-9_]*)>")
_LOC_RE = re.compile(r"^loc_(\d+)$")
# Void (self-closing) tokens: locations and OTSL cell/row markers.
_OTSL_CELLS = {"fcel", "ecel", "ched", "rhed", "srow", "lcel", "ucel", "xcel", "nl"}

# DocTags element -> our block type
_TYPE_MAP = {
    "text": "text",
    "paragraph": "text",
    "title": "title",
    "section_header": "heading",
    "list_item": "list_item",
    "picture": "figure",
    "chart": "chart",
    "otsl": "table",
    "table": "table",
    "formula": "equation",
    "code": "code",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "page_header",
    "page_footer": "page_footer",
    "reference": "text",
}
_CONTAINERS = {"unordered_list", "ordered_list", "doctag", "group", "inline"}


@dataclass
class Node:
    tag: str
    children: list = field(default_factory=list)  # Node | str
    locs: list[int] = field(default_factory=list)

    def text(self) -> str:
        out = []
        for c in self.children:
            if isinstance(c, str):
                out.append(c)
            elif c.tag not in ("caption",):
                out.append(c.text())
        return "".join(out)


def tokenize_tree(doctags: str) -> Node:
    root = Node("root")
    stack = [root]
    pos = 0
    for m in _TOKEN_RE.finditer(doctags):
        if m.start() > pos:
            stack[-1].children.append(doctags[pos:m.start()])
        pos = m.end()
        closing, name = m.group(1) == "/", m.group(2)
        loc = _LOC_RE.match(name)
        if loc:
            stack[-1].locs.append(int(loc.group(1)))
            continue
        if name in _OTSL_CELLS:
            stack[-1].children.append(Node(name))
            continue
        if not closing:
            n = Node(name)
            stack[-1].children.append(n)
            stack.append(n)
        else:
            # Close the nearest matching open tag; tolerate malformed output.
            for i in range(len(stack) - 1, 0, -1):
                if stack[i].tag == name:
                    del stack[i:]
                    break
    if pos < len(doctags):
        stack[-1].children.append(doctags[pos:])
    return root


def norm_tag(tag: str) -> str:
    # section_header_level_2 -> section_header
    return re.sub(r"_level_\d+$", "", tag)


def bbox_of(node: Node) -> list[float] | None:
    if len(node.locs) < 4:
        return None
    x0, y0, x1, y1 = node.locs[:4]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [round(v / GRID, 4) for v in (x0, y0, x1, y1)]


def otsl_rows(node: Node) -> list[list[str]]:
    """Convert OTSL cell tokens to a list of rows of cell strings."""
    rows: list[list[str]] = [[]]
    cur: list[str] | None = None
    for c in node.children:
        if isinstance(c, Node) and c.tag in _OTSL_CELLS:
            if c.tag == "nl":
                rows.append([])
                cur = None
            elif c.tag in ("lcel", "ucel", "xcel"):
                rows[-1].append("")  # merged cell continuation
                cur = None
            else:
                rows[-1].append("")
                cur = rows[-1]
        elif isinstance(c, str) and cur is not None:
            cur[-1] += c
    return [[cell.strip() for cell in r] for r in rows if r]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_page(doctags: str, page: int) -> list[dict]:
    """Return blocks for one page, in DocTags (reading) order."""
    root = tokenize_tree(doctags)
    blocks: list[dict] = []

    def visit(n: Node):
        tag = norm_tag(n.tag)
        if tag in _CONTAINERS or tag == "root":
            for c in n.children:
                if isinstance(c, Node):
                    visit(c)
            return
        btype = _TYPE_MAP.get(tag)
        if btype is None:
            for c in n.children:
                if isinstance(c, Node):
                    visit(c)
            return
        bbox = bbox_of(n)
        if bbox is None:
            return
        blk: dict = {"page": page, "type": btype, "bbox": bbox}
        caps = [c for c in n.children if isinstance(c, Node) and c.tag == "caption"]
        if caps:
            blk["caption"] = _clean(caps[0].text())
        if btype == "table":
            blk["table"] = otsl_rows(n)
            blk["text"] = ""
        elif btype == "equation":
            blk["latex"] = _clean(n.text())
            blk["text"] = blk["latex"]
        elif btype in ("figure", "chart"):
            # Picture classification may appear as a child tag, e.g. <chart> or <bar_chart>.
            kinds = [c.tag for c in n.children if isinstance(c, Node) and c.tag != "caption"]
            if any("chart" in k or "plot" in k or "graph" in k for k in kinds):
                blk["type"] = "chart"
            blk["text"] = ""
        else:
            blk["text"] = _clean(n.text())
            if tag == "section_header":
                m = re.search(r"_level_(\d+)$", n.tag)
                blk["level"] = int(m.group(1)) if m else 1
        blocks.append(blk)

    visit(root)
    return blocks


def postprocess(blocks: list[dict]) -> list[dict]:
    """Drop page furniture, attach stray captions, assign ids/sections/skip flags."""
    out: list[dict] = []
    for b in blocks:
        if b["type"] in ("page_header", "page_footer"):
            continue
        if b["type"] == "text" and re.fullmatch(r"\d{1,3}", b.get("text", "")):
            continue  # page number
        if b["type"] == "text" and not b.get("text"):
            continue
        out.append(b)

    out = _fix_front_matter(out)
    for b in out:
        raw = b.get("ocr_text") or b.get("text", "")
        if b["type"] == "text" and re.match(r"^\s*(\$\^\{[^}]*\}\$|[∗*†‡§¶]\s)", raw) and b["bbox"][1] > 0.6:
            b["type"] = "footnote"

    # Attach standalone captions to the nearest figure/table on the same page.
    merged: list[dict] = []
    for i, b in enumerate(out):
        if b["type"] != "caption":
            merged.append(b)
            continue
        target = _nearest_visual(out, i)
        if target is not None and not target.get("caption"):
            target["caption"] = b["text"]
            target["caption_bbox"] = b["bbox"]
            b["skip"] = True  # read as part of the figure explanation
        merged.append(b)

    section = ""
    in_refs = False
    counters: dict[int, int] = {}
    for b in merged:
        if b["type"] in ("heading", "title"):
            section = b["text"]
            in_refs = bool(re.match(r"^(\d+\.?\s*)?(references|bibliography)\b", section, re.I))
        b["section"] = section
        b.setdefault("skip", False)
        if in_refs and b["type"] not in ("heading", "title"):
            b["skip"] = True
            b["skip_reason"] = "references"
        elif in_refs:
            b["skip_reason"] = "references"
        if b["type"] == "footnote":
            b["skip_reason"] = "footnote"
        n = counters.get(b["page"], 0)
        counters[b["page"]] = n + 1
        b["id"] = f"p{b['page']}b{n}"
    return merged


def _fix_front_matter(blocks: list[dict]) -> list[dict]:
    """On page 0: promote the first heading to title; merge author/affiliation snippets."""
    if not blocks or any(b["type"] == "title" for b in blocks):
        first = None
    else:
        first = next((b for b in blocks if b["page"] == 0 and b["type"] == "heading"), None)
    if first is not None and first["bbox"][1] < 0.35 and not re.match(r"^\d", first["text"]):
        first["type"] = "title"

    title_i = next((i for i, b in enumerate(blocks) if b["type"] == "title"), None)
    if title_i is None:
        return blocks
    out = blocks[:title_i + 1]
    group: list[dict] = []
    i = title_i + 1
    while i < len(blocks) and blocks[i]["page"] == 0 and blocks[i]["type"] == "text" \
            and len(blocks[i]["text"].split()) <= 20:
        group.append(blocks[i])
        i += 1
    if len(group) >= 2:
        out.append({"page": 0, "type": "text", "authors": True,
                    "text": "; ".join(g["text"] for g in group),
                    "bbox": [min(g["bbox"][0] for g in group), min(g["bbox"][1] for g in group),
                             max(g["bbox"][2] for g in group), max(g["bbox"][3] for g in group)]})
    else:
        out.extend(group)
    out.extend(blocks[i:])
    return out


def _nearest_visual(blocks: list[dict], i: int) -> dict | None:
    cap = blocks[i]
    is_table_cap = cap.get("text", "").lower().startswith("table")
    best, best_d = None, 1e9
    for j in (i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < len(blocks):
            b = blocks[j]
            if b["page"] != cap["page"] or b["type"] not in ("figure", "chart", "table"):
                continue
            if is_table_cap and b["type"] != "table":
                continue
            d = _vdist(b["bbox"], cap["bbox"])
            if d < best_d:
                best, best_d = b, d
    return best if best_d < 0.15 else None


def _vdist(a: list[float], b: list[float]) -> float:
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0
