"""PDF helpers: page rendering, region crops, and text-layer extraction (PyMuPDF)."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

import config


def page_count(pdf: Path) -> int:
    with pymupdf.open(pdf) as doc:
        return doc.page_count


def page_sizes(pdf: Path) -> list[tuple[float, float]]:
    with pymupdf.open(pdf) as doc:
        return [(p.rect.width, p.rect.height) for p in doc]


def render_page(pdf: Path, page: int, out: Path, long_side: int = config.LAYOUT_LONG_SIDE) -> Path:
    with pymupdf.open(pdf) as doc:
        p = doc[page]
        zoom = long_side / max(p.rect.width, p.rect.height)
        pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        pix.save(out)
    return out


def crop_region(pdf: Path, page: int, bbox: list[float], out: Path,
                dpi: int = config.CROP_DPI, pad: float = 0.008) -> Path:
    """Render a normalized bbox region (0..1) at the given DPI."""
    with pymupdf.open(pdf) as doc:
        p = doc[page]
        W, H = p.rect.width, p.rect.height
        x0, y0, x1, y1 = bbox
        clip = pymupdf.Rect(max(0, (x0 - pad) * W), max(0, (y0 - pad) * H),
                            min(W, (x1 + pad) * W), min(H, (y1 + pad) * H))
        pix = p.get_pixmap(dpi=dpi, clip=clip, alpha=False)
        pix.save(out)
    return out


class TextLayer:
    """Words of a born-digital PDF, used to replace OCR text inside layout boxes."""

    def __init__(self, pdf: Path):
        self.pages: list[list[tuple]] = []
        self.sizes: list[tuple[float, float]] = []
        with pymupdf.open(pdf) as doc:
            for p in doc:
                self.pages.append(p.get_text("words", sort=False))
                self.sizes.append((p.rect.width, p.rect.height))

    def has_text(self, page: int) -> bool:
        return len(self.pages[page]) > 20

    def text_in(self, page: int, bbox: list[float]) -> str:
        W, H = self.sizes[page]
        x0, y0, x1, y1 = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
        tol = 2.0
        hits = []
        for w in self.pages[page]:
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            if x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol:
                hits.append(w)
        # Words carry (block_no, line_no, word_no) at indices 5..7; keep PDF order.
        hits.sort(key=lambda w: (w[5], w[6], w[7]))
        text = " ".join(w[4] for w in hits)
        text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # de-hyphenate line breaks
        return text.strip()
