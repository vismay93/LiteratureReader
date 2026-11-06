"""PDF ingestion: uploads and arXiv fetches, content-addressed cache."""
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

import config

# New-style (2301.12345v2) and old-style (cs.LG/0601001) arXiv IDs.
_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?)$")
_ALLOWED_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


class IngestError(ValueError):
    pass


def parse_arxiv(ref: str) -> str:
    """Return a canonical arXiv ID from an ID, 'arXiv:ID', or an arxiv.org abs/pdf URL."""
    ref = ref.strip()
    if ref.lower().startswith("arxiv:"):
        ref = ref[6:].strip()
    if _ID_RE.match(ref):
        return ref
    u = urlparse(ref if "://" in ref else "https://" + ref)
    if u.scheme not in ("http", "https") or (u.hostname or "").lower() not in _ALLOWED_HOSTS:
        raise IngestError("Only arxiv.org IDs or URLs are accepted")
    m = re.match(r"^/(abs|pdf|html)/(.+?)(\.pdf)?/?$", u.path)
    if not m or not _ID_RE.match(m.group(2)):
        raise IngestError(f"Could not find an arXiv ID in {ref!r}")
    return m.group(2)


def fetch_arxiv(arxiv_id: str) -> bytes:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    headers = {"User-Agent": "paper-reader/0.1 (local personal tool)"}
    with httpx.Client(follow_redirects=False, timeout=60, headers=headers) as c:
        # Follow redirects manually so we never leave arxiv.org.
        for _ in range(5):
            r = c.get(url)
            if r.is_redirect:
                nxt = httpx.URL(url).join(r.headers["location"])
                if (nxt.host or "").lower() not in _ALLOWED_HOSTS or nxt.scheme != "https":
                    raise IngestError("arXiv redirected off-site")
                url = str(nxt)
                continue
            break
        r.raise_for_status()
        if len(r.content) > config.MAX_PDF_BYTES:
            raise IngestError("PDF too large")
        return r.content


def store_pdf(data: bytes) -> str:
    """Validate and store a PDF; return its paper_id (sha256 prefix)."""
    if len(data) > config.MAX_PDF_BYTES:
        raise IngestError("PDF too large")
    if not data.startswith(b"%PDF"):
        raise IngestError("Not a PDF file")
    pid = hashlib.sha256(data).hexdigest()[:16]
    d = paper_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "source.pdf"
    if not p.exists():
        p.write_bytes(data)
    return pid


_PID_RE = re.compile(r"^[0-9a-f]{16}$")


def paper_dir(pid: str) -> Path:
    if not _PID_RE.match(pid):
        raise IngestError("bad paper id")
    return config.DATA_DIR / pid
