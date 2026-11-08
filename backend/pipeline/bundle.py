"""Portable paper bundles: export a processed paper on one machine, import it on another.

A bundle is `<paper_id>.tar.gz` containing `<paper_id>/` with doc.json, source.pdf, audio,
crops, narration and layout caches. Importing validates every member path, refuses links,
and checks that source.pdf hashes to the paper id, so a bundle can't write outside its slot.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import config
from . import audio as audio_io
from .ingest import paper_dir

BUNDLE_SUFFIX = ".tar.gz"
FORMAT_VERSION = 1
_PID = r"[0-9a-f]{16}"
_MEMBER_RE = re.compile(
    rf"^(?P<pid>{_PID})/(doc\.json|source\.pdf|manifest\.json|"
    r"audio/p\d+b\d+\.(wav|mp3|json)|crops/p\d+b\d+\.png|"
    r"narration/p\d+b\d+\.json|layout/\d+\.json)$")
_DIR_RE = re.compile(rf"^{_PID}(/(audio|crops|narration|layout))?/?$")
MAX_MEMBER_BYTES = 512 * 1024 * 1024


class BundleError(ValueError):
    pass


def processed_ids() -> list[str]:
    if not config.DATA_DIR.exists():
        return []
    return sorted(d.name for d in config.DATA_DIR.iterdir() if (d / "doc.json").exists())


def export_paper(pid: str, out_dir: Path, audio: str = "wav", skip_fresh: bool = True) -> Path | None:
    """Write out_dir/<pid>.tar.gz. Returns the path, or None if an up-to-date bundle exists."""
    d = paper_dir(pid)
    doc_f = d / "doc.json"
    if not doc_f.exists():
        raise BundleError(f"{pid} is not fully processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pid}{BUNDLE_SUFFIX}"
    if skip_fresh and out.exists() and out.stat().st_mtime >= doc_f.stat().st_mtime:
        return None

    doc = json.loads(doc_f.read_text())
    tmp = out.with_suffix(".partial")
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        tar.add(d / "source.pdf", arcname=f"{pid}/source.pdf")
        for sub, pat in (("crops", "*.png"), ("narration", "*.json"), ("layout", "*.json")):
            for f in sorted((d / sub).glob(pat)):
                tar.add(f, arcname=f"{pid}/{sub}/{f.name}")
        for f in sorted((d / "audio").glob("*.json")):
            tar.add(f, arcname=f"{pid}/audio/{f.name}")
        for b in doc["blocks"]:
            if not b.get("audio"):
                continue
            src = d / b["audio"].split("?")[0]
            b["audio"] = f"audio/{src.stem}.{audio}"
            if src.suffix == f".{audio}":
                tar.add(src, arcname=f"{pid}/{b['audio']}")
            else:
                _add_bytes(tar, f"{pid}/{b['audio']}", audio_io.convert(src, audio))
        _add_bytes(tar, f"{pid}/doc.json", json.dumps(doc).encode())
        manifest = {"format": FORMAT_VERSION, "paper_id": pid, "title": doc.get("title", ""),
                    "audio": audio, "exported_at": time.time(),
                    "models": {"layout": config.LAYOUT_MODEL, "voice": config.KOKORO_VOICE}}
        _add_bytes(tar, f"{pid}/manifest.json", json.dumps(manifest, indent=1).encode())
    tmp.replace(out)
    return out


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = int(time.time())
    ti.mode = 0o644
    tar.addfile(ti, io.BytesIO(data))



def import_bundle(path: Path, force: bool = False) -> tuple[str, str]:
    """Validate and install a bundle into DATA_DIR. Returns (paper_id, 'imported'|'skipped')."""
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        pids = set()
        for m in members:
            name = m.name.rstrip("/")
            if m.isdir() and _DIR_RE.match(name):
                continue
            if not m.isreg():
                raise BundleError(f"{path.name}: refusing non-regular member {m.name!r}")
            mm = _MEMBER_RE.match(m.name)
            if not mm:
                raise BundleError(f"{path.name}: unexpected member {m.name!r}")
            if m.size > MAX_MEMBER_BYTES:
                raise BundleError(f"{path.name}: member too large {m.name!r}")
            pids.add(mm.group("pid"))
        if len(pids) != 1:
            raise BundleError(f"{path.name}: bundle must contain exactly one paper")
        pid = pids.pop()
        names = {m.name for m in members}
        for req in ("doc.json", "source.pdf"):
            if f"{pid}/{req}" not in names:
                raise BundleError(f"{path.name}: missing {req}")

        dest = paper_dir(pid)
        if (dest / "doc.json").exists() and not force:
            return pid, "skipped"

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=config.DATA_DIR) as tmp:
            for m in members:
                if not m.isreg():
                    continue
                target = Path(tmp) / m.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            staged = Path(tmp) / pid
            digest = hashlib.sha256((staged / "source.pdf").read_bytes()).hexdigest()
            if digest[:16] != pid:
                raise BundleError(f"{path.name}: source.pdf does not match paper id")
            doc = json.loads((staged / "doc.json").read_text())
            if doc.get("id") != pid:
                raise BundleError(f"{path.name}: doc.json id mismatch")
            for b in doc.get("blocks", []):
                for key in ("audio", "crop"):
                    if b.get(key) and f"{pid}/{b[key]}" not in names:
                        raise BundleError(f"{path.name}: {key} file missing for block {b.get('id')}")
            (staged / "status.json").write_text(json.dumps(
                {"state": "done", "stage": "done", "done": 1, "total": 1, "message": "Imported"}))
            compact(staged)  # e.g. a WAV bundle is stored as MP3
            if dest.exists():
                shutil.rmtree(dest)
            staged.replace(dest)
    return pid, "imported"


def _size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def compact(d: Path) -> int:
    """Shrink a processed paper folder in place; returns bytes saved.

    Re-encodes block audio into config.AUDIO_FORMAT (MP3 by default) and deletes the page
    renders, which only feed the layout model and are re-created if ever needed.
    """
    doc_f = d / "doc.json"
    if not doc_f.exists():
        return 0
    before = _size(d)
    doc = json.loads(doc_f.read_text())
    fmt, changed = config.AUDIO_FORMAT, False
    for b in doc.get("blocks", []):
        rel = (b.get("audio") or "").split("?")[0]
        src = d / rel
        if not rel or src.suffix == f".{fmt}" or not src.exists():
            continue
        dst = src.with_suffix(f".{fmt}")
        tmp = dst.with_name(dst.name + ".partial")
        tmp.write_bytes(audio_io.convert(src, fmt))
        tmp.replace(dst)
        b["audio"] = f"audio/{dst.name}"
        changed = True
        src.unlink()
    if changed:
        tmp = doc_f.with_name("doc.json.partial")
        tmp.write_text(json.dumps(doc))
        tmp.replace(doc_f)
    shutil.rmtree(d / "pages", ignore_errors=True)
    return before - _size(d)
