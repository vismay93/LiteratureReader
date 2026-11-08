"""Headless batch processing + export/import of processed papers.

Typical GPU-box flow:
    python batch.py process ~/papers/*.pdf 1706.03762 @reading-list.txt --parallel 2 --export exports/
Then on your laptop:
    python batch.py import exports/*.tar.gz      (or use scripts/pull_from_gpu.sh)

Inputs to `process` can be PDF files, directories (all *.pdf inside), arXiv IDs/URLs,
or @file.txt (one input per line, # comments allowed).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import config
from pipeline import bundle, ingest, jobs

log = logging.getLogger("paper-reader.batch")


# ------------------------------------------------------------------ inputs

def expand_inputs(items: list[str]) -> list[tuple[str, str]]:
    """Return [(kind, value)] with kind 'pdf' or 'arxiv'."""
    out: list[tuple[str, str]] = []
    for it in items:
        if it.startswith("@"):
            lines = Path(it[1:]).expanduser().read_text().splitlines()
            out += expand_inputs([ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()])
            continue
        p = Path(it).expanduser()
        if p.is_dir():
            out += [("pdf", str(f)) for f in sorted(p.glob("*.pdf"))]
        elif p.is_file():
            out.append(("pdf", str(p)))
        else:
            try:
                out.append(("arxiv", ingest.parse_arxiv(it)))
            except ingest.IngestError:
                raise SystemExit(f"Not a file, directory or arXiv reference: {it!r}")
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


async def ingest_all(inputs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Store every input; returns [(paper_id, label)]. arXiv fetches are spaced 3 s apart."""
    res, last_fetch = [], 0.0
    for kind, val in inputs:
        try:
            if kind == "pdf":
                data = Path(val).read_bytes()
                label = Path(val).name
            else:
                wait = 3.0 - (time.time() - last_fetch)
                if wait > 0:
                    await asyncio.sleep(wait)
                data = await asyncio.to_thread(ingest.fetch_arxiv, val)
                last_fetch = time.time()
                label = f"arXiv:{val}"
            res.append((ingest.store_pdf(data), label))
        except Exception as e:  # keep going on bad inputs
            print(f"  ✗ {val}: {e}", flush=True)
    return res


# ------------------------------------------------------------------ process

async def cmd_process(a) -> int:
    inputs = expand_inputs(a.inputs)
    if not inputs:
        print("No inputs.")
        return 1
    print(f"Ingesting {len(inputs)} input(s)…", flush=True)
    papers = await ingest_all(inputs)
    todo = [(pid, lbl) for pid, lbl in papers
            if a.force or not (ingest.paper_dir(pid) / "doc.json").exists()]
    done_already = len(papers) - len(todo)
    if done_already:
        print(f"  {done_already} already processed (use --force to redo)")
    if a.force:  # keep layout cache; regenerate narration + speech
        for pid, _ in todo:
            d = ingest.paper_dir(pid)
            (d / "doc.json").unlink(missing_ok=True)
            for sub in ("narration", "audio"):
                shutil.rmtree(d / sub, ignore_errors=True)

    sem = asyncio.Semaphore(config.CONCURRENCY)   # shared LLM slots across papers
    papers_sem = asyncio.Semaphore(a.parallel)
    results: dict[str, dict] = {}
    t_all = time.time()

    async def run(pid: str, label: str):
        async with papers_sem:
            t0 = time.time()
            last = {"stage": None, "q": 0}

            def report(**kw):
                changed = "stage" in kw and kw["stage"] != last["stage"]
                last.update(kw)
                tot, dn = last.get("total") or 0, last.get("done") or 0
                q = 4 * dn // tot if tot else 0  # print at each quarter
                if changed or q > last["q"]:
                    last["q"] = q
                    print(f"  [{pid}] {last['stage']}: {dn}/{tot}  ({label})", flush=True)

            try:
                await jobs.process(pid, report, sem=sem)
                (ingest.paper_dir(pid) / "status.json").write_text(json.dumps(
                    {"state": "done", "stage": "done", "done": 1, "total": 1, "message": "Ready"}))
                results[pid] = {"label": label, "ok": True, "seconds": round(time.time() - t0)}
                print(f"  ✓ [{pid}] done in {time.time() - t0:.0f}s  ({label})", flush=True)
                if a.export:
                    bundle.export_paper(pid, Path(a.export), audio=a.audio)
            except Exception as e:
                log.exception("failed %s", pid)
                results[pid] = {"label": label, "ok": False, "error": f"{type(e).__name__}: {e}"}
                print(f"  ✗ [{pid}] {type(e).__name__}: {e}  ({label})", flush=True)

    print(f"Processing {len(todo)} paper(s), {a.parallel} at a time, {config.CONCURRENCY} LLM slots…",
          flush=True)
    await asyncio.gather(*(run(pid, lbl) for pid, lbl in todo))

    if a.export:  # also export papers that were already processed
        for pid, _ in papers:
            if pid not in results:
                bundle.export_paper(pid, Path(a.export), audio=a.audio)

    ok = sum(r["ok"] for r in results.values())
    failed = [r for r in results.values() if not r["ok"]]
    print(f"\nFinished {ok}/{len(todo)} in {time.time() - t_all:.0f}s"
          + (f"; bundles in {a.export}" if a.export else ""))
    for r in failed:
        print(f"  FAILED {r['label']}: {r['error']}")
    if a.report:
        Path(a.report).write_text(json.dumps(results, indent=1))
    return 1 if failed else 0


# ------------------------------------------------------------------ export / import / list

def cmd_export(a) -> int:
    ids = bundle.processed_ids() if a.all or not a.ids else a.ids
    out = Path(a.out)
    n = 0
    for pid in ids:
        try:
            p = bundle.export_paper(pid, out, audio=a.audio, skip_fresh=not a.force)
        except (bundle.BundleError, ingest.IngestError) as e:
            print(f"  ✗ {pid}: {e}")
            continue
        if p:
            n += 1
            print(f"  ✓ {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  = {pid} up to date")
    print(f"Exported {n} bundle(s) to {out}/")
    return 0


def cmd_import(a) -> int:
    files: list[Path] = []
    for x in a.bundles:
        p = Path(x).expanduser()
        files += sorted(p.glob(f"*{bundle.BUNDLE_SUFFIX}")) if p.is_dir() else [p]
    bad = 0
    for f in files:
        try:
            pid, what = bundle.import_bundle(f, force=a.force)
            print(f"  {'✓' if what == 'imported' else '='} {pid} {what}  ({f.name})")
            if a.delete:
                f.unlink()
        except (bundle.BundleError, OSError, json.JSONDecodeError) as e:
            bad += 1
            print(f"  ✗ {f.name}: {e}")
    return 1 if bad else 0


def cmd_compact(a) -> int:
    saved = 0
    for pid in a.ids or bundle.processed_ids():
        try:
            n = bundle.compact(ingest.paper_dir(pid))
        except (ingest.IngestError, OSError, RuntimeError) as e:
            print(f"  ✗ {pid}: {e}")
            continue
        saved += n
        print(f"  {'✓' if n else '='} {pid}  {n / 1e6:.1f} MB saved")
    print(f"Saved {saved / 1e6:.1f} MB in total")
    return 0


def cmd_list(a) -> int:
    for pid in bundle.processed_ids():
        doc = json.loads((ingest.paper_dir(pid) / "doc.json").read_text())
        mins = sum(b.get("duration", 0) for b in doc["blocks"]) / 60
        print(f"{pid}  {mins:5.1f} min  {doc.get('title', '')[:70]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="process PDFs / arXiv papers headlessly")
    p.add_argument("inputs", nargs="+", help="PDF files, dirs, arXiv IDs/URLs, or @list.txt")
    p.add_argument("--parallel", type=int, default=2, help="papers processed at once (default 2)")
    p.add_argument("--export", metavar="DIR", help="write a bundle per paper into DIR")
    p.add_argument("--audio", choices=["wav", "mp3"], default="mp3", help="audio format in bundles")
    p.add_argument("--force", action="store_true", help="re-run narration/TTS even if done")
    p.add_argument("--report", metavar="FILE", help="write a JSON summary")

    e = sub.add_parser("export", help="bundle processed papers for transfer")
    e.add_argument("ids", nargs="*", help="paper ids (default: all processed)")
    e.add_argument("--all", action="store_true")
    e.add_argument("--out", default=str(config.ROOT / "exports"))
    e.add_argument("--audio", choices=["wav", "mp3"], default="mp3")
    e.add_argument("--force", action="store_true", help="rebuild even if bundle is up to date")

    i = sub.add_parser("import", help="install bundles into the local library")
    i.add_argument("bundles", nargs="+", help="bundle files or directories")
    i.add_argument("--force", action="store_true", help="overwrite papers that already exist")
    i.add_argument("--delete", action="store_true",
                   help="delete each bundle file once it is imported (or already in the library)")

    c = sub.add_parser("compact", help="shrink stored papers: audio to MP3, drop page renders")
    c.add_argument("ids", nargs="*", help="paper ids (default: all processed)")

    sub.add_parser("list", help="list processed papers")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "process":
        return asyncio.run(cmd_process(a))
    return {"export": cmd_export, "import": cmd_import, "compact": cmd_compact,
            "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
