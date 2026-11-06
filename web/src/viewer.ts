// PDF rendering (pdf.js) with a clickable, highlightable block overlay per page.
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { Block, Doc } from "./api";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

const TYPE_LABEL: Record<string, string> = {
  figure: "Figure", chart: "Chart", table: "Table", equation: "Equation", code: "Code",
};

export class Viewer {
  private pageEls: HTMLElement[] = [];
  private blockEls = new Map<string, HTMLElement>();
  private rendered = new Set<number>();
  private pdf?: pdfjs.PDFDocumentProxy;
  private observer?: IntersectionObserver;
  private active?: string;
  zoom = 1;

  constructor(
    private root: HTMLElement,
    private doc: Doc,
    private onBlockClick: (b: Block) => void,
  ) {}

  async load(url: string) {
    this.pdf = await pdfjs.getDocument({ url }).promise;
    this.layout();
  }

  /** Build page shells + overlays; canvases are rendered lazily when visible. */
  layout() {
    this.root.innerHTML = "";
    this.pageEls = [];
    this.blockEls.clear();
    this.rendered.clear();
    this.observer?.disconnect();
    const width = Math.min(this.root.clientWidth - 32, 1000) * this.zoom;

    this.observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) this.renderPage(Number((e.target as HTMLElement).dataset.page));
      }
    }, { root: this.root, rootMargin: "800px 0px" });

    this.doc.pages.forEach((p, i) => {
      const el = document.createElement("div");
      el.className = "page";
      el.dataset.page = String(i);
      el.style.width = `${width}px`;
      el.style.height = `${(width * p.height) / p.width}px`;
      const overlay = document.createElement("div");
      overlay.className = "overlay";
      el.appendChild(overlay);
      this.root.appendChild(el);
      this.pageEls.push(el);
      this.observer!.observe(el);
    });

    for (const b of this.doc.blocks) {
      if (b.type === "caption") continue; // captions highlight with their figure/table
      const box = document.createElement("div");
      box.className = `block t-${b.type}${b.skip ? " skipped" : ""}`;
      box.dataset.id = b.id;
      const [x0, y0, x1, y1] = unionBox(b);
      Object.assign(box.style, {
        left: `${x0 * 100}%`, top: `${y0 * 100}%`,
        width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`,
      });
      if (TYPE_LABEL[b.type]) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = TYPE_LABEL[b.type];
        box.appendChild(tag);
      }
      box.title = b.skip ? "Skipped" : "Click to read from here";
      box.addEventListener("click", () => this.onBlockClick(b));
      this.pageEls[b.page]?.querySelector(".overlay")!.appendChild(box);
      this.blockEls.set(b.id, box);
    }
    if (this.active) this.setActive(this.active, false);
  }

  private async renderPage(i: number) {
    if (!this.pdf || this.rendered.has(i)) return;
    this.rendered.add(i);
    const page = await this.pdf.getPage(i + 1);
    const el = this.pageEls[i];
    const cssW = el.clientWidth;
    const base = page.getViewport({ scale: 1 });
    const dpr = window.devicePixelRatio || 1;
    const viewport = page.getViewport({ scale: (cssW / base.width) * dpr });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    el.insertBefore(canvas, el.firstChild);
    await page.render({ canvas, viewport }).promise;
  }

  setActive(id: string | undefined, scroll = true) {
    if (this.active) this.blockEls.get(this.active)?.classList.remove("active");
    this.active = id;
    if (!id) return;
    const el = this.blockEls.get(id);
    if (!el) return;
    el.classList.add("active");
    if (scroll) {
      const r = el.getBoundingClientRect();
      const vr = this.root.getBoundingClientRect();
      // Only scroll if the block isn't comfortably within view.
      if (r.top < vr.top + 60 || r.bottom > vr.bottom - 60) {
        this.root.scrollBy({ top: r.top - vr.top - vr.height * 0.25, behavior: "smooth" });
      }
    }
  }

  setZoom(z: number) {
    this.zoom = Math.max(0.5, Math.min(2.5, z));
    this.layout();
  }
}

function unionBox(b: Block): [number, number, number, number] {
  if (!b.caption_bbox) return b.bbox;
  const [a, c] = [b.bbox, b.caption_bbox];
  return [Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.max(a[2], c[2]), Math.max(a[3], c[3])];
}
