// Sequential block-audio player in reading order.
import { api, type Block, type Doc } from "./api";

export interface PlayerEvents {
  onBlock: (b: Block | undefined, index: number) => void;
  onSentence: (b: Block, sentence: number) => void;
  onState: (playing: boolean) => void;
  onTime: (elapsed: number, total: number) => void;
}

export interface Filters { footnotes: boolean; visuals: boolean }

export class Player {
  private audio = new Audio();
  private preload = new Audio();
  private queue: Block[] = [];
  private idx = -1;
  private sentence = -1;
  private offsets: number[] = []; // cumulative start time of each queue item
  rate = Number(localStorage.getItem("pr.rate") || "1.25");

  constructor(private doc: Doc, private ev: PlayerEvents, filters: Filters) {
    this.audio.preservesPitch = true;
    this.preload.preload = "auto";
    this.setFilters(filters);
    this.audio.addEventListener("ended", () => this.next(true));
    this.audio.addEventListener("play", () => this.ev.onState(true));
    this.audio.addEventListener("pause", () => this.ev.onState(false));
    this.audio.addEventListener("timeupdate", () => this.tick());
  }

  setFilters(f: Filters) {
    const cur = this.current();
    this.queue = this.doc.blocks.filter((b) =>
      !b.skip && b.audio && b.type !== "caption" &&
      (f.footnotes || b.type !== "footnote") &&
      (f.visuals || !["figure", "chart", "table", "equation"].includes(b.type)));
    let t = 0;
    this.offsets = this.queue.map((b) => { const s = t; t += b.duration ?? 0; return s; });
    if (cur) this.idx = this.queue.findIndex((b) => b.id === cur.id);
  }

  get totalDuration() {
    const n = this.queue.length;
    return n ? this.offsets[n - 1] + (this.queue[n - 1].duration ?? 0) : 0;
  }
  get length() { return this.queue.length; }
  get index() { return this.idx; }
  get playing() { return !this.audio.paused; }
  current(): Block | undefined { return this.queue[this.idx]; }

  /** Start at a given block (or the next playable one after it in document order). */
  playFrom(b: Block) {
    let i = this.queue.findIndex((q) => q.id === b.id);
    if (i < 0) {
      const docIdx = this.doc.blocks.indexOf(b);
      i = this.queue.findIndex((q) => this.doc.blocks.indexOf(q) > docIdx);
    }
    if (i >= 0) this.go(i, true);
  }

  go(i: number, autoplay: boolean, at = 0) {
    if (i < 0 || i >= this.queue.length) {
      this.idx = this.queue.length;
      this.audio.pause();
      this.ev.onBlock(undefined, i);
      return;
    }
    this.idx = i;
    this.sentence = -1;
    const b = this.queue[i];
    this.audio.src = api.assetUrl(this.doc.id, b.audio!);
    this.audio.playbackRate = this.rate;
    this.audio.currentTime = at;
    this.ev.onBlock(b, i);
    const nxt = this.queue[i + 1];
    if (nxt) this.preload.src = api.assetUrl(this.doc.id, nxt.audio!);
    if (autoplay) void this.audio.play();
    localStorage.setItem(`pr.pos.${this.doc.id}`, b.id);
  }

  toggle() {
    if (this.idx < 0 || this.idx >= this.queue.length) return this.go(0, true);
    if (this.audio.paused) void this.audio.play(); else this.audio.pause();
  }

  next(auto = false) { this.go(this.idx + 1, auto || this.playing); }
  prev() {
    // Restart the current block if we're more than 2s in, otherwise go back one.
    if (this.audio.currentTime > 2) { this.audio.currentTime = 0; return; }
    this.go(Math.max(0, this.idx - 1), this.playing);
  }

  setRate(r: number) {
    this.rate = Math.round(Math.max(0.5, Math.min(3, r)) * 100) / 100;
    this.audio.playbackRate = this.rate;
    localStorage.setItem("pr.rate", String(this.rate));
  }

  seekSentence(k: number) {
    const b = this.current();
    const s = b?.sentences?.[k];
    if (s) { this.audio.currentTime = s.start; if (this.audio.paused) void this.audio.play(); }
  }

  /** Restore the last position for this paper without autoplay. */
  restore() {
    const id = localStorage.getItem(`pr.pos.${this.doc.id}`);
    const i = id ? this.queue.findIndex((b) => b.id === id) : -1;
    if (i >= 0) this.go(i, false);
  }

  private tick() {
    const b = this.current();
    if (!b) return;
    const t = this.audio.currentTime;
    this.ev.onTime(this.offsets[this.idx] + t, this.totalDuration);
    const ss = b.sentences ?? [];
    let k = ss.findIndex((s) => t >= s.start && t < s.end + 0.18);
    if (k < 0 && ss.length && t >= ss[ss.length - 1].start) k = ss.length - 1;
    if (k !== this.sentence) {
      this.sentence = k;
      this.ev.onSentence(b, k);
    }
  }
}
