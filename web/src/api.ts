export type BlockType =
  | "title" | "heading" | "text" | "list_item" | "figure" | "chart" | "table"
  | "equation" | "code" | "caption" | "footnote";

export interface Sentence { start: number; end: number; text: string }

export interface Block {
  id: string;
  page: number;
  type: BlockType;
  bbox: [number, number, number, number]; // normalized 0..1
  caption_bbox?: [number, number, number, number];
  text: string;
  caption?: string;
  latex?: string;
  section?: string;
  skip?: boolean;
  skip_reason?: string;
  narration?: string;
  audio?: string;
  crop?: string;
  duration?: number;
  sentences?: Sentence[];
}

export interface Doc {
  id: string;
  title: string;
  pages: { width: number; height: number }[];
  blocks: Block[];
}

export interface Status {
  state: "queued" | "running" | "done" | "error" | "unknown";
  stage?: string;
  done?: number;
  total?: number;
  message?: string;
}

export interface PaperSummary { paper_id: string; title: string; status: Status }

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  list: () => fetch("/api/papers").then((r) => json<PaperSummary[]>(r)),
  upload: (file: File) => {
    const f = new FormData();
    f.append("file", file);
    return fetch("/api/papers", { method: "POST", body: f }).then((r) => json<{ paper_id: string }>(r));
  },
  arxiv: (ref: string) => {
    const f = new FormData();
    f.append("arxiv", ref);
    return fetch("/api/papers", { method: "POST", body: f }).then((r) => json<{ paper_id: string }>(r));
  },
  doc: (id: string) => fetch(`/api/papers/${id}`).then((r) => json<Doc>(r)),
  status: (id: string) => fetch(`/api/papers/${id}/status`).then((r) => json<Status>(r)),
  regenerate: (id: string, bid: string) =>
    fetch(`/api/papers/${id}/blocks/${bid}/regenerate`, { method: "POST" }).then((r) => json<Block>(r)),
  pdfUrl: (id: string) => `/api/papers/${id}/pdf`,
  assetUrl: (id: string, rel: string) => `/api/papers/${id}/${rel}`,
  events: (id: string, onStatus: (s: Status) => void): EventSource => {
    const es = new EventSource(`/api/papers/${id}/events`);
    es.onmessage = (e) => onStatus(JSON.parse(e.data));
    return es;
  },
};
