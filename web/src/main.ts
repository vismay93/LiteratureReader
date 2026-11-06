import "./styles.css";
import { api, type Block, type Doc, type Status } from "./api";
import { Viewer } from "./viewer";
import { Player, type Filters } from "./player";

const app = document.getElementById("app")!;
let cleanup: (() => void) | undefined;

function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...kids: (Node | string)[]) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  el.append(...kids);
  return el;
}

function route() {
  cleanup?.();
  cleanup = undefined;
  const m = location.hash.match(/^#\/paper\/([0-9a-f]{16})$/);
  if (m) void showPaper(m[1]); else void showHome();
}
window.addEventListener("hashchange", route);
route();

// ---------------------------------------------------------------- Home

async function showHome() {
  app.innerHTML = `
    <div class="home">
      <h1>Paper Reader</h1>
      <p class="sub">Drop a paper and listen to it. Everything runs on your machine.</p>
      <label class="drop" id="drop">
        <input type="file" accept="application/pdf" hidden id="file" />
        <div><strong>Drop a PDF here</strong> or click to choose</div>
      </label>
      <form class="arxiv" id="arxiv">
        <input name="ref" placeholder="arXiv ID or URL, e.g. 1706.03762" autocomplete="off" />
        <button>Fetch</button>
      </form>
      <div class="err" id="err"></div>
      <h2>Library</h2>
      <ul class="library" id="lib"><li class="muted">Loading…</li></ul>
    </div>`;
  const err = app.querySelector<HTMLElement>("#err")!;
  const start = async (p: Promise<{ paper_id: string }>) => {
    err.textContent = "";
    try { location.hash = `#/paper/${(await p).paper_id}`; } catch (e) { err.textContent = String((e as Error).message); }
  };
  const drop = app.querySelector<HTMLElement>("#drop")!;
  const file = app.querySelector<HTMLInputElement>("#file")!;
  file.onchange = () => file.files?.[0] && start(api.upload(file.files[0]));
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer?.files?.[0];
    if (f) void start(api.upload(f));
  };
  app.querySelector<HTMLFormElement>("#arxiv")!.onsubmit = (e) => {
    e.preventDefault();
    const ref = new FormData(e.target as HTMLFormElement).get("ref") as string;
    if (ref.trim()) void start(api.arxiv(ref.trim()));
  };

  const lib = app.querySelector<HTMLElement>("#lib")!;
  try {
    const papers = await api.list();
    lib.innerHTML = "";
    if (!papers.length) lib.append(h("li", { class: "muted" }, "No papers yet."));
    for (const p of papers) {
      const a = h("a", { href: `#/paper/${p.paper_id}` }, p.title || p.paper_id);
      const st = p.status.state === "done" ? "" : ` · ${p.status.state}`;
      lib.append(h("li", {}, a, h("span", { class: "muted" }, st)));
    }
  } catch (e) {
    lib.innerHTML = "";
    lib.append(h("li", { class: "err" }, `Backend unreachable: ${(e as Error).message}`));
  }
}

// ---------------------------------------------------------------- Processing

async function showPaper(id: string) {
  const st = await api.status(id).catch(() => ({ state: "unknown" }) as Status);
  if (st.state === "done") return showReader(id);
  app.innerHTML = `
    <div class="home">
      <a href="#/" class="back">← Library</a>
      <h1>Preparing your paper</h1>
      <p class="sub" id="msg">Starting…</p>
      <div class="bar"><div id="fill"></div></div>
      <p class="muted" id="stage"></p>
      <pre class="err" id="err"></pre>
    </div>`;
  const STAGES = ["render", "layout", "narrate"];
  let finished = false;
  const onStatus = (s: Status) => {
    if (finished) return;
    const msg = app.querySelector<HTMLElement>("#msg");
    if (!msg) return;
    msg.textContent = s.message ?? s.state;
    const si = Math.max(0, STAGES.indexOf(s.stage ?? ""));
    const frac = s.total ? (s.done ?? 0) / s.total : 0;
    app.querySelector<HTMLElement>("#fill")!.style.width = `${((si + frac) / STAGES.length) * 100}%`;
    app.querySelector<HTMLElement>("#stage")!.textContent =
      s.stage && STAGES.includes(s.stage) ? `Step ${si + 1} of ${STAGES.length}` : "";
    if (s.state === "done") { stop(); void showReader(id); }
    if (s.state === "error") { stop(); app.querySelector<HTMLElement>("#err")!.textContent = s.message ?? "Failed"; }
  };
  const es = api.events(id, onStatus);
  // Polling fallback for proxies that buffer server-sent events (e.g. Google Colab's port proxy).
  const poll = window.setInterval(() => { void api.status(id).then(onStatus).catch(() => {}); }, 3000);
  const stop = () => { finished = true; es.close(); window.clearInterval(poll); };
  cleanup = stop;
}

// ---------------------------------------------------------------- Reader

async function showReader(id: string) {
  const doc: Doc = await api.doc(id);
  document.title = doc.title || "Paper Reader";
  app.innerHTML = `
    <div class="reader">
      <header>
        <a href="#/" class="back">← Library</a>
        <div class="title"></div>
        <div class="zoom"><button id="zout" title="Zoom out">−</button><button id="zin" title="Zoom in">+</button></div>
      </header>
      <main id="pages"></main>
      <aside>
        <div class="kind" id="kind"></div>
        <div class="section" id="section"></div>
        <div class="narr" id="narr"><p class="muted">Press play, or click any paragraph, figure, table or equation in the paper.</p></div>
        <details class="orig"><summary>Original</summary><div id="orig"></div></details>
        <div class="opts">
          <label><input type="checkbox" id="optVis" checked /> Explain figures, tables & equations</label>
          <label><input type="checkbox" id="optFoot" /> Read footnotes</label>
          <button id="regen" class="link">Regenerate this block</button>
        </div>
      </aside>
      <footer>
        <button id="prev" title="Previous (←)">⏮</button>
        <button id="play" class="play" title="Play/Pause (Space)">▶</button>
        <button id="next" title="Next (→)">⏭</button>
        <div class="speed"><button id="slow">−</button><span id="rate"></span><button id="fast">+</button></div>
        <div class="progress"><div class="track"><div id="pfill"></div></div><span id="time"></span></div>
      </footer>
    </div>`;
  const $ = <T extends HTMLElement>(s: string) => app.querySelector<T>(s)!;
  $(".title").textContent = doc.title;

  const filters: Filters = {
    visuals: localStorage.getItem("pr.visuals") !== "0",
    footnotes: localStorage.getItem("pr.footnotes") === "1",
  };
  $<HTMLInputElement>("#optVis").checked = filters.visuals;
  $<HTMLInputElement>("#optFoot").checked = filters.footnotes;

  let viewer: Viewer;
  const player = new Player(doc, {
    onBlock: (b) => { viewer.setActive(b?.id); showBlock(b); },
    onSentence: (_b, k) => {
      $("#narr").querySelectorAll(".s").forEach((el, i) => el.classList.toggle("on", i === k));
    },
    onState: (p) => { $("#play").textContent = p ? "⏸" : "▶"; },
    onTime: (t, total) => {
      $("#time").textContent = `${fmt(t / player.rate)} / ${fmt(total / player.rate)}`;
      $("#pfill").style.width = `${total ? (t / total) * 100 : 0}%`;
    },
  }, filters);
  viewer = new Viewer($("#pages"), doc, (b) => player.playFrom(b));

  const KIND: Record<string, string> = {
    title: "Title", heading: "Section", text: "Paragraph", list_item: "List item", figure: "Figure",
    chart: "Chart", table: "Table", equation: "Equation", code: "Code", footnote: "Footnote",
  };
  function showBlock(b?: Block) {
    if (!b) { $("#narr").innerHTML = `<p class="muted">End of paper.</p>`; return; }
    $("#kind").textContent = KIND[b.type] ?? b.type;
    $("#kind").className = `kind k-${b.type}`;
    $("#section").textContent = b.section ?? "";
    const narr = $("#narr");
    narr.innerHTML = "";
    const sents = b.sentences?.length ? b.sentences : [{ text: b.narration ?? "", start: 0, end: 0 }];
    sents.forEach((s, k) => {
      const span = h("span", { class: "s" }, s.text + " ");
      span.onclick = () => player.seekSentence(k);
      narr.append(span);
    });
    const orig = $("#orig");
    orig.innerHTML = "";
    if (b.crop) orig.append(h("img", { src: api.assetUrl(id, b.crop), alt: "" }));
    orig.append(h("p", {}, b.caption ?? ""), h("p", {}, b.latex ?? b.text ?? ""));
  }

  const rateLabel = () => { $("#rate").textContent = `${player.rate.toFixed(2)}×`; };
  rateLabel();
  $("#play").onclick = () => player.toggle();
  $("#next").onclick = () => player.next();
  $("#prev").onclick = () => player.prev();
  $("#slow").onclick = () => { player.setRate(player.rate - 0.25); rateLabel(); };
  $("#fast").onclick = () => { player.setRate(player.rate + 0.25); rateLabel(); };
  $("#zin").onclick = () => viewer.setZoom(viewer.zoom + 0.15);
  $("#zout").onclick = () => viewer.setZoom(viewer.zoom - 0.15);
  const onFilter = () => {
    filters.visuals = $<HTMLInputElement>("#optVis").checked;
    filters.footnotes = $<HTMLInputElement>("#optFoot").checked;
    localStorage.setItem("pr.visuals", filters.visuals ? "1" : "0");
    localStorage.setItem("pr.footnotes", filters.footnotes ? "1" : "0");
    player.setFilters(filters);
  };
  $("#optVis").onchange = onFilter;
  $("#optFoot").onchange = onFilter;
  $("#regen").onclick = async () => {
    const b = player.current();
    if (!b) return;
    $("#regen").textContent = "Regenerating…";
    try {
      const nb = await api.regenerate(id, b.id);
      Object.assign(b, nb);
      b.audio = `${nb.audio}?v=${Date.now()}`; // bust cache
      player.setFilters(filters);
      player.go(player.index, true);
    } finally { $("#regen").textContent = "Regenerate this block"; }
  };

  const onKey = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); player.toggle(); }
    else if (e.key === "ArrowRight") player.next();
    else if (e.key === "ArrowLeft") player.prev();
    else if (e.key === "+" || e.key === "=") { player.setRate(player.rate + 0.25); rateLabel(); }
    else if (e.key === "-") { player.setRate(player.rate - 0.25); rateLabel(); }
  };
  let resizeT = 0;
  const onResize = () => { clearTimeout(resizeT); resizeT = window.setTimeout(() => viewer.layout(), 200); };
  window.addEventListener("keydown", onKey);
  window.addEventListener("resize", onResize);
  cleanup = () => {
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", onResize);
    if (player.playing) player.toggle();
  };
  try {
    await viewer.load(api.pdfUrl(id));
  } catch (e) {
    $("#pages").textContent = `Could not render PDF: ${(e as Error).message}`;
  }
  player.restore();
}

function fmt(s: number) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}
