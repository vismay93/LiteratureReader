"""OpenAI-compatible chat server on Hugging Face transformers (alternative to llama-server).

One process serves both models, each under its own prefix, so the backend just points at them:
    PR_LAYOUT_URL=http://127.0.0.1:8081/layout   PR_VLM_URL=http://127.0.0.1:8081/vlm

    python inference/hf_server.py --layout-model ibm-granite/granite-docling-258M \
                                  --vlm-model Qwen/Qwen3-VL-4B-Instruct --max-batch 4

Endpoints (per prefix): POST /v1/chat/completions, GET /health, GET /v1/models.
Concurrent requests are micro-batched (left-padded) per model for GPU throughput.
Binds to loopback only; it is an internal service for the Paper Reader backend.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

log = logging.getLogger("hf_server")

# DocTags end markers / padding that should not reach the parser.
# transformers' lazy imports are not thread-safe, and loading two models at once spikes memory:
# load one engine at a time.
_LOAD_LOCK = threading.Lock()

_STRIP_TOKENS = ("<end_of_utterance>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>")


def pick_device() -> tuple[str, "object"]:
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


@dataclass
class Request:
    messages: list
    max_tokens: int
    temperature: float
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    images: list = field(default_factory=list)
    conv: list = field(default_factory=list)


class Engine:
    """A model + processor with a background thread that batches queued requests."""

    def __init__(self, name: str, model_id: str, *, quant: str = "none", dtype: str = "auto",
                 max_batch: int = 4, keep_special: bool = False, max_pixels: int | None = None,
                 allow_system: bool = True):
        self.name, self.model_id = name, model_id
        self.max_batch, self.keep_special = max_batch, keep_special
        self.max_pixels, self.allow_system = max_pixels, allow_system
        self.ready = threading.Event()
        self.error: str | None = None
        self._quant, self._dtype = quant, dtype
        self._q: queue.Queue[Request] = queue.Queue()
        self._pending: list[Request] = []
        threading.Thread(target=self._load_and_serve, daemon=True, name=f"engine-{name}").start()

    # ------------------------------------------------------------ loading
    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        device, auto_dtype = pick_device()
        dtype = auto_dtype if self._dtype == "auto" else getattr(torch, self._dtype)
        kw: dict = {"dtype": dtype}
        if self._quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self._quant == "4bit", load_in_8bit=self._quant == "8bit",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        if device == "cuda":
            kw["device_map"] = "cuda"
        t = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kw)
        if device != "cuda":
            self.model.to(device)
        self.model.eval()
        self.device = device
        log.info("%s: loaded %s on %s (%s, quant=%s) in %.0fs", self.name, self.model_id, device,
                 str(dtype).replace("torch.", ""), self._quant, time.time() - t)

    def _load_and_serve(self):
        try:
            with _LOAD_LOCK:
                self._load()
        except Exception as e:  # surfaced through /health
            log.exception("%s: failed to load", self.name)
            self.error = f"{type(e).__name__}: {e}"
            return
        self.ready.set()
        while True:
            batch = self._next_batch()
            try:
                outs = self._run(batch)
                for r, o in zip(batch, outs):
                    r.loop.call_soon_threadsafe(_resolve, r.future, o, None)
            except Exception as e:
                log.exception("%s: batch of %d failed", self.name, len(batch))
                for r in batch:
                    r.loop.call_soon_threadsafe(_resolve, r.future, None, e)

    # ------------------------------------------------------------ batching
    def _next_batch(self) -> list[Request]:
        if not self._pending:
            self._pending.append(self._q.get())
        deadline = time.time() + 0.03  # brief window to gather concurrent requests
        while True:
            timeout = deadline - time.time()
            try:
                self._pending.append(self._q.get(timeout=max(0.0, timeout)) if timeout > 0 else self._q.get_nowait())
            except queue.Empty:
                break
        # Only batch requests that agree on having images (processors dislike mixed batches).
        first = self._pending[0]
        kind = bool(first.images)
        batch, rest = [], []
        for r in self._pending:
            (batch if bool(r.images) == kind and len(batch) < self.max_batch else rest).append(r)
        self._pending = rest
        return batch

    def _run(self, batch: list[Request]) -> list[dict]:
        import torch
        texts = [self.processor.apply_chat_template(r.conv, tokenize=False, add_generation_prompt=True)
                 for r in batch]
        images = [r.images for r in batch] if batch[0].images else None
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        temps = [r.temperature for r in batch]
        gen = {"max_new_tokens": max(r.max_tokens for r in batch), "pad_token_id": self.tok.pad_token_id}
        if max(temps) > 0.01:
            gen.update(do_sample=True, temperature=sum(temps) / len(temps), top_p=0.9)
        else:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        n_in = inputs["input_ids"].shape[1]
        new = out[:, n_in:]
        eos = set(_as_list(self.model.generation_config.eos_token_id)) | {self.tok.eos_token_id}
        results = []
        for i, r in enumerate(batch):
            ids = new[i][: r.max_tokens].tolist()
            ids = [x for x in ids if x != self.tok.pad_token_id]
            finish = "stop" if any(x in eos for x in ids) else "length"
            text = self.tok.decode(ids, skip_special_tokens=not self.keep_special)
            for s in _STRIP_TOKENS + ((self.tok.pad_token,) if self.tok.pad_token else ()):
                text = text.replace(s, "")
            results.append({"text": text.strip(), "finish": finish, "prompt_tokens": n_in,
                            "completion_tokens": len(ids)})
        log.info("%s: batch=%d in=%d out<=%d %.1fs", self.name, len(batch), n_in, new.shape[1], time.time() - t)
        return results

    # ------------------------------------------------------------ API
    async def submit(self, messages: list, max_tokens: int, temperature: float) -> dict:
        conv, images = self._convert(messages)
        loop = asyncio.get_running_loop()
        r = Request(messages, max_tokens, temperature, loop.create_future(), loop, images, conv)
        self._q.put(r)
        return await r.future

    def _convert(self, messages: list) -> tuple[list, list]:
        """OpenAI chat messages (with data: URI images) -> HF chat conversation + PIL images."""
        from PIL import Image
        conv, images = [], []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "system" and not self.allow_system:
                role = "user"
            parts = []
            for p in ([{"type": "text", "text": content}] if isinstance(content, str) else content):
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"] if isinstance(p.get("image_url"), dict) else p["image_url"]
                    if not url.startswith("data:"):
                        raise ValueError("only data: URI images are accepted")
                    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                    if self.max_pixels and img.width * img.height > self.max_pixels:
                        s = (self.max_pixels / (img.width * img.height)) ** 0.5
                        img = img.resize((max(28, int(img.width * s)), max(28, int(img.height * s))))
                    images.append(img)
                    parts.append({"type": "image"})
            conv.append({"role": role, "content": parts})
        return conv, images


def _as_list(x):
    return [] if x is None else (list(x) if isinstance(x, (list, tuple)) else [x])


def _resolve(fut: asyncio.Future, value, exc):
    if fut.done():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(value)


def build_app(engines: dict[str, Engine]):
    app = FastAPI(title="Paper Reader HF inference")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    def eng(name: str) -> Engine:
        e = engines.get(name)
        if e is None:
            raise HTTPException(404, f"unknown model prefix {name!r}")
        return e

    @app.get("/health")
    async def health_all():
        bad = {n: e.error or "loading" for n, e in engines.items() if not e.ready.is_set()}
        return JSONResponse({"status": "ok" if not bad else "loading", "engines": bad},
                            status_code=200 if not bad else 503)

    @app.get("/{name}/health")
    async def health(name: str):
        e = eng(name)
        if e.error:
            return JSONResponse({"status": "error", "error": e.error}, status_code=500)
        if not e.ready.is_set():
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok", "model": e.model_id}

    @app.get("/{name}/v1/models")
    async def models(name: str):
        e = eng(name)
        return {"object": "list", "data": [{"id": e.model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/{name}/v1/chat/completions")
    async def chat(name: str, req: HTTPRequest):
        e = eng(name)
        if not e.ready.is_set():
            raise HTTPException(503, e.error or "model is still loading")
        body = await req.json()
        if body.get("stream"):
            raise HTTPException(400, "streaming is not supported")
        try:
            r = await e.submit(body["messages"], int(body.get("max_tokens") or 1024),
                               float(body.get("temperature") if body.get("temperature") is not None else 0.2))
        except (KeyError, ValueError) as ex:
            raise HTTPException(400, str(ex))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": e.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                         "finish_reason": r["finish"]}],
            "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                      "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
        }

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--layout-model", default="ibm-granite/granite-docling-258M")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                    help="bitsandbytes quantization for the VLM (CUDA only)")
    ap.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    ap.add_argument("--max-batch", type=int, default=4, help="max requests per generate() call")
    ap.add_argument("--vlm-max-pixels", type=int, default=1_300_000,
                    help="downscale VLM images above this many pixels (bounds vision tokens)")
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("hf_server is an internal service and only binds to loopback.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch  # noqa: F401  - import eagerly in the main thread (see _LOAD_LOCK)
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
    engines = {
        # DocTags tags are special tokens: keep them in the decoded text.
        "layout": Engine("layout", a.layout_model, max_batch=a.max_batch, keep_special=True,
                         allow_system=False, dtype=a.dtype),
        "vlm": Engine("vlm", a.vlm_model, quant=a.quant, max_batch=a.max_batch,
                      max_pixels=a.vlm_max_pixels, dtype=a.dtype),
    }
    import uvicorn
    uvicorn.run(build_app(engines), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
