"""Minimal client for OpenAI-compatible chat endpoints (llama-server, vLLM, hf_server)."""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import httpx

import config


def image_part(path: Path, max_pixels: int | None = None) -> dict:
    """PNG as a data: URI, downscaled to at most max_pixels (default config.MAX_IMAGE_PIXELS)."""
    data = Path(path).read_bytes()
    limit = config.MAX_IMAGE_PIXELS if max_pixels is None else max_pixels
    if limit:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > limit:
            s = (limit / (img.width * img.height)) ** 0.5
            img = img.convert("RGB").resize((max(32, int(img.width * s)), max(32, int(img.height * s))),
                                            Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            data = buf.getvalue()
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def model_name(base_url: str) -> str | None:
    if base_url == config.LAYOUT_URL:
        return config.LAYOUT_MODEL_NAME
    if base_url == config.VLM_URL:
        return config.VLM_MODEL_NAME
    return None


async def chat(client: httpx.AsyncClient, base_url: str, messages: list[dict], *,
               max_tokens: int = 1024, temperature: float = 0.2, **extra) -> str:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, **extra}
    name = model_name(base_url)
    if name:
        body["model"] = name
    if base_url == config.VLM_URL and config.VLM_DISABLE_THINKING:
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    r = await client.post(f"{base_url}/v1/chat/completions", json=body, timeout=config.LLM_TIMEOUT)
    r.raise_for_status()
    return strip_reasoning(r.json()["choices"][0]["message"]["content"] or "")


_THINK_RE = re.compile(r"<think>.*?</think>\s*|^.*?</think>\s*", re.S)


def strip_reasoning(text: str) -> str:
    """Drop any <think>...</think> block a reasoning model emitted despite the switch."""
    return _THINK_RE.sub("", text, count=1) if "</think>" in text else text


async def healthy(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        r = await client.get(f"{base_url}/health", timeout=3)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
