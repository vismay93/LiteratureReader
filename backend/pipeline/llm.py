"""Minimal client for llama-server's OpenAI-compatible chat endpoint (text + images)."""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

import config


def image_part(path: Path) -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


async def chat(client: httpx.AsyncClient, base_url: str, messages: list[dict], *,
               max_tokens: int = 1024, temperature: float = 0.2, **extra) -> str:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, **extra}
    r = await client.post(f"{base_url}/v1/chat/completions", json=body, timeout=config.LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


async def healthy(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        r = await client.get(f"{base_url}/health", timeout=3)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
