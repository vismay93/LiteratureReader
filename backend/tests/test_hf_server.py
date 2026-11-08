"""API-layer tests for inference/hf_server.py using a stub engine (no torch/model needed)."""
import base64
import io
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference"))
import hf_server  # noqa: E402


class StubEngine:
    def __init__(self, ready=True, error=None):
        self.model_id = "stub/model"
        self.ready = threading.Event()
        if ready:
            self.ready.set()
        self.error = error
        self.calls = []

    async def submit(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return {"text": "hello", "finish": "stop", "prompt_tokens": 3, "completion_tokens": 1}


def client(**engines):
    return TestClient(hf_server.build_app(engines), base_url="http://127.0.0.1")


def test_chat_completion_shape():
    e = StubEngine()
    r = client(vlm=e).post("/vlm/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 7, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["usage"]["total_tokens"] == 4
    assert e.calls[0][1:] == (7, 0.0)


def test_health_states():
    c = client(layout=StubEngine(ready=False), vlm=StubEngine())
    assert c.get("/vlm/health").status_code == 200
    assert c.get("/layout/health").status_code == 503
    assert c.get("/health").status_code == 503
    assert client(vlm=StubEngine(ready=False, error="boom")).get("/vlm/health").status_code == 500


def test_rejects_unknown_prefix_stream_and_bad_host():
    c = client(vlm=StubEngine())
    assert c.post("/nope/v1/chat/completions", json={"messages": []}).status_code == 404
    assert c.post("/vlm/v1/chat/completions", json={"messages": [], "stream": True}).status_code == 400
    assert c.get("/vlm/health", headers={"Host": "evil.example"}).status_code == 400


def _data_uri(w=64, h=48):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_convert_messages_images_and_system():
    e = object.__new__(hf_server.Engine)
    e.max_pixels, e.allow_system = 1000, False
    conv, imgs = e._convert([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_uri()}},
                                     {"type": "text", "text": "describe"}]},
    ])
    assert conv[0] == {"role": "user", "content": [{"type": "text", "text": "be brief"}]}
    assert conv[1]["content"] == [{"type": "image"}, {"type": "text", "text": "describe"}]
    assert len(imgs) == 1 and imgs[0].width * imgs[0].height <= 1000 * 1.05


def test_convert_rejects_remote_urls():
    e = object.__new__(hf_server.Engine)
    e.max_pixels, e.allow_system = None, True
    with pytest.raises(ValueError):
        e._convert([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}])


def test_refuses_non_loopback_bind():
    with pytest.raises(SystemExit):
        hf_server.main(["--host", "0.0.0.0"])
