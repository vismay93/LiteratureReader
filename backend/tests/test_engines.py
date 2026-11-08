"""Engine plumbing: request fields for vLLM-style servers, image downscaling, notebook helper."""
import asyncio
import base64
import io
import sys
from pathlib import Path

import httpx
from PIL import Image

import config
from pipeline import layout, llm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import cloud  # noqa: E402


def _png(tmp_path, w, h):
    p = tmp_path / "x.png"
    Image.new("RGB", (w, h), "white").save(p)
    return p


def test_image_part_downscales(tmp_path, monkeypatch):
    p = _png(tmp_path, 2000, 1000)
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 500_000)
    img = Image.open(io.BytesIO(base64.b64decode(llm.image_part(p)["image_url"]["url"].split(",")[1])))
    assert img.width * img.height <= 500_000 and abs(img.width / img.height - 2) < 0.02
    full = Image.open(io.BytesIO(base64.b64decode(llm.image_part(p, max_pixels=0)["image_url"]["url"].split(",")[1])))
    assert full.size == (2000, 1000)


def test_layout_request_keeps_special_tokens_and_names_model(tmp_path, monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content":
            "<doctag><text><loc_1><loc_2><loc_100><loc_20>Hi</text></doctag>"}}]})

    monkeypatch.setattr(config, "LAYOUT_URL", "http://layout.test")
    monkeypatch.setattr(config, "LAYOUT_MODEL_NAME", "layout")
    monkeypatch.setattr(config, "LAYOUT_MAX_TOKENS", 6800)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await layout.analyze_page(c, _png(tmp_path, 100, 130), 0)

    blocks, _ = asyncio.run(run())
    assert blocks[0]["text"] == "Hi"
    assert seen["model"] == "layout" and seen["max_tokens"] == 6800
    assert seen["skip_special_tokens"] is False and seen["spaces_between_special_tokens"] is False


def test_no_model_field_when_unnamed(monkeypatch):
    monkeypatch.setattr(config, "VLM_MODEL_NAME", None)
    assert llm.model_name(config.VLM_URL) is None


def test_vllm_install_args_by_driver():
    assert cloud.vllm_install_args(13.0) == ["vllm"]
    old = cloud.vllm_install_args(12.4)
    assert old[0].startswith("vllm @ https://github.com/vllm-project/vllm/releases/download/")
    assert "cu129" in old[0] and "https://download.pytorch.org/whl/cu129" in old


def test_vllm_command_is_loopback_and_named():
    cmd = cloud.vllm_command("vlm", "Qwen/Qwen3-VL-4B-Instruct", 8082, {"dtype": "half"},
                             util=0.68, max_len=8192, seqs=8, eager=True)
    assert cmd[1:3] == ["serve", "Qwen/Qwen3-VL-4B-Instruct"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--served-model-name") + 1] == "vlm"
    assert cmd[cmd.index("--dtype") + 1] == "half" and "--enforce-eager" in cmd


def test_engine_env_vllm():
    env = cloud._engine_env({"engine": "vllm", "concurrency": 12, "hf_home": "/x"})
    assert env["PR_LAYOUT_URL"].endswith(":8081") and env["PR_VLM_URL"].endswith(":8082")
    assert env["PR_VLM_MODEL_NAME"] == "vlm" and env["PR_NP"] == "12" and env["HF_HOME"] == "/x"
    assert int(env["PR_LAYOUT_MAX_TOKENS"]) < 8192
