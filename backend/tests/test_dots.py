"""dots.ocr layout parsing, Qwen3.x thinking switch, and the vLLM plan for dots/Qwen3.8."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
from PIL import Image

import config
from pipeline import doctags, layout, llm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import cloud  # noqa: E402

W, H = 1232, 1596
RAW = json.dumps([
    {"bbox": [100, 50, 1100, 90], "category": "Title", "text": "# Attention Is All You Need"},
    {"bbox": [100, 120, 600, 150], "category": "Section-header", "text": "## 1 Introduction"},
    {"bbox": [100, 160, 600, 400], "category": "Text", "text": "Recurrent models **dominate** [3]."},
    {"bbox": [650, 160, 1150, 500], "category": "Picture"},
    {"bbox": [650, 505, 1150, 540], "category": "Caption", "text": "Figure 1: The model."},
    {"bbox": [100, 420, 600, 460], "category": "Formula", "text": "$$E = mc^2$$"},
    {"bbox": [100, 600, 1100, 800], "category": "Table",
     "text": "<table><tr><th>Model</th><th>BLEU</th></tr><tr><td>Base</td><td>27.3</td></tr></table>"},
    {"bbox": [100, 1500, 1100, 1550], "category": "Page-footer", "text": "3"},
])


def test_smart_resize_matches_dots_rule():
    h, w = layout.smart_resize(1600, 1236)
    assert h % 28 == 0 and w % 28 == 0 and abs(h - 1600) <= 14 and abs(w - 1236) <= 14
    h, w = layout.smart_resize(10000, 8000)          # over max_pixels -> scaled down
    assert h * w <= layout.DOTS_MAX_PIXELS and h % 28 == 0 and w % 28 == 0


def test_parse_dots_layout_maps_types_and_normalizes():
    blocks = layout.parse_dots_layout(RAW, 2, W, H)
    assert [b["type"] for b in blocks] == ["title", "heading", "text", "figure", "caption",
                                            "equation", "table", "page_footer"]
    assert blocks[0]["text"] == "Attention Is All You Need"
    assert blocks[1]["text"] == "1 Introduction"
    assert blocks[2]["text"] == "Recurrent models dominate [3]."
    assert blocks[2]["bbox"] == [round(100 / W, 4), round(160 / H, 4), round(600 / W, 4), round(400 / H, 4)]
    assert blocks[5]["latex"] == "E = mc^2"
    assert blocks[6]["table"] == [["Model", "BLEU"], ["Base", "27.3"]]
    # postprocess drops the footer and attaches the caption to the figure
    post = doctags.postprocess(blocks)
    fig = next(b for b in post if b["type"] == "figure")
    assert fig["caption"] == "Figure 1: The model." and "page_footer" not in [b["type"] for b in post]


def test_parse_dots_salvages_truncated_output():
    raw = RAW[: RAW.index('{"bbox": [100, 600')] + '{"bbox": [100, 600, 1100, 8'
    blocks = layout.parse_dots_layout(raw, 0, W, H)
    assert len(blocks) == 6 and blocks[-1]["type"] == "equation"


def test_parse_dots_accepts_object_wrapper_and_skips_junk():
    raw = json.dumps({"layout": [{"bbox": [0, 0, 10, 10], "category": "Text", "text": "a"},
                                 {"bbox": [0, 0, 0, 0], "category": "Text", "text": "empty box"},
                                 {"bbox": "bad", "category": "Text"},
                                 {"bbox": [1, 1, 5, 5], "category": "Unknown"}]})
    assert [b["text"] for b in layout.parse_dots_layout(raw, 0, 100, 100)] == ["a"]


def test_dots_request_format(tmp_path, monkeypatch):
    img = tmp_path / "p.png"
    Image.new("RGB", (1236, 1600), "white").save(img)
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": RAW}}]})

    monkeypatch.setattr(config, "LAYOUT_MODEL", "dots")
    monkeypatch.setattr(config, "LAYOUT_URL", "http://layout.test")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await layout.analyze_page(c, img, 0)

    blocks, _ = asyncio.run(run())
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["text"].startswith("<|img|><|imgpad|><|endofimg|>Please output the layout")
    assert seen["temperature"] == 0.1 and seen["top_p"] == 0.9
    assert len(blocks) == 8


def test_thinking_switch_and_strip(monkeypatch):
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hmm</think>\nAnswer."}}]})

    monkeypatch.setattr(config, "VLM_URL", "http://vlm.test")
    monkeypatch.setattr(config, "VLM_DISABLE_THINKING", True)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await llm.chat(c, config.VLM_URL, [{"role": "user", "content": "x"}])

    assert asyncio.run(run()) == "Answer."
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert llm.strip_reasoning("no think tags") == "no think tags"
    assert llm.strip_reasoning("dangling reasoning</think> Final") == "Final"


def test_vllm_plan_by_gpu():
    t4 = cloud.vllm_plan(15360, 7.5)
    assert t4["layout_model"] == cloud.LAYOUT_MODEL and t4["dtype"] == "half"
    l4 = cloud.vllm_plan(23034, 8.9)
    assert l4["layout_model"] == cloud.DOTS_MODEL and l4["vlm_model"].endswith("4B-Instruct-FP8")
    h100 = cloud.vllm_plan(81559, 9.0)
    assert h100["vlm_model"] == "Qwen/Qwen3.8-27B-FP8"
    for p in (t4, l4, h100):
        assert p["layout_gpu_util"] + p["vlm_gpu_util"] <= 0.9 + 1e-9
    assert cloud.vllm_plan(81559, 9.0, layout="docling")["layout_model"] == cloud.LAYOUT_MODEL


def test_dots_vllm_command_pins_revision():
    st = cloud.vllm_plan(23034, 8.9)
    cmd = cloud.vllm_command("layout", cloud.DOTS_MODEL, 8081, st, util=0.4, max_len=32768, seqs=16, eager=True)
    assert "--trust-remote-code" in cmd
    assert cmd[cmd.index("--revision") + 1] == cloud.DOTS_REVISION and len(cloud.DOTS_REVISION) == 40
    assert cmd[cmd.index("--chat-template-content-format") + 1] == "string"
    other = cloud.vllm_command("vlm", "Qwen/Qwen3.8-27B-FP8", 8082, st, util=0.5, max_len=8192, seqs=8, eager=False)
    assert "--trust-remote-code" not in other


def test_engine_env_for_dots():
    env = cloud._engine_env(cloud.vllm_plan(23034, 8.9))
    assert env["PR_LAYOUT_MODEL"] == "dots" and env["PR_LAYOUT_LONG_SIDE"] == "1600"


def test_dots_prompt_without_placeholder_for_sglang(tmp_path, monkeypatch):
    img = tmp_path / "p.png"
    Image.new("RGB", (300, 400), "white").save(img)
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    monkeypatch.setattr(config, "LAYOUT_MODEL", "dots")
    monkeypatch.setattr(config, "DOTS_IMAGE_PLACEHOLDER", False)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await layout.analyze_page(c, img, 0)

    asyncio.run(run())
    assert seen["messages"][0]["content"][1]["text"].startswith("Please output the layout")
