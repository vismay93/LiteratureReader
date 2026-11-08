"""vLLM memory planning: profiling hints, OOM detection, automatic step-down and retry."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import cloud  # noqa: E402

A10G = (23028, 8.6)
OOM_LOG = ("ValueError: No available memory for the cache blocks. Try increasing "
           "`gpu_memory_utilization` when initializing the engine")


def test_24gb_plan_fits_two_servers():
    p = cloud.vllm_plan(*A10G)
    assert p["layout_model"] == cloud.DOTS_MODEL
    assert p["vlm_model"] == "Qwen/Qwen3-VL-4B-Instruct-FP8"     # FP8 via Marlin on Ampere
    assert p["max_model_len"] == 8192 and p["layout_max_len"] == 16384
    assert p["layout_max_tokens"] < p["layout_max_len"]
    assert p["layout_gpu_util"] + p["vlm_gpu_util"] <= 0.90 + 1e-9
    assert cloud.vllm_plan(15360, 7.5)["vlm_model"] == "Qwen/Qwen3-VL-4B-Instruct"  # no FP8 on T4


def test_profiling_uses_real_image_size():
    p = cloud.vllm_plan(*A10G)
    cmd = cloud.vllm_command("layout", cloud.DOTS_MODEL, 8081, p, util=0.4, max_len=16384, seqs=8, eager=True)
    hint = json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1])["image"]
    assert hint == {"count": 1, "width": 1232, "height": 1596}
    cmd = cloud.vllm_command("vlm", p["vlm_model"], 8082, p, util=0.5, max_len=8192, seqs=8, eager=False)
    assert json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1])["image"]["width"] == 1140


def test_is_oom():
    assert cloud.is_oom(OOM_LOG) and cloud.is_oom("torch.OutOfMemoryError: CUDA out of memory.")
    assert not cloud.is_oom("FileNotFoundError: ninja")


def test_shrink_ladder_vlm_then_layout():
    st = cloud.vllm_plan(*A10G)
    seen = []
    while (st2 := cloud.shrink_plan(st, "vlm")) is not None:
        seen.append((st2["vlm_model"], st2["max_num_seqs"], st2["max_model_len"]))
        st = st2
    assert seen[0] == ("Qwen/Qwen3-VL-4B-Instruct-FP8", 8, 8192)
    assert ("Qwen/Qwen3-VL-2B-Instruct", 8, 4096) == seen[-1]
    lay = cloud.shrink_plan(cloud.vllm_plan(*A10G), "layout")
    assert lay["layout_model"] == cloud.LAYOUT_MODEL and lay["layout_gpu_util"] == 0.12
    assert lay["vlm_gpu_util"] > cloud.vllm_plan(*A10G)["vlm_gpu_util"]
    assert cloud.shrink_plan(lay, "layout") is None


def test_ladder_skips_fp8_on_t4():
    st = {**cloud.vllm_plan(15360, 7.5), "max_num_seqs": 8}
    assert cloud.shrink_plan(st, "vlm")["vlm_model"] == "Qwen/Qwen3-VL-2B-Instruct"


def test_start_pair_retries_after_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "STATE", tmp_path / ".cloud.json")
    monkeypatch.setattr(cloud, "_kill_pidfile", lambda name: None)
    monkeypatch.setattr(cloud, "_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(cloud.time, "sleep", lambda s: None)
    calls = []

    def fake_start(name, model, port, st, timeout, **kw):
        calls.append((name, model, kw["seqs"]))
        if name == "vlm" and len([c for c in calls if c[0] == "vlm"]) == 1:
            raise cloud.ServerFailed("vllm vlm exited", OOM_LOG)

    monkeypatch.setattr(cloud, "_start_vllm", fake_start)
    st = cloud.vllm_plan(*A10G)
    cloud._start_vllm_pair(st, timeout=1)
    assert [c[0] for c in calls] == ["layout", "vlm", "layout", "vlm"]
    assert calls[1][2] == 12 and calls[3][2] == 8             # concurrency halved on retry
    assert json.loads((tmp_path / ".cloud.json").read_text())["max_num_seqs"] == 8


def test_start_pair_reraises_non_oom(monkeypatch):
    def boom(name, *a, **k):
        raise cloud.ServerFailed("bad", "ImportError: something else")

    monkeypatch.setattr(cloud, "_start_vllm", boom)
    with pytest.raises(cloud.ServerFailed):
        cloud._start_vllm_pair(cloud.vllm_plan(*A10G), timeout=1)
