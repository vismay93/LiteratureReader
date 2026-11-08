"""SGLang engine plumbing in notebooks/cloud.py (pure functions; no GPU needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import cloud  # noqa: E402


def _plan(vram=23034, cc=8.9):
    return {**cloud.vllm_plan(vram, cc), "engine": "sglang", "dots_chat_template": "/x/t.jinja"}


def test_requirements_gate():
    assert "CUDA 13" in cloud.sglang_requirements_ok(12.4, (2, 35))
    assert "glibc" in cloud.sglang_requirements_ok(13.0, (2, 26))
    assert cloud.sglang_requirements_ok(13.0, (2, 35)) is None


def test_second_server_fraction_is_relative_to_free_memory():
    # 0.43 of the GPU taken by layout -> narration's 0.47 of total is ~0.82 of what's left
    assert cloud.sglang_vlm_fraction(0.43, 0.47) == 0.82
    assert cloud.sglang_vlm_fraction(0.12, 0.78) == 0.88      # capped, leaves activation slack
    assert cloud.sglang_vlm_fraction(0.9, 0.5) <= 0.88


def test_sglang_command_dots():
    st = _plan()
    cmd = cloud.sglang_command("layout", cloud.DOTS_MODEL, 8081, st, mem_fraction=0.43,
                               max_len=32768, seqs=16, eager=True)
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--served-model-name") + 1] == "layout"
    assert cmd[cmd.index("--revision") + 1] == cloud.DOTS_REVISION and "--trust-remote-code" in cmd
    assert cmd[cmd.index("--chat-template") + 1] == "/x/t.jinja" and "--disable-cuda-graph" in cmd


def test_sglang_command_vlm_has_no_remote_code():
    cmd = cloud.sglang_command("vlm", "Qwen/Qwen3.8-27B-FP8", 8082, _plan(81559, 9.0),
                               mem_fraction=0.88, max_len=32768, seqs=32, eager=False)
    assert "--trust-remote-code" not in cmd and "--chat-template" not in cmd
    assert "--disable-cuda-graph" not in cmd


def test_engine_env_sglang_drops_manual_placeholder():
    env = cloud._engine_env(_plan())
    assert env["PR_DOTS_IMAGE_PLACEHOLDER"] == "0" and env["PR_LAYOUT_MODEL"] == "dots"
    assert cloud._engine_env({**_plan(), "engine": "vllm"})["PR_DOTS_IMAGE_PLACEHOLDER"] == "1"


def test_server_env_avoids_flashinfer_jit(tmp_path):
    env = cloud.server_env("vllm", {"hf_home": "/h"}, tmp_path)
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PATH"].split(":")[0] == str(tmp_path) and env["HF_HOME"] == "/h"
    sg = cloud.server_env("sglang", {}, tmp_path)
    assert sg["PATH"].split(":")[0] == str(tmp_path)


def test_sglang_never_uses_flashinfer_jit_kernels(monkeypatch):
    # A conda nvcc without CUDA headers broke FlashInfer's attention JIT on SageMaker:
    # the command must pick JIT-free backends regardless of whether nvcc is on PATH.
    monkeypatch.setattr(cloud.shutil, "which", lambda name: "/opt/conda/bin/nvcc")
    cmd = cloud.sglang_command("layout", cloud.DOTS_MODEL, 8081, _plan(), mem_fraction=0.4,
                               max_len=32768, seqs=16, eager=True)
    assert cmd[cmd.index("--sampling-backend") + 1] == "pytorch"
    assert cmd[cmd.index("--attention-backend") + 1] == "triton"
    assert cmd[cmd.index("--mm-attention-backend") + 1] == "sdpa"
    assert "flashinfer" not in " ".join(cmd)
