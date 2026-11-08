"""Text-to-speech with Kokoro-82M (ONNX). One WAV per block plus per-sentence timings."""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import numpy as np

import config
from . import audio

# espeak's word-count mismatch warning fires on most sentences and is harmless.
logging.getLogger("phonemizer").setLevel(logging.ERROR)

_lock = threading.Lock()
_kokoro = None

# Split after . ! ? followed by space + capital/digit, avoiding common abbreviations.
_ABBREV = r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bet al\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bSec\.)(?<!\bNo\.)"
_SENT_RE = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    return parts or [text]


def tts_providers() -> list[str]:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if config.TTS_PROVIDERS != "auto":
        return [p.strip() for p in config.TTS_PROVIDERS.split(",") if p.strip()]
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]


def _engine():
    global _kokoro
    if _kokoro is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        providers = tts_providers()
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # load CUDA/cuDNN from the pip nvidia-* wheels
        sess = ort.InferenceSession(str(config.KOKORO_MODEL), providers=providers)
        _kokoro = Kokoro.from_session(sess, str(config.KOKORO_VOICES))
    return _kokoro


def synthesize(text: str, out: Path, voice: str = config.KOKORO_VOICE, speed: float = 1.0) -> dict:
    """Write out (.mp3 or .wav); return {'duration': s, 'sentences': [{'start','end','text'}]}."""
    with _lock:  # onnxruntime session already uses all cores; serialize calls
        k = _engine()
        chunks, sents, t = [], [], 0.0
        sr = 24000
        gap = None
        for s in split_sentences(text):
            samples, sr = k.create(s, voice=voice, speed=speed, lang="en-us")
            if gap is None:
                gap = np.zeros(int(sr * 0.18), dtype=np.float32)
            dur = len(samples) / sr
            sents.append({"start": round(t, 3), "end": round(t + dur, 3), "text": s})
            chunks += [samples.astype(np.float32), gap]
            t += dur + len(gap) / sr
        wav = np.concatenate(chunks) if chunks else np.zeros(sr // 2, dtype=np.float32)
        audio.write(Path(out), wav, sr)
        return {"duration": round(len(wav) / sr, 3), "sentences": sents}
