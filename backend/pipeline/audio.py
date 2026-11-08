"""Audio storage: MP3 by default (constant bitrate, so seeking to a sentence lands exactly).

Kokoro speaks at 24 kHz mono. At the default ~32 kbps an hour of narration is about 14 MB,
against about 170 MB as 16-bit WAV.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

import config

EXTS = ("mp3", "wav")


def write(path: Path, samples: np.ndarray, sr: int):
    """Write samples to path; the format follows the suffix (.mp3 or .wav)."""
    path.write_bytes(encode(samples, sr, path.suffix.lstrip(".")))


def encode(samples: np.ndarray, sr: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, samples, sr, format="MP3", subtype="MPEG_LAYER_III",
                 compression_level=config.MP3_LEVEL, bitrate_mode="CONSTANT")
    elif fmt == "wav":
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    else:
        raise ValueError(f"unknown audio format {fmt!r}")
    return buf.getvalue()


def convert(src: Path, fmt: str) -> bytes:
    """src re-encoded as fmt (bytes). Returns the file unchanged if it is already in fmt."""
    if src.suffix.lstrip(".") == fmt:
        return src.read_bytes()
    samples, sr = sf.read(src, dtype="float32")
    return encode(samples, sr, fmt)


def existing(audio_dir: Path, bid: str) -> Path | None:
    """The block's audio file in any supported format, if present."""
    return next((p for p in (audio_dir / f"{bid}.{e}" for e in EXTS) if p.exists()), None)


def target(audio_dir: Path, bid: str) -> Path:
    return audio_dir / f"{bid}.{config.AUDIO_FORMAT}"
