"""Speech recognition behind one interface, backed by `siftasr` and picked per machine and per language.

| machine        | en (+ 24 EU languages) | Vietnamese (vi) | Chinese (zh / yue)         | anything else / low LID prob   |
|----------------|------------------------|-----------------|----------------------------|--------------------------------|
| Apple Silicon  | parakeet-mlx           | quan-coreml     | qwen3-mlx (or mlx-whisper) | mlx-whisper                    |
| NVIDIA CUDA    | faster-whisper (batch) | (same)          | (same)                     | faster-whisper, batched on GPU |
| other          | faster-whisper on CPU  | (same)          | (same)                     | faster-whisper on CPU          |
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

try:
    from siftasr import asr as _sift
except ModuleNotFoundError:
    _sift_src = Path(__file__).resolve().parents[4] / "libs" / "siftasr" / "src"
    if _sift_src.is_dir():
        sys.path.insert(0, str(_sift_src))
    from siftasr import asr as _sift

from .transcript import Cue

SR = _sift.SR
PARAKEET_LANGS = _sift.PARAKEET_LANGS
ZH_LANGS = _sift.ZH_LANGS
DEFAULT_MODEL = _sift.DEFAULT_MODEL
QUAN_REPO = _sift.QUAN_REPO
QUAN_REV = _sift.QUAN_REV
QUAN_PKG = _sift.QUAN_PKG
QUAN_FN = _sift.QUAN_FN
FRAME_S = _sift.FRAME_S

AsrResult = _sift.AsrResult
Word = _sift.Word
_has = _sift._has
apple_silicon = _sift.apple_silicon
cuda_available = _sift.cuda_available
_float = _sift._float
detect_language_info = _sift.detect_language_info
detect_language = _sift.detect_language
split_quiet = _sift.split_quiet
ctc_words = _sift.ctc_words
words_to_cues = _sift.words_to_cues
diarize_cues = _sift.diarize_cues


def _log(msg: str) -> None:
    print(f"mediasift: {msg}", file=sys.stderr)


def quan_available() -> bool:
    from . import coreml
    return coreml.available()


def pick(requested: str, language: str | None) -> str:
    if requested != "auto":
        return requested
    if apple_silicon():
        if language == "vi" and quan_available():
            return "quan-coreml"
        if language in ZH_LANGS and _has("mlx_qwen3_asr"):
            return "qwen3-mlx"
        if language in PARAKEET_LANGS and _has("parakeet_mlx"):
            return "parakeet-mlx"
        if _has("mlx_whisper"):
            return "mlx-whisper"
    if not _has("faster_whisper"):
        raise SystemExit("--asr needs a backend: `uv tool install 'mediasift[asr]'` (and '[mlx]' on Apple Silicon)")
    return "faster-whisper"


def transcribe(
    pcm: np.ndarray,
    backend: str = "auto",
    model: str | None = None,
    language: str | None = None,
    *,
    diarize: bool = False,
) -> AsrResult:
    if language is None and backend in ("auto", "mlx-whisper", "faster-whisper"):
        language = detect_language(pcm)
        _log(f"detected language: {language}")
    name = pick(backend, language)
    t0 = time.perf_counter()
    if name == "parakeet-mlx":
        cues, used = _sift._parakeet_mlx(pcm, model or DEFAULT_MODEL[name])
    elif name == "qwen3-mlx":
        cues, used = _sift._qwen3_mlx(pcm, model or DEFAULT_MODEL[name], language)
    elif name == "mlx-whisper":
        cues, used = _sift._mlx_whisper(pcm, model or DEFAULT_MODEL[name], language)
    elif name == "faster-whisper":
        cues, used = _sift._faster_whisper(pcm, model, language)
    elif name == "quan-coreml":
        cues, used = _sift._quan_coreml(pcm)
    else:
        raise SystemExit(f"unknown ASR backend {name!r}")
    dt = time.perf_counter() - t0
    _log(f"ASR {name} ({used}): {len(pcm) / SR / max(dt, 1e-6):.1f}x realtime")
    res = AsrResult([Cue(c.start, c.end, c.text, list(getattr(c, "words", []) or []), getattr(c, "voice", None))
                     for c in cues], name, used, language, round(dt, 1))
    if diarize and cues:
        _, res.voices = diarize_cues(pcm, cues)
    return res


def remote(pcm: np.ndarray, host: str, language: str | None = None, connect_s: int = 3) -> AsrResult | None:
    res = _sift.remote(pcm, host, language=language, connect_s=connect_s, remote_bin="mediasift")
    if res is None:
        return None
    out = AsrResult([Cue(c.start, c.end, c.text, list(getattr(c, "words", []) or []), getattr(c, "voice", None))
                     for c in res.cues], res.backend, res.model, res.language, res.seconds)
    out.voices = res.voices
    return out


def serve_stdin(language: str | None) -> None:
    data = sys.stdin.buffer.read()
    pcm = np.frombuffer(
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
            input=data,
            capture_output=True,
            check=True,
        ).stdout,
        np.int16,
    )
    r = transcribe(pcm, "auto", None, language)
    from .sound import voices
    who = voices(pcm, [(c.start, c.end) for c in r.cues])
    print(
        json.dumps(
            {
                "cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in r.cues],
                "voices": who,
                "backend": r.backend,
                "model": r.model,
                "language": r.language,
                "seconds": r.seconds,
            },
            ensure_ascii=False,
        )
    )
