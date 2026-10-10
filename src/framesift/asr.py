"""Speech recognition behind one interface, picked per machine and per language.

| machine        | language covered by Parakeet v3 | anything else (e.g. Vietnamese) |
|----------------|---------------------------------|---------------------------------|
| Apple Silicon  | parakeet-mlx                    | mlx-whisper                     |
| NVIDIA CUDA    | faster-whisper, batched on GPU  | faster-whisper, batched on GPU  |
| other          | faster-whisper on CPU           | faster-whisper on CPU           |

Every backend takes the mono 16 kHz int16 buffer framesift already decoded,
so ffmpeg stays the only decoder.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
import wave
from dataclasses import dataclass

import numpy as np

from .transcript import Cue

SR = 16000

# Parakeet TDT 0.6B v3: 25 European languages (NVIDIA model card).
PARAKEET_LANGS = {"bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt",
                  "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"}

DEFAULT_MODEL = {
    "parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-whisper": "mlx-community/whisper-large-v3-turbo",
    "faster-whisper-cuda": "large-v3-turbo",
    "faster-whisper-cpu": "small",
}


@dataclass
class AsrResult:
    cues: list[Cue]
    backend: str
    model: str
    language: str | None
    seconds: float
    voices: list | None = None        # speaker letters per cue, when the transcriber computed them


def _log(msg: str) -> None:
    print(f"framesift: {msg}", file=sys.stderr)


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0 if pcm.dtype.kind == "i" else pcm.astype(np.float32)


def detect_language(pcm: np.ndarray) -> str | None:
    """Whisper-tiny on 30 s taken a fifth of the way in (past intros and
    music beds). Cheap on any machine; returns an ISO 639-1 code."""
    if not _has("faster_whisper"):
        return None
    from faster_whisper import WhisperModel
    start = int(len(pcm) * 0.2)
    clip = _float(pcm[start:start + 30 * SR])
    if len(clip) < SR:
        clip = _float(pcm[:30 * SR])
    _, info = WhisperModel("tiny", device="cpu", compute_type="int8").transcribe(clip, beam_size=1)
    return info.language


def pick(requested: str, language: str | None) -> str:
    if requested != "auto":
        return requested
    if apple_silicon():
        if language in PARAKEET_LANGS and _has("parakeet_mlx"):
            return "parakeet-mlx"
        if _has("mlx_whisper"):
            return "mlx-whisper"
    if not _has("faster_whisper"):
        raise SystemExit("--asr needs a backend: `uv tool install 'framesift[asr]'` (and '[mlx]' on Apple Silicon)")
    return "faster-whisper"


def transcribe(pcm: np.ndarray, backend: str = "auto", model: str | None = None,
               language: str | None = None) -> AsrResult:
    if language is None and backend in ("auto", "mlx-whisper", "faster-whisper"):
        language = detect_language(pcm)
        _log(f"detected language: {language}")
    name = pick(backend, language)
    t0 = time.perf_counter()
    if name == "parakeet-mlx":
        cues, used = _parakeet_mlx(pcm, model or DEFAULT_MODEL[name])
    elif name == "mlx-whisper":
        cues, used = _mlx_whisper(pcm, model or DEFAULT_MODEL[name], language)
    elif name == "faster-whisper":
        cues, used = _faster_whisper(pcm, model, language)
    else:
        raise SystemExit(f"unknown ASR backend {name!r}")
    dt = time.perf_counter() - t0
    _log(f"ASR {name} ({used}): {len(pcm) / SR / max(dt, 1e-6):.1f}x realtime")
    return AsrResult(cues, name, used, language, round(dt, 1))


def _mlx_lean() -> None:
    """MLX keeps freed GPU buffers in a cache: 5.5 GB after a 2-minute file on an M1 Pro,
    on top of a 3.1 GB peak, which pushed a 16 GB Mac into swap. No cache costs no speed."""
    import mlx.core as mx
    mx.set_cache_limit(0)


def _parakeet_mlx(pcm: np.ndarray, model: str) -> tuple[list[Cue], str]:
    _mlx_lean()
    from parakeet_mlx import from_pretrained
    m = from_pretrained(os.path.expanduser(model))
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.astype(np.int16).tobytes())
        # Chunked so memory stays flat on hour-long files; the library merges overlaps.
        r = m.transcribe(f.name, chunk_duration=120.0, overlap_duration=15.0)
    return [Cue(float(s.start), float(s.end), s.text.strip()) for s in r.sentences if s.text.strip()], model


def _mlx_whisper(pcm: np.ndarray, model: str, language: str | None) -> tuple[list[Cue], str]:
    _mlx_lean()
    import mlx_whisper
    # condition_on_previous_text=False avoids the known repeated-sentence loops on long audio.
    r = mlx_whisper.transcribe(_float(pcm), path_or_hf_repo=model, language=language,
                               condition_on_previous_text=False)
    return [Cue(float(s["start"]), float(s["end"]), s["text"].strip()) for s in r["segments"] if s["text"].strip()], model


def _faster_whisper(pcm: np.ndarray, model: str | None, language: str | None) -> tuple[list[Cue], str]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    audio = _float(pcm)
    if cuda_available():
        # Batched decoding (~53x realtime on an RTX 3060 with turbo) returns
        # ~30 s chunks; word timestamps are free and give sentence-sized cues.
        name = model or DEFAULT_MODEL["faster-whisper-cuda"]
        wm = WhisperModel(name, device="cuda", compute_type="float16")
        segs, _ = BatchedInferencePipeline(model=wm).transcribe(audio, batch_size=16, language=language,
                                                                word_timestamps=True)
        return words_to_cues([(w.start, w.end, w.word) for s in segs for w in (s.words or [])]), name
    name = model or DEFAULT_MODEL["faster-whisper-cpu"]
    wm = WhisperModel(name, device="cpu", compute_type="int8")
    # Word timestamps: segment times alone gave one cue spanning 4-29 s across a long silence.
    segs, _ = wm.transcribe(audio, vad_filter=True, language=language, word_timestamps=True)
    return words_to_cues([(w.start, w.end, w.word) for s in segs for w in (s.words or [])]), name


def words_to_cues(words: list[tuple[float, float, str]], max_len: float = 8.0, gap: float = 0.6) -> list[Cue]:
    """Group timed words into sentence-sized cues: break after . ? !, at a
    pause longer than `gap`, or once a cue reaches `max_len` seconds."""
    cues: list[Cue] = []
    cur: list[tuple[float, float, str]] = []
    for w in words:
        if cur and (w[0] - cur[-1][1] > gap or w[1] - cur[0][0] > max_len):
            cues.append(Cue(float(cur[0][0]), float(cur[-1][1]), "".join(x[2] for x in cur).strip()))
            cur = []
        cur.append(w)
        if w[2].strip().endswith((".", "?", "!", "。", "？", "！")):
            cues.append(Cue(float(cur[0][0]), float(cur[-1][1]), "".join(x[2] for x in cur).strip()))
            cur = []
    if cur:
        cues.append(Cue(float(cur[0][0]), float(cur[-1][1]), "".join(x[2] for x in cur).strip()))
    return [c for c in cues if c.text]


def remote(pcm: np.ndarray, host: str, language: str | None = None, connect_s: int = 3) -> AsrResult | None:
    """Transcribe on another machine (a GPU box) over ssh: the audio goes as FLAC on stdin
    (about 30 MB per hour), `framesift asr -` runs there, cues come back as JSON. Returns None
    when the host does not answer within `connect_s`, so the caller can fall back to local.
    The Mac then spends no GPU or memory on speech at all."""
    import json
    import subprocess
    if host != "local":   # "local" runs the same protocol without ssh (tests)
        ok = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_s}", host, "true"],
                            capture_output=True).returncode == 0
        if not ok:
            _log(f"{host} did not answer in {connect_s}s: transcribing here")
            return None
    flac = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                           "-c:a", "flac", "-f", "flac", "-"], input=pcm.astype(np.int16).tobytes(),
                          capture_output=True, check=True).stdout
    remote_cmd = "framesift asr -" + (f" --lang {language}" if language else "")
    cmd = ["framesift", "asr", "-"] + (["--lang", language] if language else []) if host == "local" else \
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_s}", "-o", "ServerAliveInterval=15",
         "-o", "ServerAliveCountMax=4", host, f"bash -lc '{remote_cmd}'"]
    t0 = time.perf_counter()
    # A stalled box must not hang `see`: allow 10x realtime plus model load, then give up loudly.
    r = subprocess.run(cmd, input=flac, capture_output=True, timeout=120 + len(pcm) / SR / 10)
    if r.returncode != 0:
        raise RuntimeError(f"ASR on {host} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-1500:]}")
    out_lines = r.stdout.decode(errors="replace").strip().splitlines()
    if not out_lines:
        raise RuntimeError(f"ASR on {host} printed nothing: {r.stderr.decode(errors='replace')[-800:]}")
    js = json.loads(out_lines[-1])        # a login shell may print a banner before the JSON
    cues = [Cue(float(a), float(b), t) for a, b, t in js["cues"]]
    remote_voices = js.get("voices")
    dt = time.perf_counter() - t0
    _log(f"ASR on {host}: {js['backend']} ({js['model']}), {len(pcm) / SR / max(dt, 1e-6):.1f}x realtime incl. transfer")
    res = AsrResult(cues, f"{js['backend']}@{host}", js["model"], js.get("language"), round(dt, 1))
    res.voices = remote_voices if remote_voices and len(remote_voices) == len(cues) else None
    return res


def serve_stdin(language: str | None) -> None:
    """`framesift asr -`: audio in any ffmpeg-readable format on stdin, cues as JSON on stdout."""
    import json
    import subprocess
    data = sys.stdin.buffer.read()
    pcm = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                                       input=data, capture_output=True, check=True).stdout, np.int16)
    r = transcribe(pcm, "auto", None, language)
    from .sound import voices          # the host has the audio already: group speakers here too
    who = voices(pcm, [(c.start, c.end) for c in r.cues])
    print(json.dumps({"cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in r.cues], "voices": who,
                      "backend": r.backend, "model": r.model, "language": r.language, "seconds": r.seconds},
                     ensure_ascii=False))
