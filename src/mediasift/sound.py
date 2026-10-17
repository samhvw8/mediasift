"""Non-speech sound made legible: what kind of sound, when, how many times.

The agent cannot hear, so every sound that matters becomes a line of text in
the style of subtitles for the deaf: `[music]`, `[siren]`, `[beep x5]`,
`[silence 3.0 s]`. Three cheap sources:

- CED (a distilled AudioSet tagger, 527 classes, ONNX via sherpa-onnx) on 1 s
  windows names the sound. ~150x realtime on an M1 Pro CPU.
- Onsets from spectral flux find discrete short sounds (clicks, beeps, thuds)
  and count them; CED names each group.
- Energy finds silences; CED's male/female speech classes tag each voice.

The model is a 22 MB download on first use, pinned by SHA-256.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import polite

SR = 16000
# 3 s windows: the tagger was trained on 10 s clips, and on 40 real ESC-50 recordings 1 s windows
# named 25, 2 s 31, 3 s 36 (5 s 36 with blurrier times). Spans are trimmed to audible audio after.
WIN_S = 3.0
MUSIC_MIN_S = 4.5   # shorter 'music' with few onsets is a short sound seen through a 3 s window
MODEL = "sherpa-onnx-ced-small-audio-tagging-2024-04-19"   # ced-base tied it on real sounds (25/40) at 4x size
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/{MODEL}.tar.bz2"
MODEL_SHA256 = "a3c9ef3e7f8dcf2720b148bb3c32a882a896a4abb00450a6c091772c54d11ef5"

SPEECH = {0, 1, 2, 3, 4, 5, 7, 70}          # speech, male, female, child, conversation, narration, synth voice, babble
MALE, FEMALE, CHILD = 1, 2, 3
SINGING = set(range(27, 36))
MUSIC = set(range(137, 283))                 # AudioSet's music branch: instruments, genres, moods
SILENCE = 500
# Generic labels in the music branch; anything else there (a church bell, a timpani roll) is named.
MUSIC_GENERIC = {"Music", "Musical instrument", "Background music", "Theme music", "Soundtrack music", "Video game music"}
# Too generic to name a sound on their own.
VAGUE = {"Sound effect", "Inside, small room", "Inside, large room or hall", "Outside, urban or manmade",
         "Outside, rural or natural", "Environmental noise", "Static", "Noise", "White noise", "Pink noise",
         "Mains hum", "Hum", "Echo", "Reverberation", "Single-lens reflex camera", "Silence", "Cacophony"}


def _log(msg: str) -> None:
    print(f"mediasift: {msg}", file=sys.stderr)


def cache_root() -> Path:
    base = os.environ.get("MEDIASIFT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "mediasift")
    return Path(base)


def _fetch(url: str, sha256: str, dest: Path, what: str) -> Path:
    """Download once into the cache, verify the pinned SHA-256, unpack tarballs. Crash on mismatch."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (Path(url).name + ".part")
    _log(f"downloading {what} (once) from {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != sha256:
        tmp.unlink()
        raise RuntimeError(f"{what} download has SHA-256 {digest}, expected {sha256}")
    if url.endswith(".tar.bz2"):
        with tarfile.open(tmp) as t:
            t.extractall(dest.parent, filter="data")
        tmp.unlink()
    else:
        tmp.replace(dest)
    return dest


def model_dir() -> Path:
    d = cache_root() / "models" / MODEL
    if not (d / "model.int8.onnx").exists():
        _fetch(MODEL_URL, MODEL_SHA256, d, "the sound tagger (22 MB)")
    return d


SPEAKER = "nemo_en_titanet_small.onnx"
SPEAKER_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{SPEAKER}"
SPEAKER_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"


def speaker_model() -> Path:
    f = cache_root() / "models" / SPEAKER
    return f if f.exists() else _fetch(SPEAKER_URL, SPEAKER_SHA256, f, "the speaker model (40 MB)")


@dataclass
class Tags:
    """CED class probabilities on a uniform grid of 1 s windows."""
    times: np.ndarray        # (W,) window centres
    probs: np.ndarray        # (W, 527) float16, only the top classes per window are non-zero
    names: list[str]
    hop: float

    def family(self, ids: set[int]) -> np.ndarray:
        return self.probs[:, sorted(ids)].max(axis=1).astype(np.float32)


@lru_cache(maxsize=2)
def _tagger(top_k: int):
    import sherpa_onnx
    d = model_dir()
    cfg = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(ced=str(d / "model.int8.onnx"), num_threads=polite.threads(4)),
        labels=str(d / "class_labels_indices.csv"), top_k=top_k)
    return sherpa_onnx.AudioTagging(cfg)


def speech_score(pcm: np.ndarray, t: float, win: float = 1.0) -> float:
    """How much CED hears speech in a short window at t (from 0.1 s before)."""
    a = max(0, int((t - 0.1) * SR))
    seg = pcm[a:a + int(win * SR)]
    st = _tagger(20).create_stream()
    st.accept_waveform(SR, seg.astype(np.float32) / 32768.0 if seg.dtype.kind == "i" else seg.astype(np.float32))
    return max((e.prob for e in _tagger(20).compute(st) if e.index in SPEECH), default=0.0)


# The same CED-small as a batched Core ML model (GPU), built once by `mise run ced-coreml`
# (scripts/ced_coreml.py: coremltools has no Python 3.14 build). 47 min of audio at a 1 s hop:
# 27.9 s on the CPU through sherpa-onnx, 6.2 s here.
CED_REV = "06bb40c5"            # mispeech/ced-small revision the package was converted from
CED_BATCH, CED_FRAMES, NFFT, HOP_SAMPLES, N_MELS = 64, 301, 512, 160, 64


def ced_coreml_package() -> Path:
    return cache_root() / "models" / f"ced-small-coreml-{CED_REV}" / "ced_small_b64.mlpackage"


@lru_cache(maxsize=1)
def _ced_coreml():
    from . import coreml
    if not coreml.available():
        return None
    pkg = ced_coreml_package()
    if not pkg.exists():
        _log("sound tagger on the CPU: `mise run ced-coreml` builds the 4.5x faster Core ML one")
        return None
    return coreml.load(pkg)


@lru_cache(maxsize=1)
def _mel_bank() -> tuple[np.ndarray, np.ndarray]:
    """Hann window (periodic) and torchaudio's HTK mel filters (no norm), 257 bins -> 64 mels, 0-8 kHz."""
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32)
    freqs = np.linspace(0, SR // 2, NFFT // 2 + 1)
    mel = np.linspace(2595.0 * np.log10(1.0), 2595.0 * np.log10(1.0 + (SR / 2) / 700.0), N_MELS + 2)
    f_pts = 700.0 * (10 ** (mel / 2595.0) - 1.0)
    slopes = f_pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / np.diff(f_pts)[:-1]
    up = slopes[:, 2:] / np.diff(f_pts)[1:]
    return window, np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


def _frame_mels(frames: np.ndarray) -> np.ndarray:
    """(..., 512) raw frames -> (..., 64) mel power (Hann window, |rfft|^2, HTK mel bank)."""
    window, bank = _mel_bank()
    spec = np.fft.rfft(frames * window, n=NFFT, axis=-1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32) @ bank


def _db(mel: np.ndarray) -> np.ndarray:
    """(N, 64, T) mel power -> dB, floored 120 dB under each window's own peak (AmplitudeToDB top_db=120)."""
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return np.maximum(db, db.max(axis=(1, 2), keepdims=True) - 120.0).astype(np.float32)


def log_mel(windows: np.ndarray) -> np.ndarray:
    """(N, 48000) waveforms -> (N, 64, 301) dB: CED's torchaudio features (MelSpectrogram with center and
    reflect padding, AmplitudeToDB top_db=120 per window). Kept in numpy: frame power overflows fp16."""
    x = np.pad(windows, ((0, 0), (NFFT // 2, NFFT // 2)), mode="reflect")
    idx = np.arange(0, x.shape[1] - NFFT + 1, HOP_SAMPLES)[:, None] + np.arange(NFFT)[None, :]
    return _db(_frame_mels(x[:, idx]).transpose(0, 2, 1))


def window_log_mels(y: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """log_mel of the 3 s windows starting at `starts`, sharing frames: overlapping windows repeat
    frames 2-298 (3x at a 1 s hop, 6x at 0.5 s), which one STFT over their span gives exactly; only
    frames 0, 1, 299 and 300 touch each window's reflect padding and are computed per window."""
    win, half = int(WIN_S * SR), NFFT // 2
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0 or starts[-1] + win > len(y) or (starts % HOP_SAMPLES).any():
        segs = np.zeros((len(starts), win), np.float32)
        for j, s0 in enumerate(starts):
            seg = y[s0:s0 + win]
            segs[j, :len(seg)] = seg
        return log_mel(segs)
    first = starts // HOP_SAMPLES                          # global frame index of each window's frame 0
    g0, g1 = int(first[0]) + 2, int(first[-1]) + CED_FRAMES - 3
    mels = np.empty((g1 - g0 + 1, N_MELS), np.float32)
    for a in range(g0, g1 + 1, 4096):                      # chunks keep the frame array under ~10 MB
        g = np.arange(a, min(a + 4096, g1 + 1))
        mels[a - g0:a - g0 + len(g)] = _frame_mels(y[(g * HOP_SAMPLES - half)[:, None] + np.arange(NFFT)])
    out = np.empty((len(starts), CED_FRAMES, N_MELS), np.float32)
    out[:, 2:CED_FRAMES - 2] = mels[(first - g0)[:, None] + np.arange(2, CED_FRAMES - 2)]
    # The padded window's first and last 672 samples hold its edge frames (reflection of the window).
    head = starts[:, None] + np.concatenate([np.arange(half, 0, -1), np.arange(672 - half)])
    tail = starts[:, None] + np.concatenate([np.arange(win - 672 + half, win), np.arange(win - 2, win - 2 - half, -1)])
    ends = []
    for ix in (head, tail):
        seg = y[ix]
        ends.append(_frame_mels(np.stack([seg[:, :NFFT], seg[:, HOP_SAMPLES:HOP_SAMPLES + NFFT]], axis=1)))
    out[:, :2], out[:, CED_FRAMES - 2:] = ends
    return _db(out.transpose(0, 2, 1))


@lru_cache(maxsize=1)
def _labels() -> list[str]:
    import csv
    with open(model_dir() / "class_labels_indices.csv", newline="") as f:
        return [r["display_name"] for r in csv.DictReader(f)]


def _tag_coreml(model, y: np.ndarray, hop: float, top_k: int, names: list[str]) -> Tags:
    """Every window through the GPU in batches of 64; each window keeps only its top_k classes,
    like the sherpa tagger, so the naming rules see the same kind of evidence."""
    from . import coreml
    win, step = int(WIN_S * SR), int(hop * SR)
    n = max(1, 1 + (len(y) - win) // step)
    probs = np.zeros((n, 527), np.float16)
    for b0 in range(0, n, CED_BATCH):
        k = min(CED_BATCH, n - b0)
        mel = np.zeros((CED_BATCH, N_MELS, CED_FRAMES), np.float32)
        mel[:k] = window_log_mels(y, (b0 + np.arange(k)) * step)
        p = coreml.predict(model, {"mel": mel}, ["prob"])[0][:k]
        top = np.argpartition(-p, top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(k)[:, None]
        probs[b0 + rows, top] = p[rows, top]
    # A name only for classes some window kept, as sherpa fills them: the rules read "" as never heard.
    heard = (probs > 0).any(axis=0)
    return Tags(np.arange(n) * hop + WIN_S / 2, probs, [nm if h else "" for nm, h in zip(names, heard)], hop)


def tag(pcm: np.ndarray, hop: float = 0.5, top_k: int = 12) -> Tags:
    y = pcm.astype(np.float32) / 32768.0 if pcm.dtype.kind == "i" else pcm.astype(np.float32)
    m = _ced_coreml()
    if m is not None:
        return _tag_coreml(m, y, hop, top_k, _labels())
    tagger = _tagger(top_k)
    win, step = int(WIN_S * SR), int(hop * SR)
    n = max(1, 1 + (len(y) - win) // step)
    names = [""] * 527
    probs = np.zeros((n, 527), np.float16)
    for i in range(n):
        s = tagger.create_stream()
        s.accept_waveform(SR, y[i * step: i * step + win])
        for e in tagger.compute(s):
            probs[i, e.index] = e.prob
            names[e.index] = e.name
    return Tags(np.arange(n) * hop + WIN_S / 2, probs, names, hop)


@dataclass
class SoundEvent:
    start: float
    end: float
    label: str               # "music", "speech", "silence", or a CED class name
    kind: str                # "bed" (lasting sound), "hits" (counted short sounds), "silence"
    times: list[float] = field(default_factory=list)   # for hits: each onset
    score: float = 0.0
    quality: str = ""        # how it sounds: "low, dull" / "high, bright, ringing"; music: "scary, electronic"
    moods: list = field(default_factory=list)   # long music: [(start, end, "tender"), (start, end, "scary")]
    harmony: dict = field(default_factory=dict) # {"key": "C", "mode": "minor", "harmony_arc": [...]}
    rhythm: dict = field(default_factory=dict)  # {"bpm": 120.0, "pulse_clarity": 0.82}
    beat_sync: dict = field(default_factory=dict)  # {"synced": True, "hits": 5, "total": 6, "ratio": 0.833}

def _runs(mask: np.ndarray, times: np.ndarray, hop: float, gap: float, min_len: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for t, m in zip(times, mask):
        if not m:
            continue
        s, e = t - hop / 2, t + hop / 2
        if out and s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out if e - s >= min_len]


# CED's own mood and genre classes. On 742 held-out AudioSet eval clips (top 12 classes kept per
# window, as here) the top mood was right 88% of the time at >= 0.10 (35% of clips) and ~65% at
# 0.05-0.10 (chance 14%); the top genre 83% at >= 0.15. Open-vocabulary words ("epic") from CLAP
# or the MTG-Jamendo head scored lower on the same clips (decisions/music-and-voice-mood.md).
MOODS = {276: "happy", 277: "funny", 278: "sad", 279: "tender", 280: "exciting", 281: "angry", 282: "scary"}
GENRES = set(range(216, 266)) - {218}       # Pop music .. Independent music; not Beatboxing
MOOD_SURE, MOOD_MAYBE, GENRE_SURE = 0.10, 0.05, 0.15
ARC_MIN_S, ARC_STEP_S = 60.0, 15.0


def _mood(mean: np.ndarray, maybe: bool = True) -> str:
    i = max(MOODS, key=lambda k: mean[k])
    if mean[i] >= MOOD_SURE:
        return MOODS[i]
    return f"maybe {MOODS[i]}" if maybe and mean[i] >= MOOD_MAYBE else ""


def character(mean: np.ndarray, mood: bool = True) -> str:
    """'scary, electronic' for a music run's mean class scores; only what passed the bench's cut.
    Without the mood when a mood arc says it better."""
    g = max(GENRES, key=lambda k: mean[k])
    genre = _LABELS_PLAIN.get(g, "") if mean[g] >= GENRE_SURE else ""
    return ", ".join(w for w in (_mood(mean) if mood else "", genre) if w)


def mood_arc(tags: Tags, s: float, e: float) -> list[tuple[float, float, str]]:
    """A long music run's sure moods per 15 s stretch, merged; empty unless the mood changes."""
    if e - s < ARC_MIN_S:
        return []
    arc: list[list] = []
    for a in np.arange(s, e, ARC_STEP_S):
        m = (tags.times >= a) & (tags.times < min(a + ARC_STEP_S, e))
        word = _mood(tags.probs[m].astype(np.float32).mean(axis=0), maybe=False) if m.any() else ""
        b = float(min(a + ARC_STEP_S, e))
        if not word:
            continue
        if arc and arc[-1][2] == word and a - arc[-1][1] <= ARC_STEP_S:
            arc[-1][1] = b
        else:
            arc.append([float(a), b, word])
    return [tuple(x) for x in arc] if len({x[2] for x in arc}) > 1 else []


_LABELS_PLAIN = {216: "pop", 217: "hip hop", 219: "rock", 220: "heavy metal", 221: "punk", 222: "grunge",
                 223: "progressive rock", 224: "rock and roll", 225: "psychedelic rock", 226: "rhythm and blues",
                 227: "soul", 228: "reggae", 229: "country", 230: "swing", 231: "bluegrass", 232: "funk", 233: "folk",
                 234: "Middle Eastern", 235: "jazz", 236: "disco", 237: "classical", 238: "opera", 239: "electronic",
                 240: "house", 241: "techno", 242: "dubstep", 243: "drum and bass", 244: "electronica", 245: "EDM",
                 246: "ambient", 247: "trance", 248: "Latin American", 249: "salsa", 250: "flamenco", 251: "blues",
                 252: "children's", 253: "new-age", 254: "vocal", 255: "a cappella", 256: "African", 257: "afrobeat",
                 258: "Christian", 259: "gospel", 260: "Asian", 261: "Carnatic", 262: "Bollywood", 263: "ska",
                 264: "traditional", 265: "indie"}


def beds(tags: Tags, speech_spans: list[tuple[float, float]]) -> list[SoundEvent]:
    """Lasting sounds: music, and any named sound that holds for a second or more."""
    ev: list[SoundEvent] = []
    music = np.maximum(tags.family(MUSIC), tags.family(SINGING))
    for s, e in _runs(music > 0.35, tags.times, tags.hop, 1.0, 1.5):
        m = (tags.times >= s) & (tags.times <= e)
        sung = tags.family(SINGING)[m].mean() > 0.3
        label = "singing" if sung else "music"
        # The music branch also holds everyday sounds (church bells, a gong): name the specific
        # instrument when it beats the generic "Music" class over the run (real-sound bench).
        mean = tags.probs[m].astype(np.float32).mean(axis=0)
        spec = [i for i in sorted(MUSIC) if tags.names[i] and tags.names[i] not in MUSIC_GENERIC
                and not tags.names[i].endswith("music")]
        if spec and not sung:
            top = max(spec, key=lambda i: mean[i])
            generic = max(mean[i] for i in MUSIC if tags.names[i] in ("Music",)) if any(
                tags.names[i] == "Music" for i in MUSIC) else 0.0
            if mean[top] > generic:
                label = tags.names[top]
        arc = mood_arc(tags, s, e)
        ev.append(SoundEvent(s, e, label, "bed", score=float(music[m].mean()), quality=character(mean, not arc),
                             moods=arc))
    skip = SPEECH | MUSIC | SINGING | {SILENCE}
    other = [i for i in range(527) if i not in skip and tags.names[i] and tags.names[i] not in VAGUE]
    # Beside "Music" CED scores a real sound 0.2-0.3 (200-clip song bench), so while music plays
    # the naming threshold drops; in quiet audio it stays where the quiet bench set it.
    in_music = music > 0.35
    for i in other:
        p = tags.probs[:, i].astype(np.float32)
        if p.max() < 0.3 and not (in_music.any() and p[in_music].max() >= 0.2):
            continue
        runs = _runs(p > np.where(in_music, 0.18, 0.25), tags.times, tags.hop, 0.6, 1.0)
        # A run only the lower music threshold let in must last 1.5 s: the song bench's strays were 1.0 s.
        runs = [(a, b) for a, b in runs if b - a >= 1.5 or not in_music[(tags.times >= a) & (tags.times <= b)].any()
                or p[(tags.times >= a) & (tags.times <= b)].mean() >= 0.25]
        # One strong window is enough for an intermittent sound (a baby's cries, a rooster's crow).
        runs += [r for r in _runs(p > 0.5, tags.times, tags.hop, 0.0, 0.0)
                 if not any(a <= r[0] and r[1] <= b for a, b in runs)]
        for s, e in runs:
            m = (tags.times >= s) & (tags.times <= e)
            ev.append(SoundEvent(s, e, tags.names[i], "bed", score=float(p[m].mean())))
    named = [x for x in ev if x.label not in ("music", "singing")]
    def explained(m: SoundEvent) -> bool:
        return any(min(m.end, x.end) - max(m.start, x.start) > 0.5 * (m.end - m.start) and m.score < x.score + 0.2
                   for x in named)
    ev = [x for x in ev if x.label not in ("music", "singing") or not explained(x)]
    # Several classes often describe one sound ("Siren", "Police car"): keep the
    # strongest per overlapping stretch, list the runner-up as an alias.
    ev.sort(key=lambda x: -x.score)
    kept: list[SoundEvent] = []
    for x in ev:
        clash = next((k for k in kept if x.label not in ("music", "singing") and k.label not in ("music", "singing")
                      and min(k.end, x.end) - max(k.start, x.start) > 0.5 * min(k.end - k.start, x.end - x.start)),
                     None)
        if clash is None:
            kept.append(x)
        elif x.label.lower() not in clash.label.lower() and "/" not in clash.label:
            clash.label = f"{clash.label} / {x.label}"
    return sorted(kept, key=lambda x: x.start)


def silences(rms_db: np.ndarray, times: np.ndarray, sounds: list[float] = (), min_len: float = 1.5) -> list[SoundEvent]:
    """Stretches with nothing audible: absolute floor, or far below this recording's loud level.
    `sounds` are onset times (clicks, beeps) that break a silence even when too short to lift the energy."""
    loud = np.percentile(rms_db, 95)
    quiet = (rms_db < -55) | (rms_db < loud - 35)
    for t in sounds:
        quiet[(times > t - 0.1) & (times < t + 0.15)] = False
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    return [SoundEvent(s, e, "silence", "silence") for s, e in _runs(quiet, times, hop, 0.15, min_len)]


def onsets(pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times of sharp sound starts (spectral flux peaks) and a timbre vector for each."""
    y = pcm.astype(np.float32) / 32768.0 if pcm.dtype.kind == "i" else pcm.astype(np.float32)
    hop, win = 160, 512                                    # 10 ms hop
    n = max(0, 1 + (len(y) - win) // hop)
    if n < 3:
        return np.zeros(0), np.zeros((0, 24))
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop][:n]
    w = np.hanning(win).astype(np.float32)
    bands = np.round(np.geomspace(1, 257, 25)).astype(int)  # 24 log-spaced bands over ~30 Hz - 8 kHz
    for k in range(1, len(bands)):                          # at least one FFT bin per band
        bands[k] = max(bands[k], bands[k - 1] + 1)
    spec = np.empty((n, 24), np.float32)
    level = np.empty(n, np.float32)
    for a in range(0, n, 20000):                 # blocks: a whole-file (n, 512) copy is ~1 GB per hour
        chunk = frames[a:a + 20000]
        level[a:a + len(chunk)] = 10 * np.log10((chunk ** 2).mean(axis=1) + 1e-12)
        mag = np.abs(np.fft.rfft(chunk * w))
        spec[a:a + len(mag)] = np.stack([mag[:, bands[k]:bands[k + 1]].mean(axis=1) for k in range(24)], axis=1)
    logs = np.log(spec + 1e-6)
    flux = np.maximum(0, np.diff(logs, axis=0, prepend=logs[:1])).sum(axis=1)
    # A hit is a short sound: the level jumps well above what came just before and
    # falls back soon after. Sustained sounds (a ring, a siren, music) and the
    # warble inside them fail the fall-back test; a sound's end fails the jump test.
    k = 50
    pad = np.pad(flux, k, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1), axis=1)[:n]
    thr = med + max(2.0, 3 * float(np.median(np.abs(flux - np.median(flux)))))
    jump = np.zeros(n, bool)
    jump[3:] = level[3:] - level[:-3] > 15                 # a low thud barely moves the flux, but jumps in level
    cand = [i for i in range(1, n - 1) if (flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1])
            or (jump[i] and not jump[i - 1])]
    keep: list[int] = []
    for i in cand:
        pre = float(np.median(level[max(0, i - 30):max(1, i - 5)]))
        peak = float(level[i:i + 4].max())
        post = level[i + 30:i + 60]
        if peak < -50 or peak - pre < 10 or (len(post) and float(np.median(post)) > peak - 10):
            continue
        # One sound, one onset: a second peak counts only after the level dipped in between.
        if keep and (i - keep[-1] < 8 or float(level[keep[-1] + 3:i].min()) > peak - 6):
            continue
        keep.append(i)
    times = (np.array(keep) * hop + win / 2) / SR
    timbre = np.array([logs[i:i + 8].mean(axis=0) for i in keep]) if keep else np.zeros((0, 24))
    return times, timbre


def hits(pcm: np.ndarray, tags: Tags, speech_spans: list[tuple[float, float]]) -> list[SoundEvent]:
    """Discrete short sounds outside speech, grouped by timbre and counted."""
    t, timbre = onsets(pcm)
    if not len(t):
        return []
    in_speech = np.array([any(s - 0.1 <= x <= e + 0.1 for s, e in speech_spans) for x in t])
    t, timbre = t[~in_speech], timbre[~in_speech]
    music = np.maximum(tags.family(MUSIC), tags.family(SINGING))
    # Beats inside real music are not events. With 3 s tagging windows a lone thud also reads as a
    # short "music" run, so only a long run (> 4.5 s) or a rhythmic one (4+ onsets) counts as music.
    runs = [(s, e) for s, e in _runs(music > 0.35, tags.times, tags.hop, 1.0, 1.5)
            if e - s > MUSIC_MIN_S or int(((t >= s) & (t <= e)).sum()) >= 4]
    beat = np.array([any(s <= x <= e for s, e in runs) for x in t], bool)
    t, timbre = t[~beat], timbre[~beat]
    # A syllable the 3 s windows did not mark as speech is not a hit: where the tagger hears some
    # speech nearby, ask it about 1 s at the onset itself (speech-bed bench: 40 of 49 stray hit
    # times gone, 3 of 37 right ones lost). Quiet recordings skip the check.
    near = tags.family(SPEECH)
    idx = np.clip(np.searchsorted(tags.times, t), 0, max(len(tags.times) - 1, 0))
    said = np.array([near[i] > 0.05 and speech_score(pcm, x) >= 0.3 for i, x in zip(idx, t)], bool) \
        if len(t) and len(near) else np.zeros(len(t), bool)
    t, timbre = t[~said], timbre[~said]
    if not len(t):
        return []
    z = timbre - timbre.mean(axis=1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
    groups: list[list[int]] = []
    centers: list[np.ndarray] = []
    for i in range(len(t)):
        sims = [float(z[i] @ c) for c in centers]
        j = int(np.argmax(sims)) if sims else -1
        if j >= 0 and sims[j] > 0.85:
            groups[j].append(i)
            centers[j] = z[groups[j]].mean(axis=0)
            centers[j] /= np.linalg.norm(centers[j]) + 1e-6
        else:
            groups.append([i])
            centers.append(z[i])
    out = []
    names = tags.names
    skip = SPEECH | MUSIC | SINGING | {SILENCE}
    for g in groups:
        times = sorted(float(t[i]) for i in g)
        # Name the group by what CED hears right at its onsets.
        idx = np.clip(np.searchsorted(tags.times, times), 0, len(tags.times) - 1)
        score = tags.probs[idx].astype(np.float32).mean(axis=0)
        score[sorted(skip)] = 0
        for k, nm in enumerate(names):
            if nm in VAGUE:
                score[k] *= 0.3
        best = int(np.argmax(score))
        # A lone onset the tagger cannot name, heard while music plays, is the music's own drum or
        # pluck (real-song bench: 16 stray lines, no right name among them). In silence it stays.
        if len(times) == 1 and score[best] < 0.1 and music[idx[0]] > 0.35:
            continue
        label = names[best] if score[best] > 0.05 and names[best] else "sharp sound"
        out.append(SoundEvent(times[0], times[-1], label, "hits", times, float(score[best]),
                              quality(pcm, times[0])))
    return sorted(out, key=lambda x: x.start)


def voice(tags: Tags, start: float, end: float) -> str | None:
    """'male' / 'female' / 'child' voice for a stretch of speech, when CED is clear about it."""
    m = (tags.times >= start - 0.25) & (tags.times <= end + 0.25)
    if not m.any():
        return None
    p = tags.probs[m][:, [MALE, FEMALE, CHILD]].astype(np.float32).mean(axis=0)
    if p.max() < 0.04:
        return None
    order = np.argsort(p)[::-1]
    if p[order[0]] < 1.5 * p[order[1]]:
        return None
    return ("male", "female", "child")[int(order[0])]


def speech_spans(tags: Tags) -> list[tuple[float, float]]:
    return _runs(tags.family(SPEECH) > 0.3, tags.times, tags.hop, 0.6, 0.5)


def bursts(rms_db: np.ndarray, times: np.ndarray, start: float, end: float, gap: float = 0.4) -> list[float]:
    """Start times of separate loud bursts inside a lasting sound (rings of a phone, blasts of a horn)."""
    m = (times >= start - 0.3) & (times <= end + 0.3)
    if m.sum() < 3:
        return []
    lv, tt = rms_db[m], times[m]
    loud = lv > np.percentile(lv, 90) - 12
    hop = float(np.median(np.diff(times)))
    runs = _runs(loud, tt, hop, gap, 0.2)
    return [round(s, 2) for s, _ in runs] if len(runs) > 1 else []


def voices(pcm: np.ndarray, spans: list[tuple[float, float]], thr: float = 0.6) -> list[str | None]:
    """Group speech lines by speaker via `siftasr.speakers` (TitaNet-small embeddings clustered greedily)."""
    try:
        from siftasr import speakers as _spk
    except ModuleNotFoundError:
        _sift_src = Path(__file__).resolve().parents[4] / "libs" / "siftasr" / "src"
        if _sift_src.is_dir():
            import sys
            sys.path.insert(0, str(_sift_src))
        from siftasr import speakers as _spk
    return _spk.voices(pcm, spans, thr=thr, max_workers=polite.threads(min(16, os.cpu_count() or 4)))

def quality(pcm: np.ndarray, t: float, dur: float = 0.3) -> str:
    """Words for how a short sound sounds, from its spectrum and decay: pitch register
    (spectral centroid), and whether it rings on or dies away."""
    a = int(max(0.0, t - 0.01) * SR)
    seg = pcm[a:a + int(max(dur, 0.6) * SR)]      # slice first: converting the whole file per call is ~400 MB/hour
    seg = seg.astype(np.float32) / 32768.0 if seg.dtype.kind == "i" else seg.astype(np.float32)
    if len(seg) < 512:
        return ""
    head = seg[:int(dur * SR)]
    spec = np.abs(np.fft.rfft(head * np.hanning(len(head)))) ** 2
    f = np.fft.rfftfreq(len(head), 1 / SR)
    cen = float((spec * f).sum() / (spec.sum() + 1e-12))
    pitch = "very low" if cen < 200 else "low" if cen < 500 else "mid" if cen < 1500 else "high" if cen < 4000 else "very high"
    env = np.array([np.sqrt((c ** 2).mean()) for c in np.array_split(seg, 12)]) + 1e-9
    decay = 20 * np.log10(env[-1] / env.max())
    tail = "short, dull" if decay < -30 and cen < 800 else "short, sharp" if decay < -30 else "ringing"
    return f"{pitch}-pitched (~{cen:.0f} Hz), {tail}"


def trim(ev: list[SoundEvent], track) -> list[SoundEvent]:
    """Shrink lasting sounds to the audible part of their span: long tagging windows smear a
    sound's edges into the silence around it."""
    for e in ev:
        if e.kind != "bed":
            continue
        m = (track.times >= e.start) & (track.times <= e.end) & ~track.silent
        if m.any():
            t = track.times[m]
            e.start, e.end = float(t[0] - HOP_TRACK / 2), float(t[-1] + HOP_TRACK / 2)
    return ev


HOP_TRACK = 0.05   # audio.HOP_S


# ------------------------------------------------------------------ harmony, rhythm & talk emotion

CED_LAUGH = {16, 17, 18, 19, 66}  # Laughter, Giggle, Snicker, Chuckle, Cheering
CED_TENSE = {6, 8, 14}            # Shout, Yell, Screaming (exclude Groan/Grunt: fire on close-mic chest voice)
CED_SUBDUED = {13, 20, 21, 22, 24}  # Whispering, Crying, Sigh, Gasp, Whimper

_WORDS_ANIMATED = {
    "haha", "hehe", "wow", "amazing", "awesome", "great", "fantastic", "incredible", "love", "excited",
    "yay", "fun", "fast", "tuyệt", "quá đã", "hay quá", "vui", "đỉnh", "thích",
}
_WORDS_HEATED = {
    "stop", "wrong", "never", "hate", "terrible", "awful", "worst", "angry", "furious", "ridiculous",
    "stupid", "broken", "fail", "wtf", "damn", "sai", "lỗi", "tệ", "bực", "không thể",
}
_WORDS_SUBDUED = {
    "maybe", "sorry", "sad", "unfortunately", "afraid", "unsure", "tired", "slow", "missed", "lost",
    "buồn", "tiếc", "chắc", "không biết", "mệt", "khó",
}


def music_harmony_rhythm(pcm: np.ndarray, ev: list[SoundEvent],
                         cut_times: list[float] = ()) -> list[SoundEvent]:
    """Populate `harmony`, `rhythm`, and `beat_sync` on music/singing beds >= 5 s."""
    from . import audio
    for e in ev:
        if e.kind != "bed" or not e.label.startswith(("music", "singing")):
            continue
        if e.end - e.start < 5.0:
            continue
        a = max(0, int(round(e.start * SR)))
        b = min(len(pcm), int(round(e.end * SR)))
        if b - a < 5 * SR:
            continue
        hr = audio.analyze_harmony_rhythm(pcm[a:b], SR)
        # Shift relative harmony_arc times to absolute timeline seconds
        arc_abs = [(round(e.start + s0, 2), round(e.start + s1, 2), m, k)
                   for s0, s1, m, k in hr.get("harmony_arc", [])]
        e.harmony = {
            "key": hr["key"],
            "mode": hr["mode"],
            "tension": hr["tension"],
            "harmony_arc": arc_abs if len(arc_abs) > 1 else [],
        }
        beats_abs = [round(e.start + bt, 3) for bt in hr.get("beats", [])]
        e.rhythm = {
            "bpm": hr["bpm"],
            "pulse_clarity": hr["pulse_clarity"],
        }
        bed_cuts = [c for c in cut_times if e.start <= c <= e.end]
        if beats_abs and len(bed_cuts) >= 3:
            sync = audio.cut_beat_sync(beats_abs, bed_cuts)
            if sync["synced"]:
                e.beat_sync = sync
    return ev


def _auto_talk_windows(speech: list[dict], p: dict[str, np.ndarray]) -> list[tuple[float, float, str]]:
    """Partition speech cues per speaker into coherent 6-20 s talk segments split on pauses >= 1.0 s
    or large local loudness/prosody shifts."""
    if not speech:
        return []
    out: list[tuple[float, float, str]] = []
    cur_v = (speech[0].get("voice") or "A").split(",")[0].strip()
    seg_s = float(speech[0]["start"])
    seg_e = float(speech[0]["end"])

    def _cue_rms(c: dict) -> float:
        m = (p["times"] >= c["start"]) & (p["times"] <= c["end"]) & p["voiced"]
        return float(np.median(p["rms_db"][m])) if m.any() else -35.0

    cur_rms = [_cue_rms(speech[0])]
    for c in speech[1:]:
        v = (c.get("voice") or "A").split(",")[0].strip()
        s, e = float(c["start"]), float(c["end"])
        gap = s - seg_e
        c_rms = _cue_rms(c)
        rms_jump = abs(c_rms - float(np.median(cur_rms))) >= 3.2 and (seg_e - seg_s) >= 5.0
        if v != cur_v or gap >= 1.15 or (seg_e - seg_s) >= 18.0 or rms_jump:
            if seg_e - seg_s >= 3.0:
                out.append((round(seg_s, 2), round(seg_e, 2), cur_v))
            cur_v = v
            seg_s, seg_e = s, e
            cur_rms = [c_rms]
        else:
            seg_e = max(seg_e, e)
            cur_rms.append(c_rms)
    if seg_e - seg_s >= 3.0:
        out.append((round(seg_s, 2), round(seg_e, 2), cur_v))
    return out


def talk_emotion_segments(pcm: np.ndarray, tags: Tags | None, speech: list[dict],
                          windows: list[tuple[float, float]] | None = None,
                          prosody_only: bool = False) -> list[dict]:
    """Detect within-speaker emotional state shifts across talk segments (`heated`, `animated`,
    `subdued`, or `calm` when `windows` is explicitly passed).

    Normalizes each speaker against their own baseline across the recording so naturally loud,
    quiet, or tonal (Vietnamese) speakers do not trigger false emotion labels.
    """
    from . import audio
    if not speech and not windows:
        return []
    p = audio.prosody_features(pcm, SR)
    times = p["times"]
    voiced = p["voiced"]
    rms_db = p["rms_db"]
    tilt_db = p["tilt_db"]
    log_f0 = np.log2(np.maximum(p["f0_hz"], 50.0))

    # Build per-speaker speech mask and baseline stats
    spk_cues: dict[str, list[dict]] = {}
    for c in speech:
        v = (c.get("voice") or "A").split(",")[0].strip() or "A"
        spk_cues.setdefault(v, []).append(c)
    if not spk_cues:
        spk_cues["A"] = []

    spk_base: dict[str, dict[str, float]] = {}
    for v, cues in spk_cues.items():
        cue_r, cue_t, cue_f = [], [], []
        m_spk = np.zeros(len(times), dtype=bool)
        for c in cues:
            mc = (times >= c["start"]) & (times <= c["end"]) & voiced
            m_spk |= mc
            if int(mc.sum()) >= 2:
                cue_r.append(float(np.median(rms_db[mc])))
                cue_t.append(float(np.median(tilt_db[mc])))
                cue_f.append(float(np.median(log_f0[mc])))
        mv = m_spk if m_spk.any() else voiced
        if not mv.any():
            spk_base[v] = {"rms": -25.0, "mad_rms": 2.0, "tilt": -8.0, "mad_tilt": 1.8,
                           "f0": np.log2(150.0), "mad_f0": 0.12, "pace": 3.0, "gap": 0.30}
            continue
        r_vals = rms_db[mv]
        t_vals = tilt_db[mv]
        f_vals = log_f0[mv]

        # Shortest-Half (Rousseeuw modal cluster) across utterance cues finds the speaker's
        # steady conversational baseline even when half the stream is emotional excursions
        if len(cue_r) >= 4:
            cr = np.asarray(cue_r, np.float32)
            ct = np.asarray(cue_t, np.float32)
            cf = np.asarray(cue_f, np.float32)
            order = np.argsort(cr)
            k_half = max(3, int(np.ceil(len(cr) * 0.45)))
            best_span = 1e9
            best_idx = order[:k_half]
            for i0 in range(len(cr) - k_half + 1):
                sub = order[i0:i0 + k_half]
                span = float(cr[sub[-1]] - cr[sub[0]]) + 0.35 * float(np.std(ct[sub]))
                if span < best_span:
                    best_span = span
                    best_idx = sub
            b_rms = float(np.median(cr[best_idx]))
            b_tilt = float(np.median(ct[best_idx]))
            b_f0 = float(np.median(cf[best_idx]))
        else:
            b_rms = float(np.median(r_vals))
            b_tilt = float(np.median(t_vals))
            b_f0 = float(np.median(f_vals))

        mad_r = max(1.8, float(1.4826 * np.median(np.abs(r_vals - b_rms))))
        mad_t = max(1.5, float(1.4826 * np.median(np.abs(t_vals - b_tilt))))
        mad_f = max(0.08, float(1.4826 * np.median(np.abs(f_vals - b_f0))))

        paces = []
        gaps = []
        for idx_c, c in enumerate(cues):
            dur_c = max(0.2, float(c["end"]) - float(c["start"]))
            n_w = len(c.get("words") or (c.get("text") or "").split())
            if n_w >= 2:
                paces.append(n_w / dur_c)
            if idx_c > 0:
                g = float(c["start"]) - float(cues[idx_c - 1]["end"])
                if 0.0 <= g <= 2.5:
                    gaps.append(g)
        spk_base[v] = {
            "rms": b_rms,
            "mad_rms": mad_r,
            "tilt": b_tilt,
            "mad_tilt": mad_t,
            "f0": b_f0,
            "mad_f0": mad_f,
            "pace": float(np.median(paces)) if paces else 3.0,
            "gap": float(np.median(gaps)) if gaps else 0.30,
        }
    if windows is not None:
        seg_list = [(float(s), float(e), "A") for s, e in windows]
    else:
        seg_list = _auto_talk_windows(speech, p)

    results: list[dict] = []
    for s, e, v_hint in seg_list:
        cues_in = [c for c in speech if float(c["end"]) > s + 0.1 and float(c["start"]) < e - 0.1]
        v = v_hint
        if cues_in:
            v = (cues_in[0].get("voice") or v_hint).split(",")[0].strip() or "A"
        base = spk_base.get(v) or next(iter(spk_base.values()))

        mv = (times >= s) & (times <= e) & voiced
        if int(mv.sum()) < 4:
            if windows is not None:
                results.append({"start": round(s, 2), "end": round(e, 2), "voice": v,
                                "emotion": "calm", "d_rms_db": 0.0, "pace_ratio": 1.0, "cues": []})
            continue

        seg_rms = float(np.median(rms_db[mv]))
        seg_rms_p85 = float(np.percentile(rms_db[mv], 85))
        seg_tilt = float(np.median(tilt_db[mv]))
        seg_f0 = float(np.median(log_f0[mv]))
        seg_f0_span = float(np.percentile(log_f0[mv], 90) - np.percentile(log_f0[mv], 10))

        d_rms = seg_rms - base["rms"]
        d_rms_peak = seg_rms_p85 - base["rms"]
        d_tilt = seg_tilt - base["tilt"]
        d_f0 = seg_f0 - base["f0"]

        # Speaking pace and pause gaps inside segment
        seg_words = sum(len(c.get("words") or (c.get("text") or "").split()) for c in cues_in)
        spk_dur = sum(max(0.15, min(e, float(c["end"])) - max(s, float(c["start"]))) for c in cues_in)
        seg_pace = (seg_words / spk_dur) if spk_dur > 0.3 and seg_words > 0 else base["pace"]
        pace_ratio = seg_pace / max(base["pace"], 0.5)

        seg_gaps = [float(cues_in[k]["start"]) - float(cues_in[k - 1]["end"])
                    for k in range(1, len(cues_in))
                    if 0.0 <= float(cues_in[k]["start"]) - float(cues_in[k - 1]["end"]) <= 3.0]
        mean_gap = float(np.mean(seg_gaps)) if seg_gaps else base["gap"]
        voiced_ratio = float(mv.sum()) / max(1, int(((times >= s) & (times <= e)).sum()))

        # Optional CED vocal bursts & spoken stance cues (disabled in Approach 2 prosody_only)
        ced_laugh = ced_tense = ced_sub = 0.0
        lex_anim = lex_heat = lex_sub = 0
        if not prosody_only:
            if tags is not None and len(tags.times):
                mt = (tags.times >= s) & (tags.times <= e)
                if mt.any():
                    pr = tags.probs[mt].astype(np.float32)
                    ced_laugh = float(pr[:, sorted(CED_LAUGH)].sum(axis=1).max())
                    ced_tense = float(pr[:, sorted(CED_TENSE)].sum(axis=1).max())
                    ced_sub = float(pr[:, sorted(CED_SUBDUED)].sum(axis=1).max())
            txt_all = " ".join((c.get("text") or "") for c in cues_in).lower()
            for w in _WORDS_ANIMATED:
                if w in txt_all:
                    lex_anim += 1
            for w in _WORDS_HEATED:
                if w in txt_all:
                    lex_heat += 1
            for w in _WORDS_SUBDUED:
                if w in txt_all:
                    lex_sub += 1
            if "..." in txt_all:
                lex_sub += 1

        # Score the 3 non-calm states relative to the speaker's modal calm baseline
        # 1. Subdued / hesitant: drop in vocal loudness (-2.8 dB or more) or long pauses + quiet/sigh
        score_subdued = (
            max(0.0, (-d_rms - 2.4) * 0.45)
            + max(0.0, (-d_tilt - 1.5) * 0.20)
            + (0.60 if not prosody_only and mean_gap >= 0.65 else 0.0)
            + (0.35 if not prosody_only and pace_ratio <= 0.85 else 0.0)
            + (0.75 if ced_sub >= 0.06 else 0.0)
            + (0.50 * min(lex_sub, 2))
        )
        if d_rms > -1.8 and ced_sub < 0.08 and mean_gap < 0.65:
            score_subdued = 0.0

        # 2. High arousal (heated vs animated): requires clear vocal elevation (> +2.8 dB or pitch/burst)
        arousal_lift = (
            max(0.0, d_rms - 2.2) * 0.30
            + max(0.0, d_tilt - 1.8) * 0.22
            + max(0.0, d_f0 - 0.10) * 2.2
        )
        if prosody_only:
            # Pure prosody separation: heated pushes louder RMS (> +8 dB) and harsher spectral tilt;
            # animated has moderate RMS lift (+2.2 to +8.5 dB) or higher pitch/f0 modulation
            score_heated = (
                arousal_lift
                + (0.85 if d_rms >= 8.5 or d_tilt >= 4.5 else -0.45)
            )
            score_animated = (
                arousal_lift
                + (0.75 if 2.0 <= d_rms < 8.5 or (d_f0 >= 0.12 and d_rms < 10.0) else 0.0)
                + max(0.0, seg_f0_span - 0.25) * 0.6
            )
            if d_rms < 2.2 and d_f0 < 0.12:
                score_heated = score_animated = 0.0
        else:
            score_heated = (
                arousal_lift
                + (0.70 if d_rms >= 8.5 or d_tilt >= 4.5 else 0.0)
                + (0.80 if ced_tense >= 0.08 else 0.0)
                + (0.65 * min(lex_heat, 2))
                - (0.75 if ced_laugh >= 0.08 or lex_anim > 0 else 0.0)
            )
            score_animated = (
                arousal_lift
                + (0.50 if 1.8 <= d_rms < 9.0 or d_f0 >= 0.10 else 0.0)
                + (0.35 if pace_ratio >= 1.12 else 0.0)
                + (0.90 if ced_laugh >= 0.08 else 0.0)
                + (0.65 * min(lex_anim, 2))
                - (0.65 if lex_heat > 0 and ced_laugh < 0.08 else 0.0)
            )
            # Convergence gate: when automatic segmentation runs (`windows is None`), require strong
            # acoustic shift (>= 5.0 dB) OR acoustic shift (>= 2.5 dB) + non-acoustic corroboration
            has_non_acoustic = (ced_laugh >= 0.08 or ced_tense >= 0.08 or ced_sub >= 0.08
                                or lex_anim > 0 or lex_heat > 0 or lex_sub > 0 or mean_gap >= 0.65)
            if windows is None and (abs(d_rms) < 2.5 or (not has_non_acoustic and abs(d_rms) < 5.0)):
                score_subdued = score_heated = score_animated = 0.0
        best_emo = "calm"
        best_sc = max(score_subdued, score_heated, score_animated)
        if best_sc >= 1.05:
            if score_subdued >= score_heated and score_subdued >= score_animated:
                best_emo = "subdued"
            elif score_heated >= score_animated:
                best_emo = "heated"
            else:
                best_emo = "animated"

        if best_emo == "calm" and windows is None:
            continue

        reasons: list[str] = []
        if abs(d_rms) >= 1.5:
            reasons.append(f"{d_rms:+.1f} dB")
        if abs(d_tilt) >= 1.8:
            reasons.append("sharp tone" if d_tilt > 0 else "soft tone")
        if pace_ratio >= 1.18 or pace_ratio <= 0.84:
            reasons.append(f"{pace_ratio:.1f}x pace")
        if mean_gap >= 0.65 and best_emo == "subdued":
            reasons.append("long pauses")
        if ced_laugh >= 0.05:
            reasons.append("laughter")
        elif ced_tense >= 0.05:
            reasons.append("strained voice")
        elif ced_sub >= 0.06:
            reasons.append("sigh/quiet")

        results.append({
            "start": round(s, 2),
            "end": round(e, 2),
            "voice": v,
            "emotion": best_emo,
            "score": round(float(best_sc), 3),
            "d_rms_db": round(d_rms, 1),
            "d_tilt_db": round(d_tilt, 1),
            "pace_ratio": round(pace_ratio, 2),
            "reasons": reasons,
        })
    return results
