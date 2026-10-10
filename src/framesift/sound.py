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
    print(f"framesift: {msg}", file=sys.stderr)


def cache_root() -> Path:
    base = os.environ.get("FRAMESIFT_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "framesift")
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


def tag(pcm: np.ndarray, hop: float = 0.5, top_k: int = 12) -> Tags:
    tagger = _tagger(top_k)
    y = pcm.astype(np.float32) / 32768.0 if pcm.dtype.kind == "i" else pcm.astype(np.float32)
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
    quality: str = ""        # how it sounds: "low, dull" / "high, bright, ringing"


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
        ev.append(SoundEvent(s, e, label, "bed", score=float(music[m].mean())))
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
    """Group speech lines by speaker: a TitaNet-small embedding per line (NeMo, via sherpa-onnx),
    clustered greedily against running centroids; 'A', 'B', ... in order of first appearance.
    Real-speech bench (LibriSpeech dialogues, bench/sense/voices.py): 1.00 pairwise agreement and
    the right number of voices in 8/8 dialogues; a mean-spectrum heuristic scored 0.73, and the
    CAM++ exports could not tell speakers apart under this sherpa-onnx. Lines under 0.8 s get None."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import sherpa_onnx
    # Lines in parallel, one thread each: the same embeddings 3x faster than one 4-thread call at
    # a time (300 lines, GPU box: 11.0 s -> 3.2 s; a 1h45m podcast has ~1,900 lines). Capped when polite.
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(speaker_model()), num_threads=1))

    def embed(span) -> np.ndarray | None:
        seg = pcm[int(span[0] * SR):int(span[1] * SR)]
        if len(seg) < 0.8 * SR:
            return None
        st = ex.create_stream()
        st.accept_waveform(SR, seg.astype(np.float32) / 32768.0 if seg.dtype.kind == "i" else seg.astype(np.float32))
        st.input_finished()
        v = np.array(ex.compute(st), np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    with ThreadPoolExecutor(polite.threads(min(16, os.cpu_count() or 4))) as pool:
        embs: list[np.ndarray | None] = list(pool.map(embed, spans))
    if sum(v is not None for v in embs) < 2:
        return [None] * len(spans)
    sums: list[np.ndarray] = []
    out: list[str | None] = []
    for v in embs:
        if v is None:
            out.append(None)
            continue
        d = [1 - float(v @ (c / np.linalg.norm(c))) for c in sums]
        j = int(np.argmin(d)) if d else -1
        if j >= 0 and d[j] < thr:
            sums[j] = sums[j] + v
        else:
            sums.append(v.copy())
            j = len(sums) - 1
        out.append(j)
    counts = [out.count(k) for k in range(len(sums))]
    if sum(counts) >= 20:
        # A one-line "speaker" in a long talk is a noisy fragment ("So this"): fold it into the
        # nearest voice that holds at least three lines.
        major = [k for k, c in enumerate(counts) if c >= 3]
        if major:
            for i, (k, v) in enumerate(zip(out, embs)):
                if k is not None and counts[k] == 1:
                    out[i] = max(major, key=lambda m: float(v @ (sums[m] / np.linalg.norm(sums[m]))))
    order: dict[int, str] = {}
    for k in out:
        if k is not None and k not in order:
            order[k] = chr(ord("A") + min(len(order), 25))
    named = [order[k] if k is not None else None for k in out]
    return named if len(order) > 1 else [None] * len(spans)

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
