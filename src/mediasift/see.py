"""`mediasift see VIDEO`: one command, no flags, cached. The video becomes a card.

The card is a timeline of what happened in picture and sound, written for a
reader who cannot watch or hear: events with where, how long and how many
times; sounds named and counted; speech with timing; on-screen text that
appeared. Images are files the agent opens only when it needs them:

  flow_NN.jpg   frames spread over how much the picture changed (not over the clock)
  ev_NN.jpg     before / after crops of single events, to read small detail
  look          any span on demand: frames plus a strip of the sound under them

Everything is computed once and cached by file content, so a second call is instant.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import __version__, audio, decode, events, polite, sound
from .transcript import Cue

CARD_VERSION = "see-15"   # bump when the card format changes: the cache key includes it
TILE_W, TILE_H = 320, 180          # the size uniform 1 fps sheets used in the bench: enough, ~920 tokens a sheet
FONT = "/System/Library/Fonts/Helvetica.ttc"
URL = re.compile(r"^https?://")


def log(msg: str) -> None:
    print(f"mediasift: {msg}", file=sys.stderr)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def ts(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m):02d}:{s:04.1f}"


# ------------------------------------------------------------------ cache

def content_key(path: Path, variant: str = "") -> str:
    """Size plus the first and last MiB: stable across renames and moves, cheap on huge files.
    `variant` separates cards made with different settings (`--asr off`)."""
    st = path.stat()
    h = hashlib.sha1(f"{st.st_size}:{CARD_VERSION}:{variant}".encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    return h.hexdigest()[:16]


def out_dir(path: Path, asr_mode: str = "auto") -> Path:
    return sound.cache_root() / "see" / content_key(path, "" if asr_mode == "auto" else f"asr={asr_mode}")


def fetch(url: str) -> tuple[Path, Path | None]:
    """A web video via yt-dlp into the cache, with its subtitles when the site has them."""
    vid = hashlib.sha1(url.encode()).hexdigest()[:12]
    d = sound.cache_root() / "url" / vid
    d.mkdir(parents=True, exist_ok=True)
    # yt-dlp writes stream parts (video.f398.mp4) before merging; url.txt is written only after success.
    final = lambda: [g for g in sorted(d.glob("video.*")) if re.fullmatch(r"video\.(mp4|mkv|webm|mov)", g.name)]
    if not ((d / "url.txt").exists() and final()):
        log(f"downloading {url}")
        # Original-language captions only: YouTube answers 429 for its auto-translations ('vi' on an
        # English video), and without --ignore-errors that one subtitle cost the whole video (yt-dlp
        # issue 13831; 5 of 7 real fetches). A translated track may still be tried and fail; it is ignored.
        runtimes = [r for r in ("deno", "node", "bun") if shutil.which(r)]
        if not runtimes:
            log("no JavaScript runtime (deno, node or bun): yt-dlp may miss YouTube formats; `mise use -g deno` fixes it")
        cmd = ["yt-dlp", "-q", "--no-playlist", "--ignore-errors", *[a for r in runtimes for a in ("--js-runtimes", r)],
               "-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
               "--write-subs", "--write-auto-subs", "--sub-langs", r".*-orig,en,vi", "--sub-format", "vtt",
               "--print-to-file", "%(language)s", str(d / "language.txt"),
               "-o", str(d / "video.%(ext)s"), url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and not final():
            raise RuntimeError(f"yt-dlp failed: {r.stderr.strip()[-1500:]}")
        if r.returncode != 0:                    # e.g. subtitles failed but the video merged fine
            log(f"yt-dlp warned (video kept): {r.stderr.strip()[-300:]}")
        (d / "url.txt").write_text(url + "\n")
    if not final():
        raise RuntimeError(f"yt-dlp finished but no merged video in {d}")
    return final()[0], _pick_subs(d)


def _pick_subs(d: Path) -> Path | None:
    """Captions in the video's own language, or none. Site subtitles replace speech recognition, so
    a wrong pick prints another language's words as the speech: YouTube's auto-dubs each bring an
    '-orig' track (an English talk came with de-DE-orig first). Unsure means no subtitles; ASR runs."""
    subs = sorted(d.glob("video*.vtt"))
    code = lambda s: s.name[len("video."):-len(".vtt")]
    lang = (d / "language.txt").read_text().strip() if (d / "language.txt").exists() else ""
    if lang and lang not in ("NA", "None"):
        base = lang.split("-")[0].lower()
        mine = [s for s in subs if code(s).split("-")[0].lower() == base]
        pick = ([s for s in mine if code(s).endswith("-orig")] or mine or [None])[0]
    else:
        orig = [s for s in subs if code(s).endswith("-orig")]
        pick = orig[0] if len(orig) == 1 else subs[0] if len(subs) == 1 else None
    if pick is None and subs:
        log(f"captions in {len(subs)} languages, none sure to be the video's ({lang or 'language unknown'}): "
            "transcribing the audio instead")
    return pick


def usage(event: str, **kw) -> None:
    """One line per call in <cache>/usage.jsonl: which videos were sensed, how long it took, and
    which spans agents zoomed into with `look` (a zoom marks where the card was not enough).
    Local only; the raw material for tuning thresholds and choosing what to improve next."""
    try:
        f = sound.cache_root() / "usage.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "v": CARD_VERSION, **kw},
                                default=str) + "\n")
    except OSError as e:          # logging must never break sensing, but say so
        log(f"usage log not written: {e}")


# ------------------------------------------------------------------ analysis

PASSES = ("pic", "ocr", "sounds", "harmony", "speech", "voices", "emotion")
MODES: dict[str, tuple[str, ...]] = {
    "full": PASSES,
    "auto": PASSES,
    "probe": ("pic", "sounds", "speech"),
    "podcast": ("sounds", "speech", "voices", "emotion"),
    "music": ("pic", "sounds", "harmony", "speech"),
    "reel": ("pic", "ocr", "sounds", "speech"),
    "screen": ("pic", "ocr", "sounds", "speech"),
    "adaptive": ("pic", "sounds", "speech"),
}


def _parse_pass_list(val: str | list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    if not val:
        return []
    items = [x.strip().lower() for x in (val.split(",") if isinstance(val, str) else val) if x and x.strip()]
    bad = [x for x in items if x not in PASSES]
    if bad:
        raise ValueError(f"unknown pass {', '.join(bad)!r}; choose from {', '.join(PASSES)}")
    return items


def resolve_passes_from_base(base: list[str] | tuple[str, ...],
                             with_passes: str | list[str] | tuple[str, ...] | set[str] | None = None,
                             without_passes: str | list[str] | tuple[str, ...] | set[str] | None = None) -> list[str]:
    active = set(base)
    add = set(_parse_pass_list(with_passes))
    rem = set(_parse_pass_list(without_passes))
    active |= add
    if "ocr" in add:
        active.add("pic")
    if "harmony" in add:
        active.add("sounds")
    if "voices" in add or "emotion" in add:
        active.update(("speech", "sounds"))
    active -= rem
    if "pic" in rem and "ocr" not in add:
        active.discard("ocr")
    if "speech" in rem:
        active.discard("voices")
        active.discard("emotion")
    if "sounds" in rem:
        active.discard("harmony")
    return [p for p in PASSES if p in active]


def resolve_passes(mode: str = "auto",
                   with_passes: str | list[str] | tuple[str, ...] | set[str] | None = None,
                   without_passes: str | list[str] | tuple[str, ...] | set[str] | None = None) -> list[str]:
    m = (mode or "auto").strip().lower()
    if m not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    return resolve_passes_from_base(MODES[m], with_passes, without_passes)


@dataclass
class Sensed:
    source: str
    duration: float
    width: int
    height: int
    fps: float
    analysed_fps: float
    content: str                                   # screen | camera | audio-only
    events: list[events.Event] = field(default_factory=list)
    sounds: list[sound.SoundEvent] = field(default_factory=list)
    speech: list[dict] = field(default_factory=list)   # {start, end, text, voice, words?}
    talk_emotions: list[dict] = field(default_factory=list)  # segment speaker emotion shifts
    asr: dict | None = None
    ocr: bool = False
    sound_sensed: bool = False
    holes: list[str] = field(default_factory=list)
    watermarks: list[str] = field(default_factory=list)   # platform marks OCR read, named once
    camera: list[dict] = field(default_factory=list)       # pans / tilts (scrolls on a screen), events.camera_moves
    shots: list = field(default_factory=list)              # (start, letter): the same letter = the same place again
    timings: dict = field(default_factory=dict)
    passes: list[str] = field(default_factory=lambda: list(PASSES))
    detected_kind: str = ""
    flows: list[str] = field(default_factory=list)
    refs: dict[int, str] = field(default_factory=dict)
    frame_peaks: list[list[float]] = field(default_factory=list)
    layout: dict = field(default_factory=dict)
    lasting_text: list[dict] = field(default_factory=list)


def classify_signal(s: Sensed, talk_s: float | None = None) -> tuple[str, list[str]]:
    """Infer the media profile ('screen', 'reel', 'podcast', 'music', 'film') and recommended passes
    from the lightweight probe signals (32x18 grid, CED sound beds, and speech coverage)."""
    dur = max(float(s.duration or 0.0), 1.0)
    music_s = sum(
        max(0.0, min(dur, x.end) - max(0.0, x.start))
        for x in s.sounds
        if x.kind == "bed" and x.label.lower().startswith(("music", "singing"))
    )
    speech_s = (
        float(talk_s)
        if talk_s is not None
        else sum(max(0.0, min(dur, c["end"]) - max(0.0, c["start"])) for c in s.speech)
    )
    cuts = sum(1 for e in s.events if e.kind == "cut")
    cut_rate = cuts / (dur / 60.0)
    if s.content == "audio-only" or s.width == 0:
        if music_s >= 0.30 * dur and speech_s < 0.25 * dur:
            return "music", list(MODES["music"][1:])  # sounds, harmony, speech
        return "podcast", list(MODES["podcast"])
    if s.content == "screen":
        return "screen", list(MODES["screen"])
    if s.height > s.width and dur <= 240.0:
        ps = list(MODES["reel"])
        if music_s >= 0.45 * dur and cuts >= 3 and "harmony" not in ps:
            ps.append("harmony")
        return "reel", [p for p in PASSES if p in ps]
    has_music = any(x.kind == "bed" and x.label.lower().startswith(("music", "singing")) for x in s.sounds)
    if (music_s >= 0.25 * dur and speech_s < 0.20 * dur) or (has_music and speech_s < 0.05 * dur):
        return "music", list(MODES["music"])
    if speech_s >= 0.40 * dur and music_s < 0.15 * dur and cut_rate < 8.0:
        return "podcast", ["pic", "sounds", "speech", "voices", "emotion"]
    return "film", list(PASSES)


def sensed_from_dict(js: dict) -> Sensed:
    evs = [
        events.Event(
            id=int(e["id"]),
            kind=e["kind"],
            start=float(e["start"]),
            end=float(e["end"]),
            before=float(e["before"]),
            after=float(e["after"]),
            peak=float(e["peak"]),
            region=events.Region(**e["region"]),
            path=[tuple(float(v) for v in p) for p in e.get("path", [])],
            shown_until=e.get("shown_until"),
            undoes=e.get("undoes"),
            text_added=list(e.get("text_added", [])),
            text_removed=list(e.get("text_removed", [])),
            sounds=list(e.get("sounds", [])),
        )
        for e in js.get("events", [])
    ]
    snds = [
        sound.SoundEvent(
            start=float(x["start"]),
            end=float(x["end"]),
            label=x["label"],
            kind=x["kind"],
            times=list(x.get("times", [])),
            score=float(x.get("score", 0.0)),
            quality=x.get("quality", ""),
            moods=[tuple(m) for m in x.get("moods", [])],
            harmony=dict(x.get("harmony", {})),
            rhythm=dict(x.get("rhythm", {})),
            beat_sync=dict(x.get("beat_sync", {})),
        )
        for x in js.get("sounds", [])
    ]
    return Sensed(
        source=js["source"],
        duration=float(js["duration"]),
        width=int(js["width"]),
        height=int(js["height"]),
        fps=float(js["fps"]),
        analysed_fps=float(js.get("analysed_fps", 0.0)),
        content=js.get("content", "audio-only"),
        events=evs,
        sounds=snds,
        speech=list(js.get("speech", [])),
        talk_emotions=list(js.get("talk_emotions", [])),
        asr=js.get("asr"),
        ocr=bool(js.get("ocr", False)),
        sound_sensed=bool(js.get("sound_sensed", False)),
        holes=list(js.get("holes", [])),
        watermarks=list(js.get("watermarks", [])),
        camera=list(js.get("camera", [])),
        shots=[tuple(sh) for sh in js.get("shots", [])],
        timings=dict(js.get("timings", {})),
        passes=list(js.get("passes") or PASSES),
        detected_kind=js.get("detected_kind", ""),
        flows=list(js.get("flows", [])),
        refs={int(k): v for k, v in (js.get("refs") or {}).items()},
        frame_peaks=[[float(x) for x in fp] for fp in js.get("frame_peaks", [])],
        layout=dict(js.get("layout") or {}),
        lasting_text=list(js.get("lasting_text") or []),
    )
def _visual(path: str, info: decode.MediaInfo) -> tuple[events.Grid, float]:
    # Brief events need a fast rate; long talks do not, and decoding is the cost there.
    fps = 15.0 if info.duration <= 600 else 5.0 if info.duration <= 2400 else 2.0
    # No cap at the file's average rate: a mostly still VFR screen recording averages under 1 fps
    # while its bursts run at 60. The fps filter repeats frames when the source has fewer.
    g = events.build_grid(decode.video_frames(path, fps, events.DECODE_W, events.DECODE_H))
    return g, fps


def _audio(path: str, info: decode.MediaInfo, subs: Path | None, asr_mode: str, early=None):
    pcm = early[0] if early else decode.audio_pcm(path, sound.SR)
    if len(pcm) < sound.SR // 2:
        return None
    at = audio.build_track(pcm)
    tags = early[1] if early else sound.tag(pcm, hop=0.5 if info.duration <= 1800 else 1.0)
    sp = sound.speech_spans(tags)
    hits = sound.hits(pcm, tags, sp)
    beds = sound.trim(sound.beds(tags, sp), at)
    # A short "music" run around a counted hit is that hit seen through a 3 s window, not music.
    hit_t = [t for h in hits for t in h.times]
    beds = [b for b in beds if not (b.label in ("music", "singing") and b.end - b.start <= sound.MUSIC_MIN_S
                                    and any(b.start - 0.5 <= t <= b.end + 0.5 for t in hit_t))]
    for bd in beds:                       # a phone that rings twice is one tag but two bursts
        if bd.label not in ("music", "singing"):
            bd.times = sound.bursts(at.rms_db, at.times, bd.start, bd.end)
    sil = sound.silences(at.rms_db, at.times, [t for h in hits for t in h.times])
    talk = sum(e - s for s, e in sp)
    # Speech is collected last (`finish`), so a transcription running on a GPU box overlaps
    # the picture and OCR passes instead of holding them up.
    return beds + hits + sil, lambda run_voices=True: _speech(pcm, tags, talk, subs, asr_mode, early, run_voices=run_voices), talk, tags, pcm

def _early_audio(path: str, info: decode.MediaInfo, asr_mode: str, host: str | None):
    """Audio decoded and tagged beside the picture pass; when the tags hear speech, transcription
    starts at once where it costs the Mac little memory: on the box, or with the Core ML backend
    (GPU, ~2 GB peak). MLX backends (3.3 GB) still wait for the local passes to finish.
    Returns (pcm, tags, future or None, detected language or None), or None for audio too short to tag."""
    from concurrent.futures import ThreadPoolExecutor
    from . import asr
    pcm = decode.audio_pcm(path, sound.SR)
    if len(pcm) < sound.SR // 2:
        return None
    tags = sound.tag(pcm, hop=0.5 if info.duration <= 1800 else 1.0)
    fut, lang = None, None
    if asr_mode == "on" or sum(e - s_ for s_, e in sound.speech_spans(tags)) >= 2.0:
        job = None
        if host:
            job = (asr.remote, pcm, host)
        else:
            lang = asr.detect_language(pcm)
            if asr.pick("auto", lang) == "quan-coreml":
                job = (asr.transcribe, pcm, "quan-coreml", None, lang)
        if job:
            pool = ThreadPoolExecutor(1)
            fut = pool.submit(*job)
            pool.shutdown(wait=False)
    return pcm, tags, fut, lang


def _speech(pcm, tags, talk: float, subs: Path | None, asr_mode: str, early, run_voices: bool = True) -> tuple[list[dict], dict | None]:
    cues, asr_info, pre_voices = [], None, None
    if subs is not None:
        from .transcript import load
        cues, asr_info = load(str(subs)), {"backend": "subtitles", "model": subs.name}
    elif asr_mode != "off" and (talk >= 2.0 or asr_mode == "on"):
        import os
        from . import asr
        host = os.environ.get("MEDIASIFT_ON")
        try:
            r = early[2].result() if early and early[2] else asr.remote(pcm, host) if host else None
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            if not host:          # local Core ML speech failed: running it again would fail the same way
                raise
            # A busy box (another job holding the GPU) costs speed, not the card: transcribe here.
            log(f"speech on {host} failed, transcribing here instead: {str(e).splitlines()[-1][:200]}")
            r = None
        try:
            if r is None:
                r = asr.transcribe(pcm, "auto", None, early[3] if early and len(early) > 3 else None)
            cues, asr_info = r.cues, {"backend": r.backend, "model": r.model, "language": r.language,
                                      "seconds": r.seconds}
            pre_voices = r.voices
        except (SystemExit, ImportError) as e:   # no backend installed: a partial card, said plainly
            log(f"speech not transcribed: {e}")
    speech = []
    if subs is not None or pre_voices is None:
        cues = _dedupe(cues, subtitles=subs is not None)
        who = sound.voices(pcm, [(c.start, c.end) for c in cues]) if run_voices else [None] * len(cues)
    else:
        who = pre_voices if run_voices else [None] * len(cues)
    genders = [sound.voice(tags, c.start, c.end) for c in cues]
    # One gender per voice letter (the majority of its lines): per-line tags flicker.
    per: dict[str, str | None] = {}
    for v in set(x for x in who if x):
        gs = [g for x, g in zip(who, genders) if x == v and g]
        top = max(set(gs), key=gs.count) if gs else None
        per[v] = top if top and gs.count(top) >= 0.6 * len(gs) else None
    for c, v, g_ in zip(cues, who, genders):
        g_ = per.get(v) if v else g_
        item = {"start": round(c.start, 2), "end": round(c.end, 2), "text": c.text,
                "voice": f"{v}" + (f", {g_}" if g_ else "") if v else g_}
        if getattr(c, "words", None):
            merged_w = _merge_subwords(c.words)
            if merged_w:
                item["words"] = merged_w
        speech.append(item)
    return speech, asr_info


def _dedupe(cues: list[Cue], subtitles: bool) -> list[Cue]:
    """Auto-subtitles repeat each line as it scrolls; keep each sentence once. Speech recognition
    output is left alone: a repeated line there was said twice."""
    if not subtitles:
        return [c for c in cues if c.text.strip()]
    out: list[Cue] = []
    for c in cues:
        t = " ".join(c.text.split())
        if not t:
            continue
        if out and (t == out[-1].text or out[-1].text.endswith(t)):
            out[-1].end = max(out[-1].end, c.end)
            continue
        if out and t.startswith(out[-1].text):
            t = t[len(out[-1].text):].strip()
            if not t:
                continue
        out.append(Cue(c.start, c.end, t))
    return out


def _ocr_events(path: str, info: decode.MediaInfo, evs: list[events.Event], work: Path, fps: float) -> bool:
    """Text that appeared or vanished in changed regions. Budget: about one event per 2 s of
    video (at least 30, at most 150), transient and pointed-at events first."""
    from . import ocr
    if not ocr.available() or not info.has_video:
        return False
    cand = [e for e in evs if e.kind in ("change", "cut", "blink") and e.region.area >= 0.002]
    budget = int(np.clip(info.duration / 2, 30, 150))
    cand.sort(key=lambda e: (-(e.shown_until is not None or e.kind == "blink" or bool(e.sounds)), -e.region.area))
    cand = sorted(cand[:budget], key=lambda e: e.start)
    times = sorted({_calm(e, w, fps) for e in cand for w in ("before", "after")})
    files = dict(zip(times, _frames_at(path, times, work, 1280)))
    cache: dict[tuple, list[str]] = {}

    def read(t: float, box):
        k = (t, box)
        if k not in cache:
            cache[k] = [x for x, _ in ocr.read(files[t], box)]
        return cache[k]

    for e in cand:
        x0, y0, x1, y1 = e.region.box()
        box = None if e.kind == "cut" else (round(max(x0 - 0.03, 0), 3), round(max(y0 - 0.05, 0), 3),
                                            round(min(x1 + 0.03, 1), 3), round(min(y1 + 0.05, 1), 3))
        e.text_added, e.text_removed = ocr.diff(read(_calm(e, "before", fps), box), read(_calm(e, "after", fps), box))
    return True


def _norm_ocr_key(s: str) -> str:
    import unicodedata
    s_low = s.lower().replace("đ", "d")
    s_nfd = "".join(c for c in unicodedata.normalize("NFD", s_low) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s_nfd).strip()


def scan_persistent_text(path: str, info: decode.MediaInfo, s: Sensed, work: Path) -> list[dict]:
    """Read static/persistent headlines, banners, and listicle overlays across settled anchor frames.
    Catches text that is already on screen at t=0 or never triggers a change event in _ocr_events."""
    from . import ocr
    if not ocr.available() or not info.has_video or info.duration <= 0:
        return []
    d = max(info.duration, 0.5)
    anchors = [round(min(0.5, d * 0.2), 3), round(d * 0.5, 3)]
    if d >= 6.0:
        anchors.append(round(min(d - 0.3, max(1.5, d * 0.82)), 3))
    for e in s.events:
        if e.kind == "cut" and len(anchors) < 6:
            anchors.append(round(min(d - 0.2, e.after + 0.35), 3))
    times: list[float] = []
    for t in sorted(set(anchors)):
        if not times or t - times[-1] >= 1.2:
            times.append(t)
    times = times[:6]
    files = dict(zip(times, _frames_at(path, times, work, 768)))
    ev_keys = {_norm_ocr_key(t) for e in s.events for t in e.text_added if len(_norm_ocr_key(t)) >= 4}
    speech_words = {w for c in s.speech for w in _norm_ocr_key(c.get("text", "")).split() if len(w) >= 3}
    tracks: list[dict] = []
    for t in times:
        fpath = files.get(t)
        if not fpath or not Path(fpath).exists():
            continue
        for raw_txt, box in ocr.read(str(fpath)):
            core = raw_txt.strip(" •·:-\"'")
            if len(core) < 3:
                continue
            if WATERMARK.search(core) or (s.content == "camera" and HANDLE.search(core)):
                wkey = re.sub(r"\W", "", core.lower())[:5]
                if not any(re.sub(r"\W", "", w.lower())[:5] == wkey for w in s.watermarks):
                    s.watermarks.append(core)
                continue
            k = _norm_ocr_key(core)
            if len(re.sub(r"\s+", "", k)) < 3:
                continue
            x0, y0, x1, y1 = box
            yc = 0.5 * (y0 + y1)
            zone = "top" if yc < 0.34 else ("bottom" if yc > 0.67 else "middle")
            toks = set(k.split())
            matched = False
            for tr in tracks:
                tr_toks = tr["toks"]
                overlap = len(toks & tr_toks) / max(1, min(len(toks), len(tr_toks)))
                if abs(yc - tr["yc"]) <= 0.12 and (k in tr["key"] or tr["key"] in k or overlap >= 0.6):
                    tr["end"] = t
                    tr["count"] += 1
                    if len(core) > len(tr["text"]):
                        tr["text"], tr["key"], tr["toks"] = core, k, toks
                    matched = True
                    break
            if not matched:
                tracks.append({
                    "text": core, "key": k, "toks": toks,
                    "zone": zone, "yc": yc, "start": t, "end": t, "count": 1,
                    "box": [round(float(v), 3) for v in box],
                })
    out: list[dict] = []
    for tr in tracks:
        k = tr["key"]
        if tr["count"] < 2 and any(k in ek or ek in k for ek in ev_keys):
            continue
        w = [x for x in tr["toks"] if len(x) >= 3]
        if w and speech_words and len(set(w) & speech_words) / len(w) >= 0.65:
            continue
        if tr["count"] >= 2 or (tr["start"] == times[0] and len(k) >= 5):
            out.append({
                "text": tr["text"],
                "zone": tr["zone"],
                "start": round(0.0 if tr["start"] == times[0] else tr["start"], 2),
                "end": round(info.duration if tr["end"] == times[-1] and tr["count"] >= 2 else tr["end"], 2),
                "box": tr["box"],
            })
    return out[:8]


def _sync_sound_events(s: Sensed) -> None:
    for snd in s.sounds:
        for t in (snd.times or [snd.start]):
            for e in s.events:
                if snd.label in ("silence", "music", "singing"):
                    continue
                if abs(e.start - t) <= 0.15:
                    if _snd_name(snd.label) not in e.sounds:
                        e.sounds.append(_snd_name(snd.label))
                elif e.kind == "motion" and e.start < t < e.end and snd.kind == "hits":
                    tag = f"{_snd_name(snd.label)} at {t:.1f}s, mid-motion"
                    if tag not in e.sounds:
                        e.sounds.append(tag)


def sense(path: Path, out: Path, asr_mode: str = "auto", subs: Path | None = None,
          mode: str = "auto", with_passes: str | None = None, without_passes: str | None = None) -> tuple[Sensed, events.Grid | None]:
    decode.require_ffmpeg()
    t0 = time.perf_counter()
    info = decode.probe(str(path))
    active = set(resolve_passes(mode, with_passes, without_passes))
    if asr_mode == "off":
        active -= {"speech", "voices", "emotion"}
    s = Sensed(str(path), info.duration, info.width, info.height, info.fps, 0.0,
               "camera" if info.has_video else "audio-only")
    g = None
    class _Now:                      # run in place (no pool): a picture failure stops before speech runs
        def __init__(self, fn, *a): self.fn, self.a = fn, a
        def result(self): return self.fn(*self.a)
    early, prep = None, None
    import os
    host = os.environ.get("MEDIASIFT_ON")
    want_audio = info.has_audio and bool(active & {"sounds", "harmony", "speech", "voices", "emotion"})
    want_speech = info.has_audio and info.has_video and ("pic" in active) and ("speech" in active) and subs is None and asr_mode != "off"
    if want_speech:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        prep = pool.submit(_early_audio, str(path), info, asr_mode, host)
        pool.shutdown(wait=False)
    finish = None
    talk = 0.0
    tags = None
    pcm_audio = None
    if info.has_video and "pic" in active:
        tv = time.perf_counter()
        g, s.analysed_fps = _visual(str(path), info)
        s.events = events.detect(g)
        s.content = "screen" if g.identical > 0.3 else "camera"
        s.camera = events.camera_moves(g, screen=s.content == "screen")
        s.shots = events.shot_letters(g, [e.start for e in s.events if e.kind == "cut"])
        s.frame_peaks = events.extract_frame_peaks(g)
        s.layout = events.detect_layout(g)
        s.timings["visual"] = round(time.perf_counter() - tv, 1)
    if prep is not None:
        early = prep.result()
        if early is None:
            s.holes.append("audio track too short to tag")
    if want_audio and (prep is None or early is not None):
        ta = time.perf_counter()
        r = _audio(str(path), info, subs, asr_mode if "speech" in active else "off", early)
        if r is not None:
            s.sounds, finish, talk, tags, pcm_audio = r
            s.sound_sensed = "sounds" in active
            s.timings["audio"] = round(time.perf_counter() - ta, 1)
        else:
            s.holes.append("audio track too short to tag")
    s.detected_kind, auto_passes = classify_signal(s, talk_s=talk)
    if (mode or "auto").strip().lower() == "adaptive":
        active = set(resolve_passes_from_base(auto_passes, with_passes, without_passes))
        if asr_mode == "off":
            active -= {"speech", "voices", "emotion"}
        s.sound_sensed = "sounds" in active and bool(s.sounds or pcm_audio is not None)
    if pcm_audio is not None and "harmony" in active:
        cut_t = [e.start for e in s.events if e.kind == "cut"]
        sound.music_harmony_rhythm(pcm_audio, s.sounds, cut_t)
    if info.has_video and "pic" in active:
        if s.analysed_fps < 10:
            s.holes.append(f"picture read at {s.analysed_fps:g} fps: changes shorter than {1 / s.analysed_fps:.1f} s can be missed")
        if "sounds" in active:
            _sync_sound_events(s)
        if "ocr" in active:
            t1 = time.perf_counter()
            s.ocr = _ocr_events(str(path), info, s.events, out / "frames", s.analysed_fps)
            if s.ocr:
                s.lasting_text = scan_persistent_text(str(path), info, s, out / "frames")
                s.timings["ocr"] = round(time.perf_counter() - t1, 1)
            else:
                s.holes.append("on-screen text not read (install indra-ocr on Linux or pyobjc Vision on macOS)")
    if finish is not None and "speech" in active:
        t2 = time.perf_counter()
        s.speech, s.asr = finish(run_voices="voices" in active)
        s.timings["speech_wait"] = round(time.perf_counter() - t2, 1)
    if not info.has_audio:
        s.holes.append("no audio track")
    elif "speech" in active and not s.speech and s.asr is None and any(x.label == "speech" for x in s.sounds):
        s.holes.append("speech heard but not transcribed")
    if s.speech:
        if "ocr" in active:
            _drop_captions(s)
        if pcm_audio is not None and "emotion" in active:
            s.talk_emotions = sound.talk_emotion_segments(pcm_audio, tags, s.speech)
    s.passes = [p for p in PASSES if p in active]
    s.timings["total"] = round(time.perf_counter() - t0, 1)
    return s, g


def _drop_captions(s: Sensed, window: float = 6.0) -> None:
    """Burned-in subtitles are the speech again, read by OCR: drop text whose words were said nearby."""
    import re as _re
    words = lambda x: set(_re.findall(r"\w{3,}", x.lower()))
    for e in s.events:
        if not (e.text_added or e.text_removed):
            continue
        said = set()
        for c in s.speech:
            if c["end"] >= e.start - window and c["start"] <= e.start + window:
                said |= words(c["text"])
        keep = lambda t: not (w := words(t)) or len(w & said) / len(w) < 0.6
        e.text_added = [t for t in e.text_added if keep(t)]
        e.text_removed = [t for t in e.text_removed if keep(t)]
    if s.lasting_text:
        all_said = {w for c in s.speech for w in _norm_ocr_key(c.get("text", "")).split() if len(w) >= 3}
        s.lasting_text = [
            item for item in s.lasting_text
            if not (ws := [w for w in _norm_ocr_key(item.get("text", "")).split() if len(w) >= 3])
            or len(set(ws) & all_said) / len(ws) < 0.65
        ]


# ------------------------------------------------------------------ words

ALIAS = {
    "Heart sounds, heartbeat": "thump (heartbeat-like)", "Heart murmur": "low rumble", "Sine wave": "tone",
    "Speech synthesizer": "speech", "Tick": "click", "Tick-tock": "ticking", "Beep, bleep": "beep",
    "Dial tone": "telephone tone", "Whack, thwack": "whack", "sharp sound": "sharp sound",
    "Change ringing (campanology)": "church bells ringing",
}


def _snd_name(label: str) -> str:
    if label in ("music", "singing", "silence"):
        return label
    from .feedback import learned       # names agents corrected the same way, again and again
    taught = learned()
    parts = [ALIAS.get(p.strip(), p.strip()) for p in label.split("/")]
    out = []
    for p in parts:
        if p.lower() not in [o.lower() for o in out]:
            out.append(p)
    name = " / ".join(out[:2]).lower()
    first = label.split("/")[0].strip()
    return f"{name} (usually: {taught[first]})" if first in taught else name


def _times(ts_: list[float], limit: int = 12) -> str:
    s = ", ".join(f"{t:.1f}" for t in ts_[:limit])
    return s + (f", ... (+{len(ts_) - limit})" if len(ts_) > limit else "")


def _event_line(grp: list[events.Event], img: dict[int, str]) -> tuple[float, str]:
    e = grp[0]
    span = f"{ts(e.start)}" + (f"-{ts(grp[-1].end)}" if grp[-1].end - e.start >= 0.3 else "")
    where = e.region.words()
    if len(grp) > 1:
        what = {"change": "changed", "blink": "flashed (changed and changed back)"}[e.kind]
        line = f"{where} {what} {len(grp)} times, at {_times([x.start for x in grp])} s"
        if e.region.area >= 0.002 and e.kind == "change":
            line += " (pixel count: a change may be one event or several)"
    elif e.kind == "cut":
        line = "CUT: new shot"
    elif e.kind == "scene":
        line = f"the whole picture kept moving for {e.end - e.start:.1f} s (camera move or busy footage)"
    elif e.kind == "motion":
        lg = events.legs(e.path)
        if len(lg) <= 3:
            path = "; then ".join(events.leg_words(x) for x in lg) or "moved"
        else:   # jittery: say where it went overall, not every turn
            t0, x0, y0 = e.path[0]
            t1, x1, y1 = e.path[-1]
            net = (events.leg_words((t0, x0, y0, t1, x1, y1)) if np.hypot(x1 - x0, y1 - y0) > 0.1
                   else f"stayed around the {events._where(x0, y0)}")
            path = f"moved around ({len(lg)} turns); overall {net}"
        line = f"{where.replace(' area', ' thing')} moved: {path}"
    elif e.kind == "blink":
        line = f"{where} flashed for {max(e.end - e.start, 1 / 15):.1f} s and changed back"
    elif e.kind == "animating":
        line = f"{where} kept changing in place for {e.end - e.start:.1f} s (animation, spinner or video)"
    else:
        line = f"{where} changed"
        if e.shown_until is not None:
            line += f"; it lasted {e.shown_until - e.start:.1f} s (until {ts(e.shown_until)})"
        if e.undoes is not None:
            line += f"; this undid #{e.undoes}"
    added = [t for x in grp for t in x.text_added]
    removed = [t for x in grp for t in x.text_removed]
    if added:
        line += " | text appeared: " + "; ".join(f'"{t}"' for t in added[-4:])
    if removed and not added:
        line += " | text vanished: " + "; ".join(f'"{t}"' for t in removed[:3])
    snd = sorted({s for x in grp for s in x.sounds})
    if snd and len(grp) > 1:
        with_s = [i + 1 for i, x in enumerate(grp) if x.sounds]
        if len(with_s) < len(grp):
            line += f" | together with a {', '.join(snd)}: #{', #'.join(map(str, with_s))} of the {len(grp)} (at {_times([grp[i - 1].start for i in with_s])} s)"
        else:
            line += f" | each with a {', '.join(snd)}"
    elif snd:
        line += f" | with sound: {', '.join(snd)}"
    ref = next((img[x.id] for x in grp if x.id in img), None)
    if ref:
        line += f" [{ref}]"
    ids = f"#{e.id}" if len(grp) == 1 else f"#{e.id}..#{grp[-1].id}"
    return e.start, f"{span:<15} pic {ids:<9} {line}"


def _sound_line(s_: sound.SoundEvent) -> tuple[float, str]:
    name = _snd_name(s_.label)
    if s_.kind == "bed" and 0 < s_.score < 0.3 and s_.label not in ("music", "singing"):
        name = f"maybe {name}"     # under a real song, names below 0.3 were wrong more often than right
    if s_.kind == "hits":
        n = len(s_.times)
        span = ts(s_.times[0]) + (f"-{ts(s_.times[-1])}" if n > 1 else "")
        body = f"[{name}]" if n == 1 else f"[{name} x{n}] at {_times(s_.times)} s"
        if s_.quality:
            body += f" ({s_.quality})"
    elif s_.kind == "silence":
        span, body = f"{ts(s_.start)}-{ts(s_.end)}", f"[silence {s_.end - s_.start:.1f} s]"
    else:
        span, body = f"{ts(s_.start)}-{ts(s_.end)}", f"[{name}]"
        if len(s_.times) > 1:
            body = f"[{name}, {len(s_.times)} separate bursts] starting at {_times(s_.times)} s"
        # Music: its mood, genre, harmony, tempo, beat-sync, and where a score shifts.
        arc = ", then ".join(f"{w} {ts(a)}-{ts(b)}" for a, b, w in s_.moods)
        harm = ""
        if s_.harmony:
            h_arc = s_.harmony.get("harmony_arc") or []
            if len(h_arc) >= 2:
                harm = f"{h_arc[0][2]} -> {h_arc[1][2]} at {ts(h_arc[1][0])}"
            elif s_.harmony.get("mode"):
                harm = s_.harmony["mode"]
        rhy = f"~{s_.rhythm['bpm']:.0f} BPM" if s_.rhythm and s_.rhythm.get("bpm") else ""
        bsync = (f"cuts sync {s_.beat_sync['hits']}/{s_.beat_sync['total']}"
                 if s_.beat_sync and s_.beat_sync.get("synced") else "")
        meta = "; ".join(x for x in (s_.quality, arc, harm, rhy, bsync) if x)
        if meta:
            body += f" ({meta})"
    return s_.start, f"{span:<15} snd {body}"


def _merge_beds(sounds: list[sound.SoundEvent], gap: float = 6.0) -> list[sound.SoundEvent]:
    silent = [(x.start, x.end) for x in sounds if x.kind == "silence"]
    """Music that drops out under speech for a few seconds is one bed, 'on and off'.
    A 'lasting' tag over counted short sounds of the same kind is the same thing said twice: dropped."""
    hits = [x for x in sounds if x.kind == "hits"]
    sounds = [x for x in sounds if not (x.kind == "bed" and any(
        _snd_name(h.label) == _snd_name(x.label) and any(x.start - 0.6 <= t <= x.end + 0.6 for t in h.times) for h in hits))]
    out: list[sound.SoundEvent] = []
    for s_ in sorted(sounds, key=lambda x: x.start):
        prev = next((o for o in reversed(out) if o.kind == "bed" and o.label == s_.label), None)
        quiet = sum(max(0.0, min(e, s_.start) - max(b, prev.end)) for b, e in silent) if prev else 0.0
        if s_.kind == "bed" and prev is not None and s_.start - prev.end <= gap and quiet < 1.0:
            prev.end = max(prev.end, s_.end)
            prev.times.append(-1.0)   # marks "on and off"
            # Each stretch's music mood survives the merge, in order, said once.
            said = [q for q in prev.quality.split(", then ") if q]
            if s_.quality and s_.quality not in said:
                prev.quality = ", then ".join([*said, s_.quality])
            prev.moods += s_.moods
            if not prev.harmony and s_.harmony:
                prev.harmony = dict(s_.harmony)
            if not prev.rhythm and s_.rhythm:
                prev.rhythm = dict(s_.rhythm)
            if not prev.beat_sync and s_.beat_sync:
                prev.beat_sync = dict(s_.beat_sync)
            continue
        out.append(sound.SoundEvent(s_.start, s_.end, s_.label, s_.kind, list(s_.times), s_.score, s_.quality,
                                    list(s_.moods), dict(s_.harmony), dict(s_.rhythm), dict(s_.beat_sync)))
    for o in out:
        if o.kind == "bed" and -1.0 in o.times:
            o.label = f"{o.label}, on and off"
            o.times = []
    return out


# ------------------------------------------------------------------ images

def _label(im: Image.Image, text: str, size: int = 18) -> None:
    d = ImageDraw.Draw(im)
    f = font(size)
    b = d.textbbox((0, 0), text, font=f)
    d.rectangle((0, 0, b[2] + 10, b[3] + 8), fill="black")
    d.text((5, 3), text, fill="yellow", font=f)


def _fit(im: Image.Image, w: int, h: int) -> Image.Image:
    im = im.convert("RGB")
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
    canvas = Image.new("RGB", (w, h), (12, 12, 12))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def _calm(e: events.Event, which: str, fps: float) -> float:
    """Seek time for an event's before/after frame. A seek can land one source frame
    late, so step half an analysis frame away from the change."""
    half = 0.5 / (fps or 15)
    if which == "before":
        return max(0.0, e.before - half)
    return (e.peak if e.kind == "blink" else e.after) + (0 if e.kind == "blink" else half)


def _frames_at(path: str, times: list[float], work: Path, width: int = 768) -> list[Path]:
    """JPEGs of exact moments, cached by time and width; the missing ones come from one decode pass."""
    work.mkdir(parents=True, exist_ok=True)
    name = lambda t: work / f"t{round(t, 3):010.3f}_{width}.jpg"
    todo = [t for t in times if not name(t).exists()]
    if todo and (todo[-1] - todo[0]) > 600 and len(todo) < (todo[-1] - todo[0]) / 5:
        # Sparse moments over a long file: one fast seek each beats decoding the whole file.
        with ThreadPoolExecutor(polite.threads(4)) as ex:
            list(ex.map(lambda t: decode.extract_frame(path, t, str(name(t)), width), todo))
        return [name(t) for t in times]
    for t, rgb in decode.frames_at(path, todo, width):
        if rgb is None:                    # past the last decodable frame
            decode.extract_frame(path, max(0.0, t - 0.1), str(name(t)), width)
        else:
            Image.fromarray(rgb).save(name(t), quality=90)
    return [name(t) for t in times]


def flow_sheets(path: str, s: Sensed, g: events.Grid | None, out: Path, max_frames: int | None = None) -> list[str]:
    if g is None:
        return []
    # About one frame per second up to 4 minutes (uniform 1 fps sheets were the strongest
    # baseline on short real footage), then slow growth; still frames are dropped, so a static
    # screen recording stays small. Long talks lean on the card and `look`.
    d = max(s.duration, 1.0)
    n = int(d) if d <= 240 else int(240 + 60 * math.sqrt((d - 240) / 60))
    n = int(np.clip(n, 12, 480 if d <= 1200 else 240))   # long talks: the transcript and `look` carry the rest
    if max_frames is not None:
        n = min(n, max_frames)
    if s.content == "camera":
        # On camera footage pixel change is mostly people and camera moving, not information:
        # change-weighting starved a slide animation while a lecturer walked (bench round 4).
        # Clock-uniform frames plus the first frame of every shot.
        grid_t = events.select_bucket_peaks(g, 0.0, s.duration, n, anchor_ends=False)
        shots = [e.after for e in s.events if e.kind == "cut"]
        times = []
        for t in sorted(grid_t + shots):
            if not times or t - times[-1] > 0.25:
                times.append(round(t, 3))
    else:
        times = events.change_weighted_times(g, n, s.events)
    files = _frames_at(path, times, out / "frames")
    # Tiles take the video's shape at the same pixel area (same tokens a sheet): a 9:16 reel in a
    # 16:9 box filled a third of it. Portrait sheets are 6 x 2, landscape 4 x 3.
    aspect = s.width / s.height if s.width and s.height else TILE_W / TILE_H
    tw = int(round(math.sqrt(TILE_W * TILE_H * aspect)))
    th = int(round(TILE_W * TILE_H / tw))
    cols = 4 if aspect >= 1 else 6
    # Camera footage lists almost no pixel events, so a tile's #id would point at nothing.
    shown = [e for grp in listed(s)[0] for e in grp] if s.content == "camera" else s.events
    names = []
    for k in range(0, len(files), 12):
        chunk = list(zip(times[k:k + 12], files[k:k + 12]))
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (min(cols, len(chunk)) * tw, rows * th), "black")
        for i, (t, f) in enumerate(chunk):
            tile = _fit(Image.open(f), tw, th)
            live = [e.id for e in shown if e.start - 0.05 <= t <= max(e.end, e.after) + 0.05 and e.kind != "scene"]
            _label(tile, f"{t:.1f}s" + (f"  #{live[0]}" if live else ""))
            sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
        name = f"flow_{k // 12 + 1:02d}.jpg"
        sheet.save(out / name, quality=85)
        names.append(f"{name} ({ts(chunk[0][0])}-{ts(chunk[-1][0])})")
    return names


def event_sheets(path: str, s: Sensed, out: Path, limit: int = 24) -> dict[int, str]:
    """Before | after crops for localized events, 3 per image: the detail a flow tile is too small to show.
    Screen recordings: the most telling events, listed or not (unlisted crops let answerers count and time
    brief overlays: real bench r7 lost 2 questions without them). Camera footage: only events the card
    lists, one per group (its pixel events are faces and texture; a phone reel got 7 useless crops)."""
    rank = lambda e: (-(e.shown_until is not None or bool(e.text_added) or bool(e.sounds)), -e.region.area)
    ok = lambda e: e.kind in ("change", "blink") and 0.0005 < e.region.area < 0.5
    if s.content == "camera":
        pick = [min(g, key=rank) for g in ([e for e in grp if ok(e)] for grp in listed(s)[0]) if g]
    else:
        pick = [e for e in s.events if ok(e)]
    pick = sorted(sorted(pick, key=rank)[:limit], key=lambda e: e.start)
    if not pick:
        return {}
    fps = s.analysed_fps
    times = sorted({_calm(e, w, fps) for e in pick for w in ("before", "after")})
    files = dict(zip(times, _frames_at(path, times, out / "frames")))
    W, H = 512, 288
    refs: dict[int, str] = {}
    for k in range(0, len(pick), 3):
        chunk = pick[k:k + 3]
        sheet = Image.new("RGB", (3 * W, len(chunk) * H), "black")
        for r, e in enumerate(chunk):
            tb, ta = _calm(e, "before", fps), _calm(e, "after", fps)
            before = Image.open(files[tb]).convert("RGB")
            after = Image.open(files[ta]).convert("RGB")
            x0, y0, x1, y1 = e.region.box()
            # Context: the after frame with the region boxed.
            ctx = after.copy()
            d = ImageDraw.Draw(ctx)
            d.rectangle((x0 * ctx.width, y0 * ctx.height, x1 * ctx.width, y1 * ctx.height), outline=(255, 40, 40), width=4)
            # Crops: the region with some margin, kept at the tile's aspect so text stays upright and legible.
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half_w = max((x1 - x0) / 2 + 0.04, 0.12)
            half_h = max((y1 - y0) / 2 + 0.06, half_w * (H / W) * after.width / after.height)
            half_w = max(half_w, half_h * (W / H) * after.height / after.width)
            box = tuple(int(v) for v in ((cx - half_w) * after.width, (cy - half_h) * after.height,
                                         (cx + half_w) * after.width, (cy + half_h) * after.height))
            tiles = [(_fit(ctx, W, H), f"#{e.id} {e.kind} {e.start:.1f}s (where)"),
                     (_fit(before.crop(box), W, H), f"before {tb:.2f}s"),
                     (_fit(after.crop(box), W, H), f"{'during' if e.kind == 'blink' else 'after'} {ta:.2f}s")]
            for c, (tile, lab) in enumerate(tiles):
                _label(tile, lab)
                sheet.paste(tile, (c * W, r * H))
            refs[e.id] = f"ev_{k // 3 + 1:02d}.jpg row {r + 1}"
        sheet.save(out / f"ev_{k // 3 + 1:02d}.jpg", quality=85)
    return refs


# ------------------------------------------------------------------ card

WATERMARK = re.compile(r"tik\s*tok|capcut", re.I)
HANDLE = re.compile(r"^@\s*[\w.]{3,}")      # camera footage only: in a chat recording '@sam' is content


def strip_watermarks(s: Sensed) -> None:
    """Platform watermarks (TikTok, @handle) jump between corners, and OCR read each jump as new
    text: 2-4 timeline lines per real reel. Name them once and drop them from events (idempotent)."""
    for e in s.events:
        for attr in ("text_added", "text_removed"):
            keep = []
            for t in getattr(e, attr):
                core = t.strip(" •·:-\"'")
                if WATERMARK.search(core) or (s.content == "camera" and HANDLE.search(core)):
                    key = re.sub(r"\W", "", core.lower())[:5]      # OCR variants of one handle
                    if attr == "text_added" and not any(re.sub(r"\W", "", w.lower())[:5] == key for w in s.watermarks):
                        s.watermarks.append(core)
                else:
                    keep.append(t)
            setattr(e, attr, keep)


def listed(s: Sensed) -> tuple[list[list[events.Event]], int]:
    """The picture-event groups the card lists, in time order, and how many it leaves out."""
    pics = []
    for grp in events.group(s.events):
        e0 = grp[0]
        if e0.kind == "cut":
            continue
        if (e0.kind == "motion" and e0.region.area < 0.002 and not e0.sounds
                and not any(np.hypot(l[4] - l[1], l[5] - l[2]) > 0.1 for l in events.legs(e0.path))):
            continue   # pointer-sized jitter: visible in the flow frames, too noisy as text
        if (s.content == "camera" or s.duration > 1200) and not any(x.text_added for x in grp):
            # Camera footage: pixel events are texture and motion that the flow sheets show better
            # (bench: listing them cost ~2k tokens a card and won nothing). Cuts and text stay.
            continue
        pics.append(grp)
    # Budget: busy footage yields hundreds of events. Keep the telling ones, count the rest.
    limit = int(np.clip(25 + s.duration / 12, 25, 120))
    def weight(grp):
        e0 = grp[0]
        return (3 * (e0.shown_until is not None or e0.kind == "blink" or e0.undoes is not None)
                + 3 * any(x.text_added or x.text_removed for x in grp) + 2 * any(x.sounds for x in grp)
                + 2 * (e0.kind == "cut") + 2 * (len(grp) > 1) + 3 * min(e0.region.area, 1.0)
                + 1.5 * (e0.kind == "motion"))
    kept = {id(g) for g in sorted(pics, key=weight, reverse=True)[:limit]}
    return [g for g in pics if id(g) in kept], len(pics) - len(kept)


def card(s: Sensed, flows: list[str], refs: dict[int, str], src_label: str,
         active_passes: set[str] | list[str] | None = None, show_profile: bool = False) -> str:
    strip_watermarks(s)
    active = set(active_passes if active_passes is not None else PASSES)
    kind = {"screen": "screen recording", "camera": "camera/film footage", "audio-only": "audio only"}[s.content]
    talk = "no speech" if not (s.speech and "speech" in active) else f"{len(s.speech)} speech lines"
    lines = [f"# {src_label}  ({ts(s.duration)}, {kind}, {s.width}x{s.height}, {talk})", ""]
    sensed = [f"picture: every frame at {s.analysed_fps:g} fps" if ("pic" in active and s.width and s.analysed_fps) else None,
              "sound: tagged every 0.5-1 s (AudioSet classes) + short sounds counted" if ("sounds" in active and s.sound_sensed) else None,
              f"speech: {s.asr['backend']} ({s.asr.get('model', '')})" if ("speech" in active and s.asr) else None,
              "on-screen text: OCR on changed regions" if ("ocr" in active and s.ocr) else None]
    lines.append("Sensed: " + ("; ".join(x for x in sensed if x) or "none") + ".")
    skipped = [p for p in PASSES if p not in active]
    if show_profile or skipped:
        det = s.detected_kind or classify_signal(s)[0]
        skip_note = f"; skipped: {', '.join(skipped)} — add with `--with {','.join(skipped[:2])}`" if skipped else ""
        lines.append(f"Profile: detected: {det} (active: {', '.join(p for p in PASSES if p in active)}{skip_note}).")
    if s.holes:
        lines.append("Not sensed: " + "; ".join(s.holes) + ".")
    if "ocr" in active and s.watermarks:
        marks = "; ".join(f'"{w}"' for w in s.watermarks[:4])
        lines.append(f"Watermark (moves around the frame; not listed below): {marks}.")
    if "pic" in active and s.layout and s.layout.get("kind", "full") != "full":
        vp = s.layout.get("viewport") or [0.0, 0.0, 1.0, 1.0]
        lines.append(
            f"Layout: {s.layout['kind']} (active viewport {vp[0]:.0%}-{vp[2]:.0%} x {vp[1]:.0%}-{vp[3]:.0%}, "
            f"{s.layout.get('panels', 1)} panel{'s' if s.layout.get('panels', 1) > 1 else ''})."
        )
    if "ocr" in active and s.lasting_text:
        lt_parts = [f'{x["zone"]}: "{x["text"]}" ({ts(x["start"])}-{ts(x["end"])})' for x in s.lasting_text[:5]]
        lines.append("Lasting text (persistent banner/overlay): " + "; ".join(lt_parts) + ".")
    use_flows = flows if "pic" in active else []
    use_refs = refs if "pic" in active else {}
    lines += ["", "How to use this card:",
              "- The timeline below is complete for what the sensors detect; times are seconds from the start.",
              ("- To SEE the sequence, read the flow sheets: 12 frames each, about one per second plus the first frame "
               "of every shot, each tile labelled with its time. Open the sheet that covers the time you need:"
               if s.content == "camera" else
               "- To SEE the sequence, read the flow sheets: 12 frames each, about one per second where the picture "
               "changes and one per still stretch, each tile labelled with its time and the event (#id) on screen. "
               "Open the sheet that covers the time you need:") if use_flows else "- No picture.",
              *([f"  {', '.join(use_flows[i:i + 4])}" for i in range(0, len(use_flows), 4)] if use_flows else []),
              *([f"- ev_NN.jpg rows show single events as [where | before | after] crops, for small detail"
                 f"{'' if s.content == 'camera' else '; each is labelled with its #id and time, and some brief ones have no line below'}."]
                if use_refs else []),
              f"- To zoom on any span or spoken phrase: `mediasift look \"{s.source}\" START-END` or `--say \"words\"` (frames + a strip of the sound). "
              "Use it for a gesture, a face or small text at one moment: flow tiles are ~1 small frame a second.",
              "- Words like 'small red area at bottom' come from pixel statistics; open an image before relying on them.", ""]
    rows: list[tuple[float, str]] = []
    if "pic" in active:
        cut_times = [e.start for e in s.events if e.kind == "cut"]
        letter = {round(t, 2): lab for t, lab in s.shots}
        lab = lambda t: f" {letter[round(t, 2)]}" if round(t, 2) in letter else ""
        if len(cut_times) == 1:
            t0 = cut_times[0]
            again = s.shots and lab(t0).strip() == s.shots[0][1]
            what = f"same place again (shot {s.shots[0][1]}, as at the start)" if again else f"new shot{lab(t0)}"
            rows.append((t0, f"{ts(t0):<15} pic CUT: {what}"))
        elif cut_times:
            opens = f"; the video opens on shot {s.shots[0][1]}, the same letter again = the same place" if s.shots else ""
            rows.append((cut_times[0], f"{ts(cut_times[0])}-{ts(cut_times[-1]):<7} pic CUTS ({len(cut_times)} new shots{opens}) at "
                                       + ", ".join(f"{t:.1f}{lab(t)}" for t in cut_times[:60])
                                       + (f", ... (+{len(cut_times) - 60})" if len(cut_times) > 60 else "") + " s"))
        if "ocr" not in active:
            saved = [(e.text_added, e.text_removed) for e in s.events]
            for e in s.events:
                e.text_added, e.text_removed = [], []
        try:
            keep, dropped = listed(s)
            for grp in keep:
                rows.append(_event_line(grp, use_refs))
        finally:
            if "ocr" not in active:
                for e, (ta, tr) in zip(s.events, saved):
                    e.text_added, e.text_removed = ta, tr
        for m in s.camera[:40]:
            who = "page" if m["words"].startswith("scrolls") else "camera"
            pace = " slowly" if m["speed"] < 0.08 else " fast" if m["speed"] > 0.5 else ""
            rows.append((m["start"], f"{ts(m['start'])}-{ts(m['end']):<7} pic {who} {m['words']}{pace}"))
        if len(s.camera) > 40:
            lines.insert(-1, f"- {len(s.camera) - 40} more camera moves are not listed.")
        if dropped:
            lines.insert(-1, f"- {dropped} smaller picture changes are not listed; the flow sheets show them.")
    if "sounds" in active:
        talk_spans = [(c["start"], c["end"]) for c in s.speech]
        def under_speech(x):
            cover = sum(max(0.0, min(e, x.end) - max(b, x.start)) for b, e in talk_spans)
            return cover > 0.7 * max(x.end - x.start, 1e-6)
        snds = [x for x in _merge_beds(s.sounds) if not (x.kind == "bed" and not x.label.startswith(("music", "singing"))
                                                           and under_speech(x))]
        cap = int(np.clip(30 + s.duration / 60, 30, 150))
        if len(snds) > cap:
            rank = sorted(snds, key=lambda x: (x.kind == "silence", -(len(x.times) or 1) * (x.score or 0.3) * (x.end - x.start + 1)))
            keep_ids = {id(x) for x in rank[:cap]}
            lines.insert(-1, f"- {len(snds) - cap} minor sound lines (short silences, weak tags) are not listed.")
            snds = [x for x in snds if id(x) in keep_ids]
        for snd in snds:
            if "harmony" not in active and (snd.harmony or snd.rhythm or snd.beat_sync):
                snd_copy = sound.SoundEvent(snd.start, snd.end, snd.label, snd.kind, list(snd.times),
                                            snd.score, snd.quality, list(snd.moods))
                rows.append(_sound_line(snd_copy))
            else:
                rows.append(_sound_line(snd))
    voices = (sorted({c["voice"] for c in s.speech if _letter(c.get("voice"))})
              or sorted({c["voice"] for c in s.speech if c.get("voice")})) if "voices" in active else []
    tag = ("voices" in active) and (len(_letters(s.speech)) > 1)
    if "emotion" in active:
        for emo in s.talk_emotions:
            v_emo = f" {emo['voice']}" if tag and emo.get("voice") else ""
            why = f" ({', '.join(emo['reasons'])})" if emo.get("reasons") else ""
            rows.append((emo["start"], f"{ts(emo['start'])}-{ts(emo['end']):<7} talk{v_emo} [{emo['emotion']}{why}]"))
    if "speech" in active:
        if len(s.speech) <= 150:
            for c in s.speech:
                v = f"[voice {c['voice']}] " if tag and c.get("voice") else ""
                rows.append((c["start"], f"{ts(c['start'])}-{ts(c['end']):<7} say {v}\"{c['text']}\""))
        else:
            mins: dict[int, list] = {}
            for c in s.speech:
                mins.setdefault(int(c["start"] // 60), []).append(c)
            for m, cs in sorted(mins.items()):
                words = " ".join(c["text"] for c in cs).split()
                vs = _letters(cs) if tag else []
                v = f"[voices {', '.join(vs)}] " if tag and vs else ""
                rows.append((m * 60.0, f"{ts(m * 60)}-{ts(m * 60 + 60):<7} say {v}\"{' '.join(words[:20])}{' ...' if len(words) > 20 else ''}\""))
        if len(s.speech) > 150:
            lines.insert(-1, f"- Speech is summarised one line per minute below; every line with times is in transcript.txt "
                             f"({len(s.speech)} lines): grep it.")
    rows.sort(key=lambda r: r[0])
    lines.append("## Timeline")
    lines.append("```")
    lines += [r[1] for r in rows] or ["(nothing changed in picture or sound)"]
    lines.append("```")
    if "speech" in active and "voices" in active:
        if not tag and voices:
            lines += ["", f"One voice throughout ({voices[0]}); no change of speaker was found."]
        elif tag:
            lines += ["", f"Voices: {'; '.join(voices)}. Letters group lines by speaker (a speaker-embedding model; "
                          "right on every real-speech bench dialogue); they are not names. Male/female comes from the "
                          "sound tagger. Lines under 0.8 s carry no letter."]
    return "\n".join(lines) + "\n"

def _letter(voice: str | None) -> str | None:
    """The speaker letter of a voice tag ("B, male" -> "B"); None for a gender-only tag ("male")."""
    head = (voice or "").split(",")[0].strip()
    return head if len(head) == 1 and head.isupper() else None


def _letters(speech: list[dict]) -> list[str]:
    return sorted({lt for c in speech if (lt := _letter(c.get("voice")))})


def transcript_text(s: Sensed) -> str:
    """Every speech line with times, and the speaker letter when there is more than one speaker,
    so `grep` answers who said it."""
    many = len(_letters(s.speech)) > 1
    out = []
    for c in s.speech:
        lt = _letter(c.get("voice")) if many else None
        out.append(f"[{ts(c['start'])}-{ts(c['end'])}] {lt + ': ' if lt else ''}{c['text']}\n")
    return "".join(out)


def _upgrade_sensed(path: Path, out: Path, s: Sensed, missing: set[str], wanted: set[str],
                    subs: Path | None, asr_mode: str) -> Sensed:
    info = decode.probe(str(path))
    t0 = time.perf_counter()
    if "pic" in missing and info.has_video:
        tv = time.perf_counter()
        g, s.analysed_fps = _visual(str(path), info)
        s.events = events.detect(g)
        s.content = "screen" if g.identical > 0.3 else "camera"
        s.camera = events.camera_moves(g, screen=s.content == "screen")
        s.shots = events.shot_letters(g, [e.start for e in s.events if e.kind == "cut"])
        s.frame_peaks = events.extract_frame_peaks(g)
        s.layout = events.detect_layout(g)
        if s.sounds:
            _sync_sound_events(s)
        s.flows = flow_sheets(str(path), s, g, out)
        s.refs = event_sheets(str(path), s, out)
        s.timings["visual"] = round(time.perf_counter() - tv, 1)
    if "ocr" in missing and info.has_video:
        t1 = time.perf_counter()
        s.ocr = _ocr_events(str(path), info, s.events, out / "frames", s.analysed_fps or 15.0)
        if s.ocr:
            s.lasting_text = scan_persistent_text(str(path), info, s, out / "frames")
            s.timings["ocr"] = round(time.perf_counter() - t1, 1)
            s.holes = [h for h in s.holes if "OCR" not in h]
        if s.speech:
            _drop_captions(s)
        strip_watermarks(s)
        s.refs = event_sheets(str(path), s, out)
    audio_missing = missing & {"sounds", "harmony", "speech", "voices", "emotion"}
    if audio_missing and info.has_audio:
        pcm = decode.audio_pcm(str(path), sound.SR)
        tags = None
        if missing & {"sounds", "speech", "voices", "emotion"}:
            tags = sound.tag(pcm, hop=0.5 if info.duration <= 1800 else 1.0)
        if "sounds" in missing and tags is not None:
            r = _audio(str(path), info, subs, "off", (pcm, tags, None, None))
            if r is not None:
                s.sounds = r[0]
                s.sound_sensed = True
                if s.events:
                    _sync_sound_events(s)
        if "harmony" in missing and s.sounds:
            sound.music_harmony_rhythm(pcm, s.sounds, [e.start for e in s.events if e.kind == "cut"])
        if "speech" in missing and asr_mode != "off":
            talk = sum(e - b for b, e in sound.speech_spans(tags)) if tags is not None else 0.0
            s.speech, s.asr = _speech(pcm, tags, talk, subs, asr_mode, None, run_voices="voices" in wanted)
            if s.ocr:
                _drop_captions(s)
        elif "voices" in missing and s.speech and tags is not None:
            who = sound.voices(pcm, [(c["start"], c["end"]) for c in s.speech])
            genders = [sound.voice(tags, c["start"], c["end"]) for c in s.speech]
            per: dict[str, str | None] = {}
            for v in set(x for x in who if x):
                gs = [g_ for x, g_ in zip(who, genders) if x == v and g_]
                top = max(set(gs), key=gs.count) if gs else None
                per[v] = top if top and gs.count(top) >= 0.6 * len(gs) else None
            for c, v, g_ in zip(s.speech, who, genders):
                g_ = per.get(v) if v else g_
                c["voice"] = f"{v}" + (f", {g_}" if g_ else "") if v else g_
        if "emotion" in missing and s.speech and tags is not None:
            s.talk_emotions = sound.talk_emotion_segments(pcm, tags, s.speech)
    s.passes = [p for p in PASSES if p in (set(s.passes) | wanted)]
    s.detected_kind = s.detected_kind or classify_signal(s)[0]
    s.timings["upgrade"] = round(time.perf_counter() - t0, 2)
    return s


def see(src: str, force: bool = False, asr_mode: str = "auto",
        mode: str = "auto", with_passes: str | None = None, without_passes: str | None = None) -> Path:
    subs = None
    if URL.match(src):
        path, subs = fetch(src)
        label = src
    else:
        path = Path(src).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(src)
        label = path.name
    out = out_dir(path, asr_mode)
    m = (mode or "auto").strip().lower()
    explicit_profile = (m not in ("auto", "full")) or bool(with_passes) or bool(without_passes)
    if (out / "sensed.json").exists() and not force:
        js = json.loads((out / "sensed.json").read_text())
        s = sensed_from_dict(js)
        if not s.flows and (out / "flow_01.jpg").exists():
            s.flows = [p.name for p in sorted(out.glob("flow_*.jpg"))]
        cached_passes = set(s.passes)
        if m == "adaptive":
            core_missing = {"pic", "sounds", "speech"} - cached_passes
            if core_missing:
                s = _upgrade_sensed(path, out, s, core_missing, cached_passes | core_missing, subs, asr_mode)
                cached_passes = set(s.passes)
            s.detected_kind, auto_p = classify_signal(s)
            wanted = set(resolve_passes_from_base(auto_p, with_passes, without_passes))
        else:
            wanted = set(resolve_passes(m, with_passes, without_passes))
        if asr_mode == "off":
            wanted -= {"speech", "voices", "emotion"}
        missing = wanted - cached_passes
        if missing:
            polite.enter()
            log(f"upgrading {label} (+{','.join(p for p in PASSES if p in missing)}) -> {out}")
            s = _upgrade_sensed(path, out, s, missing, wanted, subs, asr_mode)
            (out / "sensed.json").write_text(json.dumps({"mediasift": __version__, "card": CARD_VERSION, **asdict(s)},
                                                        indent=1, ensure_ascii=False, default=str))
            if s.speech:
                (out / "transcript.txt").write_text(transcript_text(s))
        tmp = out / "card.md.tmp"
        tmp.write_text(card(s, s.flows, s.refs, label, active_passes=wanted, show_profile=explicit_profile))
        tmp.replace(out / "card.md")
        usage("see", key=out.name, cached=not bool(missing), mode=m, passes=sorted(wanted))
        return out
    if force and (out / "frames").exists():
        shutil.rmtree(out / "frames", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    polite.enter()
    log(f"sensing {label} -> {out}")
    s, g = sense(path, out, asr_mode, subs, mode=m, with_passes=with_passes, without_passes=without_passes)
    strip_watermarks(s)
    t = time.perf_counter()
    if "pic" in s.passes and g is not None:
        s.flows = flow_sheets(str(path), s, g, out, max_frames=12 if m == "probe" else None)
        if m != "probe":
            s.refs = event_sheets(str(path), s, out)
    s.timings["images"] = round(time.perf_counter() - t, 1)
    (out / "sensed.json").write_text(json.dumps({"mediasift": __version__, "card": CARD_VERSION, **asdict(s)},
                                                indent=1, ensure_ascii=False, default=str))
    (out / "source.txt").write_text(str(path) + "\n")
    if s.speech:
        (out / "transcript.txt").write_text(transcript_text(s))
    tmp = out / "card.md.tmp"
    tmp.write_text(card(s, s.flows, s.refs, label, active_passes=s.passes, show_profile=explicit_profile))
    tmp.replace(out / "card.md")
    usage("see", key=out.name, cached=False, polite=polite.active(), duration=round(s.duration, 1), content=s.content,
          mode=m, passes=s.passes, events=len(s.events), sounds=len(s.sounds), speech=len(s.speech), timings=s.timings)
    log(f"done in {s.timings['total'] + s.timings['images']:.1f}s ({', '.join(f'{k} {v}s' for k, v in s.timings.items())})")
    return out


# ------------------------------------------------------------------ look

def parse_span(span: str, duration: float) -> tuple[float, float]:
    def sec(x: str) -> float:
        parts = [float(p) for p in x.split(":")]
        v = 0.0
        for p in parts:
            v = v * 60 + p
        return v
    a, _, b = span.partition("-")
    s_, e_ = sec(a), sec(b) if b else sec(a) + 2.0
    s_, e_ = max(0.0, s_), min(duration, e_)
    if e_ <= s_:
        raise ValueError(f"empty span {span!r}")
    return s_, e_


def _word_entry(w) -> tuple[float, float, str] | None:
    if isinstance(w, (list, tuple)) and len(w) >= 3:
        return float(w[0]), float(w[1]), str(w[2])
    if isinstance(w, dict) and "start" in w and "end" in w:
        return float(w["start"]), float(w["end"]), str(w.get("text", w.get("word", "")))
    if hasattr(w, "start") and hasattr(w, "end"):
        return float(w.start), float(w.end), str(getattr(w, "text", getattr(w, "word", "")))
    return None


def _merge_subwords(words: list) -> list[list]:
    entries = [e for w in (words or []) if (e := _word_entry(w)) is not None]
    if not entries:
        return []
    sp_mode = any(raw.startswith(" ") for _, _, raw in entries[1:])
    merged: list[list] = []
    for ws, we, raw in entries:
        if not raw:
            continue
        is_cont = (
            (sp_mode and not raw[0].isspace())
            or (not raw[0].isalnum() and not raw[0].isspace())
            or (bool(merged) and merged[-1][2].endswith("'"))
        )
        if merged and is_cont:
            merged[-1][1] = round(max(merged[-1][1], we), 2)
            merged[-1][2] += raw
        elif raw.strip():
            merged.append([round(ws, 2), round(we, 2), raw.strip()])
    return merged


def _norm_tokens(text: str, strip_accents: bool = False) -> list[str]:
    import unicodedata
    s = text.lower()
    if strip_accents:
        s = s.replace("đ", "d")
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return re.findall(r"\w+", s)


def _match_say(speech: list[dict], phrase: str, duration: float) -> tuple[float, float, tuple[float, float]]:
    """Match `phrase` against `speech[i]['words']` (or cue text fallback).
    Returns `(span_start, span_end, (word_start, word_end))`."""
    import difflib
    q_toks = _norm_tokens(phrase)
    if not q_toks:
        raise ValueError(f"empty --say phrase {phrase!r}")
    q_ascii = _norm_tokens(phrase, strip_accents=True)

    flat_words: list[tuple[float, float, str, str, str]] = []
    for c in speech:
        for ws, we, wt in _merge_subwords(c.get("words") or []):
            toks = _norm_tokens(wt)
            atoks = _norm_tokens(wt, strip_accents=True)
            for tok, atok in zip(toks, atoks):
                flat_words.append((float(ws), float(we), wt, tok, atok))

    def _pad(ws: float, we: float) -> tuple[float, float, tuple[float, float]]:
        span_len = max(0.05, we - ws)
        need = max(0.0, 1.2 - span_len)
        a = max(0.0, ws - max(0.25, need * 0.45))
        b = min(duration, we + max(0.35, need * 0.55))
        if b <= a:
            b = min(duration, a + 1.2)
        return a, b, (ws, we)

    if flat_words:
        k = len(q_toks)
        q_str = " ".join(q_toks)
        qa_str = " ".join(q_ascii)
        for use_ascii in (False, True):
            target = qa_str if use_ascii else q_str
            for i in range(len(flat_words) - k + 1):
                cand = " ".join(flat_words[j][4 if use_ascii else 3] for j in range(i, i + k))
                if cand == target:
                    return _pad(flat_words[i][0], flat_words[i + k - 1][1])
        best_ratio = 0.0
        best_span: tuple[float, float] | None = None
        for win_k in sorted({max(1, k - 1), k, min(len(flat_words), k + 1)}):
            for i in range(len(flat_words) - win_k + 1):
                cand = " ".join(flat_words[j][3] for j in range(i, i + win_k))
                r = difflib.SequenceMatcher(None, q_str, cand).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_span = (flat_words[i][0], flat_words[i + win_k - 1][1])
        if best_span is not None and best_ratio >= 0.68:
            return _pad(best_span[0], best_span[1])

    q_str = " ".join(q_toks)
    qa_str = " ".join(q_ascii)
    best_cue_score = 0.0
    best_cue: dict | None = None
    for c in speech:
        c_str = " ".join(_norm_tokens(c.get("text", "")))
        ca_str = " ".join(_norm_tokens(c.get("text", ""), strip_accents=True))
        if q_str in c_str or qa_str in ca_str:
            log("no word timestamps on matched line; interpolating cue span")
            idx = ca_str.find(qa_str) if q_str not in c_str else c_str.find(q_str)
            ref = ca_str if q_str not in c_str else c_str
            tgt = qa_str if q_str not in c_str else q_str
            c_dur = max(0.2, float(c["end"]) - float(c["start"]))
            f0 = max(0.0, idx / max(1, len(ref)))
            f1 = min(1.0, (idx + len(tgt)) / max(1, len(ref)))
            ws = float(c["start"]) + f0 * c_dur
            we = float(c["start"]) + f1 * c_dur
            return _pad(ws, we)
        c_toks = c_str.split()
        for i in range(max(1, len(c_toks) - len(q_toks) + 1)):
            sub = " ".join(c_toks[i:i + len(q_toks)])
            sc = difflib.SequenceMatcher(None, q_str, sub).ratio()
            if sc > best_cue_score:
                best_cue_score = sc
                best_cue = c

    if best_cue is not None and best_cue_score >= 0.72:
        log("no exact word match; using closest speech cue")
        return _pad(float(best_cue["start"]), float(best_cue["end"]))

    scored = sorted(
        speech,
        key=lambda c: difflib.SequenceMatcher(None, q_str, " ".join(_norm_tokens(c.get("text", "")))).ratio(),
        reverse=True,
    )[:3]
    candidates = "; ".join(f"[{ts(c['start'])}-{ts(c['end'])}] \"{c['text']}\"" for c in scored) or "(no speech)"
    raise ValueError(f"phrase {phrase!r} not found in transcript; closest lines: {candidates}")


def _active_crop_row(files: list[Path], times: list[float], cols: int, tw: int, th: int) -> Image.Image | None:
    """Codec Canvas active-region zoom strip: finds the highest-residual localized sub-region
    across the look() frames (cancelling global camera pan and ignoring bottom caption overlays
    when upper-frame gesture/UI motion is present) and renders 4 chronological 2x crops."""
    if len(files) < 4 or cols < 4:
        return None
    imgs = [Image.open(f).convert("RGB") for f in files]
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in imgs])
    n_frames, h, w, _ = arr.shape
    if h < 64 or w < 64:
        return None
    yh, yw = h // 2, w // 2
    win = np.outer(np.hanning(yh), np.hanning(yw)).astype(np.float32)
    lumas = [
        (arr[i] @ np.array([0.299, 0.587, 0.114], np.float32)).reshape(yh, 2, yw, 2).mean(axis=(1, 3))
        for i in range(n_frames)
    ]
    step_diffs = []
    for i in range(n_frames - 1):
        dx, dy, pk = events._shift(lumas[i], lumas[i + 1], win)
        prev = arr[i]
        ix, iy = int(round(dx * 2)), int(round(dy * 2))
        if pk >= 0.20 and (abs(ix) >= 1 or abs(iy) >= 1) and abs(ix) < w // 3 and abs(iy) < h // 3:
            warped = prev.copy()
            sy0, sy1 = max(0, -iy), min(h, h - iy)
            dy0, dy1 = max(0, iy), min(h, h + iy)
            sx0, sx1 = max(0, -ix), min(w, w - ix)
            dx0, dx1 = max(0, ix), min(w, w + ix)
            warped[dy0:dy1, dx0:dx1] = prev[sy0:sy1, sx0:sx1]
            if dy0 > 0:
                warped[:dy0, :] = arr[i + 1][:dy0, :]
            if dy1 < h:
                warped[dy1:, :] = arr[i + 1][dy1:, :]
            if dx0 > 0:
                warped[:, :dx0] = arr[i + 1][:, :dx0]
            if dx1 < w:
                warped[:, dx1:] = arr[i + 1][:, dx1:]
            prev = warped
        d = np.abs(arr[i + 1] - prev).max(axis=2)
        my, mx = max(2, h // 32), max(2, w // 32)
        d[:my, :] = 0.0
        d[-my:, :] = 0.0
        d[:, :mx] = 0.0
        d[:, -mx:] = 0.0
        step_diffs.append(d)
    sdiff = np.stack(step_diffs)
    gh, gw = 16, 16
    bh, bw = h // gh, w // gw
    grid_steps = sdiff[:, :gh * bh, :gw * bw].reshape(len(step_diffs), gh, bh, gw, bw).mean(axis=(2, 4))
    act_grid = np.percentile(grid_steps, 85, axis=0)
    if h > w:  # Portrait clip: if upper 75% has clear motion, ignore burned-in caption band at bottom
        upper_max = float(act_grid[:12, :].max())
        if upper_max >= 14.0:
            act_grid[12:, :] = np.minimum(act_grid[12:, :], upper_max * 0.30)
    peak_val = float(act_grid.max())
    if peak_val < 12.0:
        return None
    mask = act_grid >= max(11.0, 0.38 * peak_val)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    gy0, gy1 = int(ys.min()), int(ys.max()) + 1
    gx0, gx1 = int(xs.min()), int(xs.max()) + 1
    if (gy1 - gy0) * (gx1 - gx0) > 0.62 * (gh * gw):
        return None
    cx = (gx0 + gx1) * 0.5 * bw
    cy = (gy0 + gy1) * 0.5 * bh
    raw_w = (gx1 - gx0) * bw * 1.45
    raw_h = (gy1 - gy0) * bh * 1.45
    aspect = tw / max(th, 1)
    crop_w = float(np.clip(max(raw_w, raw_h * aspect), 0.36 * w, 0.68 * w))
    crop_h = float(np.clip(crop_w / aspect, 0.36 * h, 0.68 * h))
    crop_w = min(float(w), crop_h * aspect)
    x0 = int(round(np.clip(cx - crop_w * 0.5, 0, w - crop_w)))
    y0 = int(round(np.clip(cy - crop_h * 0.5, 0, h - crop_h)))
    x1 = min(w, int(round(x0 + crop_w)))
    y1 = min(h, int(round(y0 + crop_h)))
    box_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames - 1):
        sc = float(np.percentile(sdiff[i, y0:y1, x0:x1], 90))
        box_scores[i] = max(box_scores[i], sc * 0.85)
        box_scores[i + 1] = max(box_scores[i + 1], sc)
    interior = list(range(1, n_frames - 1))
    interior.sort(key=lambda idx: float(box_scores[idx]), reverse=True)
    picked_mid = sorted(interior[: cols - 2])
    picked = [0] + picked_mid + [n_frames - 1]
    row_im = Image.new("RGB", (cols * tw, th), "black")
    for k, idx in enumerate(picked[:cols]):
        crop = imgs[idx].crop((x0, y0, x1, y1)).resize((tw, th), Image.Resampling.LANCZOS)
        _label(crop, f"zoom {times[idx]:.2f}s", 18)
        row_im.paste(crop, (k * tw, 0))
    return row_im


def look(src: str, span: str | None = None, n: int = 8, width: int = TILE_W, say: str | None = None) -> Path:
    """Frames spread over a span (snapped to intra-bucket motion peaks when sensed) or around
    `--say PHRASE`, with an active-region zoom strip and the sound drawn underneath."""
    path = Path(fetch(src)[0] if URL.match(src) else Path(src).expanduser().resolve())
    info = decode.probe(str(path))
    focus_words: tuple[float, float] | None = None
    side = out_dir(path) / "sensed.json"
    js: dict = {}
    if say is not None:
        if not side.exists():
            see(str(path))
        js = json.loads(side.read_text())
        a, b, focus_words = _match_say(js.get("speech", []), say, info.duration)
    elif span is not None:
        a, b = parse_span(span, info.duration)
        if side.exists():
            js = json.loads(side.read_text())
    else:
        raise ValueError("give a span (START-END) or --say PHRASE")
    out = out_dir(path) / "look"
    out.mkdir(parents=True, exist_ok=True)
    if not info.has_video:              # audio only: the sound strip is the whole picture
        sheet = Image.new("RGB", (1280, 160), "black")
        _sound_strip(sheet, path, a, b, [a, b], 0, 160, focus_words=focus_words)
        name = f"look_{a:09.2f}-{b:09.2f}.jpg"
        usage("look", key=out.parent.name, start=round(a, 2), end=round(b, 2), n=0, say=bool(say))
        sheet.save(out / name, quality=88)
        return out / name
    n = max(1, min(n, int((b - a) * min(info.fps or 30, 30)) or 1))
    frame_peaks = js.get("frame_peaks") or []
    if frame_peaks and n >= 4:
        times = events.snap_look_times(a, b, n, frame_peaks)
    else:
        times = [a + (b - a) * (i + 0.5) / n for i in range(n)]
    files = _frames_at(str(path), times, out / "frames", 768 if n <= 2 else 512)
    cols = 1 if n == 1 else 2 if n <= 4 else 4
    tw = 1024 if n == 1 else 640 if n <= 4 else width
    th = round(tw * (info.height or 9) / (info.width or 16))
    rows = (n + cols - 1) // cols
    crop_row = _active_crop_row(files, times, cols, tw, th) if (n >= 6 and cols == 4 and (b - a) <= 6.0) else None
    crop_h = crop_row.height if crop_row is not None else 0
    strip_h = 120 if info.has_audio else 0
    sheet = Image.new("RGB", (cols * tw, rows * th + crop_h + strip_h), "black")
    for i, (t, f) in enumerate(zip(times, files)):
        tile = _fit(Image.open(f), tw, th)
        _label(tile, f"{t:.2f}s", 20)
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    if crop_row is not None:
        sheet.paste(crop_row, (0, rows * th))
    if info.has_audio:
        _sound_strip(sheet, path, a, b, times, rows * th + crop_h, strip_h, focus_words=focus_words)
    name = f"look_{a:09.2f}-{b:09.2f}.jpg"
    usage("look", key=out.parent.name, start=round(a, 2), end=round(b, 2), n=n, say=bool(say))
    sheet.save(out / name, quality=88)
    return out / name


def _sound_strip(sheet: Image.Image, path: Path, a: float, b: float, times: list[float], y0: int, h: int,
                 focus_words: tuple[float, float] | None = None) -> None:
    """Loudness envelope, silence shaded, the tagger's labels and speech words, on the frames' time axis."""
    pcm = decode.audio_pcm(str(path), sound.SR)
    seg = pcm[int(a * sound.SR): int(b * sound.SR)].astype(np.float32) / 32768.0
    W = sheet.width
    d = ImageDraw.Draw(sheet)
    d.rectangle((0, y0, W, y0 + h), fill=(18, 18, 24))
    if len(seg):
        cols = np.array_split(seg, W)
        db = np.array([20 * np.log10(np.sqrt((c ** 2).mean()) + 1e-6) if len(c) else -90 for c in cols])
        lv = np.clip((db + 60) / 60, 0, 1)
        mid = y0 + h - 30
        for x, v in enumerate(lv):
            d.line((x, mid - v * (h - 62), x, mid), fill=(90, 160, 255) if v > 0.08 else (60, 60, 70))
    f = font(15)
    for t in times:
        x = (t - a) / (b - a) * W
        d.line((x, y0, x, y0 + 8), fill="yellow")
    side = out_dir(path) / "sensed.json"
    if side.exists():
        js = json.loads(side.read_text())
        for snd in js.get("sounds", []):
            pts = snd["times"] or [snd["start"]]
            for t in pts:
                if a <= t <= b and snd["kind"] == "hits":
                    x = (t - a) / (b - a) * W
                    d.line((x, y0 + 10, x, y0 + h - 30), fill=(255, 120, 60), width=2)
            if snd["kind"] != "hits" and snd["end"] >= a and snd["start"] <= b:
                x0 = max(0, (snd["start"] - a) / (b - a) * W)
                x1 = min(W, (snd["end"] - a) / (b - a) * W)
                d.rectangle((x0, y0 + h - 26, x1, y0 + h - 4), outline=(200, 200, 200))
                d.text((x0 + 4, y0 + h - 25), _snd_name(snd["label"]), fill="white", font=f)
        if focus_words is not None:
            fx0 = max(0, (focus_words[0] - a) / max(b - a, 1e-6) * W)
            fx1 = min(W, (focus_words[1] - a) / max(b - a, 1e-6) * W)
            d.rectangle((fx0, y0 + 8, fx1, y0 + h - 28), outline=(255, 220, 80), width=2)
        said = [c for c in js.get("speech", []) if c["end"] >= a and c["start"] <= b]
        raw_words: list[tuple[float, float, str]] = []
        for c in said:
            for ws, we, wt in _merge_subwords(c.get("words") or []):
                raw_words.append((float(ws), float(we), str(wt)))
        in_words = [(ws, we, wt) for ws, we, wt in raw_words if we >= a and ws <= b]
        if in_words and (focus_words is not None or b - a <= 5.0):
            step_w = max(1, (len(in_words) + 7) // 8)  # cap at 8 word labels across the strip
            picked = in_words[::step_w][:8]
            for k, (ws, we, wt) in enumerate(picked):
                x0 = min(W - 48, max(2, (ws - a) / max(b - a, 1e-6) * W))
                in_focus = focus_words is not None and ws >= focus_words[0] - 0.05 and we <= focus_words[1] + 0.05
                col = (255, 235, 110) if in_focus else (180, 255, 180)
                d.text((x0 + 2, y0 + 10 + 17 * (k % 2)), wt, fill=col, font=f)
        else:
            for k, c in enumerate(said):
                x0 = max(0, (c["start"] - a) / (b - a) * W)
                x1 = (said[k + 1]["start"] - a) / (b - a) * W if k + 1 < len(said) else W
                text = f"\"{c['text']}\""
                while len(text) > 4 and d.textlength(text, font=f) > max(x1 - x0 - 8, 40) * (2 if k % 2 else 1.6):
                    text = text[:-5] + "...\""          # fit before the next line starts (rows alternate)
                d.text((x0 + 4, y0 + 10 + 17 * (k % 2)), text, fill=(180, 255, 180), font=f)
    d.text((W - 295, y0 + h - 25), f"sound {a:.1f}-{b:.1f}s (orange = short sounds)", fill=(150, 150, 150), font=f)
