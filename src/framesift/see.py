"""`framesift see VIDEO`: one command, no flags, cached. The video becomes a card.

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

CARD_VERSION = "see-6"   # bump when the card format changes: the cache key includes it
TILE_W, TILE_H = 320, 180          # the size uniform 1 fps sheets used in the bench: enough, ~920 tokens a sheet
FONT = "/System/Library/Fonts/Helvetica.ttc"
URL = re.compile(r"^https?://")


def log(msg: str) -> None:
    print(f"framesift: {msg}", file=sys.stderr)


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
        cmd = ["yt-dlp", "-q", "--no-playlist", "-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
               "--write-subs", "--write-auto-subs", "--sub-langs", "en.*,vi.*,orig", "--sub-format", "vtt",
               "-o", str(d / "video.%(ext)s"), url]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and not final():
            raise RuntimeError(f"yt-dlp failed: {r.stderr.strip()[-1500:]}")
        if r.returncode != 0:                    # e.g. subtitles failed but the video merged fine
            log(f"yt-dlp warned (video kept): {r.stderr.strip()[-300:]}")
        (d / "url.txt").write_text(url + "\n")
    if not final():
        raise RuntimeError(f"yt-dlp finished but no merged video in {d}")
    video = final()[0]
    subs = sorted(d.glob("video*.vtt"))
    manual = [s for s in subs if "orig" in s.name or not re.search(r"\.\w\w(-\w+)?\.vtt$", s.name)]
    return video, (manual or subs or [None])[0]


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
    speech: list[dict] = field(default_factory=list)   # {start, end, text, voice}
    asr: dict | None = None
    ocr: bool = False
    sound_sensed: bool = False
    holes: list[str] = field(default_factory=list)
    watermarks: list[str] = field(default_factory=list)   # platform marks OCR read, named once
    timings: dict = field(default_factory=dict)


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
    return beds + hits + sil, lambda: _speech(pcm, tags, talk, subs, asr_mode, early), talk, tags


def _speech(pcm, tags, talk: float, subs: Path | None, asr_mode: str, early) -> tuple[list[dict], dict | None]:
    cues, asr_info, pre_voices = [], None, None
    if subs is not None:
        from .transcript import load
        cues, asr_info = load(str(subs)), {"backend": "subtitles", "model": subs.name}
    elif asr_mode != "off" and (talk >= 2.0 or asr_mode == "on"):
        import os
        from . import asr
        host = os.environ.get("FRAMESIFT_ON")
        try:
            r = early[2].result() if early and early[2] else asr.remote(pcm, host) if host else None
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            # A busy box (another job holding the GPU) costs speed, not the card: transcribe here.
            log(f"speech on {host} failed, transcribing here instead: {str(e).splitlines()[-1][:200]}")
            r = None
        try:
            if r is None:
                r = asr.transcribe(pcm, "auto", None, None)
            cues, asr_info = r.cues, {"backend": r.backend, "model": r.model, "language": r.language,
                                      "seconds": r.seconds}
            pre_voices = r.voices
        except (SystemExit, ImportError) as e:   # no backend installed: a partial card, said plainly
            log(f"speech not transcribed: {e}")
    speech = []
    if subs is not None or pre_voices is None:
        cues = _dedupe(cues, subtitles=subs is not None)
        who = sound.voices(pcm, [(c.start, c.end) for c in cues])
    else:
        who = pre_voices                       # computed on the transcribing host
    genders = [sound.voice(tags, c.start, c.end) for c in cues]
    # One gender per voice letter (the majority of its lines): per-line tags flicker.
    per: dict[str, str | None] = {}
    for v in set(x for x in who if x):
        gs = [g for x, g in zip(who, genders) if x == v and g]
        top = max(set(gs), key=gs.count) if gs else None
        per[v] = top if top and gs.count(top) >= 0.6 * len(gs) else None
    for c, v, g_ in zip(cues, who, genders):
        g_ = per.get(v) if v else g_
        speech.append({"start": round(c.start, 2), "end": round(c.end, 2), "text": c.text,
                       "voice": f"{v}" + (f", {g_}" if g_ else "") if v else g_})
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


def sense(path: Path, out: Path, asr_mode: str = "auto", subs: Path | None = None) -> tuple[Sensed, events.Grid | None]:
    decode.require_ffmpeg()
    t0 = time.perf_counter()
    info = decode.probe(str(path))
    s = Sensed(str(path), info.duration, info.width, info.height, info.fps, 0.0, "audio-only")
    g = None
    # One pass after the other: overlapping them stacked the decoder's and the speech
    # model's memory and pushed a 16 GB Mac into swap; the time saved was a few seconds.
    class _Now:                      # run in place (no pool): a picture failure stops before speech runs
        def __init__(self, fn, *a): self.fn, self.a = fn, a
        def result(self): return self.fn(*self.a)
    # Speech on a GPU box is the exception: the Mac only waits on ssh (11.5 s of a 29 s card on a
    # real 2:44 reel), so that wait runs beside the local passes. No local CPU or memory is added.
    early = None
    import os
    host = os.environ.get("FRAMESIFT_ON")
    if host and info.has_audio and subs is None and asr_mode != "off":
        from concurrent.futures import ThreadPoolExecutor
        from . import asr
        pcm = decode.audio_pcm(str(path), sound.SR)
        if len(pcm) >= sound.SR // 2:
            # Tag first: only audio with speech goes to the box (the same test `_audio` applies).
            tags = sound.tag(pcm, hop=0.5 if info.duration <= 1800 else 1.0)
            fut = None
            if asr_mode == "on" or sum(e - s_ for s_, e in sound.speech_spans(tags)) >= 2.0:
                pool = ThreadPoolExecutor(1)
                fut = pool.submit(asr.remote, pcm, host)
                pool.shutdown(wait=False)
            early = (pcm, tags, fut)
    finish = None
    if True:
        fv = _Now(_visual, str(path), info) if info.has_video else None
        fa = _Now(_audio, str(path), info, subs, asr_mode, early) if info.has_audio else None
        if fv is not None:
            g, s.analysed_fps = fv.result()
            s.events = events.detect(g)
            s.content = "screen" if g.identical > 0.3 else "camera"
            s.timings["visual"] = round(time.perf_counter() - t0, 1)
        if fa is not None:
            r = fa.result()
            if r is not None:
                s.sounds, finish, talk, tags = r
                s.sound_sensed = True
                s.timings["audio"] = round(time.perf_counter() - t0, 1)
            else:
                s.holes.append("audio track too short to tag")
    if info.has_video:
        if s.analysed_fps < 10:
            s.holes.append(f"picture read at {s.analysed_fps:g} fps: changes shorter than {1 / s.analysed_fps:.1f} s can be missed")
        # Sound and picture together: a click that lands with a change is probably its cause.
        for snd in s.sounds:
            for t in (snd.times or [snd.start]):
                for e in s.events:
                    if snd.label in ("silence", "music", "singing"):
                        continue
                    if abs(e.start - t) <= 0.15:
                        e.sounds.append(_snd_name(snd.label))
                    elif e.kind == "motion" and e.start < t < e.end and snd.kind == "hits":
                        e.sounds.append(f"{_snd_name(snd.label)} at {t:.1f}s, mid-motion")
        t1 = time.perf_counter()
        s.ocr = _ocr_events(str(path), info, s.events, out / "frames", s.analysed_fps)
        if s.ocr:
            s.timings["ocr"] = round(time.perf_counter() - t1, 1)
        else:
            s.holes.append("on-screen text not read (OCR needs macOS Vision)")
    if finish is not None:
        t2 = time.perf_counter()
        s.speech, s.asr = finish()
        s.timings["speech_wait"] = round(time.perf_counter() - t2, 1)
    if not info.has_audio:
        s.holes.append("no audio track")
    elif not s.speech and s.asr is None and any(x.label == "speech" for x in s.sounds):
        s.holes.append("speech heard but not transcribed")
    if s.speech:
        _drop_captions(s)
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


# ------------------------------------------------------------------ words

ALIAS = {
    "Heart sounds, heartbeat": "thump (heartbeat-like)", "Heart murmur": "low rumble", "Sine wave": "tone",
    "Speech synthesizer": "speech", "Tick": "click", "Tick-tock": "ticking", "Beep, bleep": "beep",
    "Dial tone": "telephone tone", "Whack, thwack": "whack", "sharp sound": "sharp sound",
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
            continue
        out.append(sound.SoundEvent(s_.start, s_.end, s_.label, s_.kind, list(s_.times), s_.score, s_.quality))
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


def flow_sheets(path: str, s: Sensed, g: events.Grid | None, out: Path) -> list[str]:
    if g is None:
        return []
    # About one frame per second up to 4 minutes (uniform 1 fps sheets were the strongest
    # baseline on short real footage), then slow growth; still frames are dropped, so a static
    # screen recording stays small. Long talks lean on the card and `look`.
    d = max(s.duration, 1.0)
    n = int(d) if d <= 240 else int(240 + 60 * math.sqrt((d - 240) / 60))
    n = int(np.clip(n, 12, 480 if d <= 1200 else 240))   # long talks: the transcript and `look` carry the rest
    if s.content == "camera":
        # On camera footage pixel change is mostly people and camera moving, not information:
        # change-weighting starved a slide animation while a lecturer walked (bench round 4).
        # Clock-uniform frames plus the first frame of every shot.
        grid_t = [float(t) for t in (np.arange(n) + 0.5) * s.duration / n]
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


def card(s: Sensed, flows: list[str], refs: dict[int, str], src_label: str) -> str:
    strip_watermarks(s)
    kind = {"screen": "screen recording", "camera": "camera/film footage", "audio-only": "audio only"}[s.content]
    talk ="no speech" if not s.speech else f"{len(s.speech)} speech lines"
    lines = [f"# {src_label}  ({ts(s.duration)}, {kind}, {s.width}x{s.height}, {talk})", ""]
    sensed = [f"picture: every frame at {s.analysed_fps:g} fps" if s.width else None,
              "sound: tagged every 0.5-1 s (AudioSet classes) + short sounds counted" if s.sound_sensed else None,
              f"speech: {s.asr['backend']} ({s.asr.get('model', '')})" if s.asr else None,
              "on-screen text: OCR on changed regions" if s.ocr else None]
    lines.append("Sensed: " + "; ".join(x for x in sensed if x) + ".")
    if s.holes:
        lines.append("Not sensed: " + "; ".join(s.holes) + ".")
    if s.watermarks:
        marks = "; ".join(f'"{w}"' for w in s.watermarks[:4])
        lines.append(f"Watermark (moves around the frame; not listed below): {marks}.")
    lines += ["", "How to use this card:",
              "- The timeline below is complete for what the sensors detect; times are seconds from the start.",
              ("- To SEE the sequence, read the flow sheets: 12 frames each, about one per second plus the first frame "
               "of every shot, each tile labelled with its time. Open the sheet that covers the time you need:"
               if s.content == "camera" else
               "- To SEE the sequence, read the flow sheets: 12 frames each, about one per second where the picture "
               "changes and one per still stretch, each tile labelled with its time and the event (#id) on screen. "
               "Open the sheet that covers the time you need:") if flows else "- No picture.",
              *([f"  {', '.join(flows[i:i + 4])}" for i in range(0, len(flows), 4)] if flows else []),
              *([f"- ev_NN.jpg rows show single events as [where | before | after] crops, for small detail"
                 f"{'' if s.content == 'camera' else '; each is labelled with its #id and time, and some brief ones have no line below'}."]
                if refs else []),
              f"- To zoom on any span: `framesift look \"{s.source}\" START-END` (frames + a strip of the sound).",
              "- Words like 'small red area at bottom' come from pixel statistics; open an image before relying on them.", ""]
    rows: list[tuple[float, str]] = []
    cut_times = [e.start for e in s.events if e.kind == "cut"]
    if len(cut_times) == 1:
        rows.append((cut_times[0], f"{ts(cut_times[0]):<15} pic CUT: new shot"))
    elif cut_times:
        rows.append((cut_times[0], f"{ts(cut_times[0])}-{ts(cut_times[-1]):<7} pic CUTS ({len(cut_times)} new shots) at "
                                   f"{_times(cut_times, 60)} s"))
    keep, dropped = listed(s)
    for grp in keep:
        rows.append(_event_line(grp, refs))
    if dropped:
        lines.insert(-1, f"- {dropped} smaller picture changes are not listed; the flow sheets show them.")
    talk = [(c["start"], c["end"]) for c in s.speech]
    def under_speech(x):
        cover = sum(max(0.0, min(e, x.end) - max(b, x.start)) for b, e in talk)
        return cover > 0.7 * max(x.end - x.start, 1e-6)
    snds = [x for x in _merge_beds(s.sounds) if not (x.kind == "bed" and not x.label.startswith(("music", "singing"))
                                                       and under_speech(x))]
    # A voice through a podcast mic reads to the tagger as 'radio' or 'groan'; under speech that is the voice.
    cap = int(np.clip(30 + s.duration / 60, 30, 150))
    if len(snds) > cap:
        rank = sorted(snds, key=lambda x: (x.kind == "silence", -(len(x.times) or 1) * (x.score or 0.3) * (x.end - x.start + 1)))
        keep_ids = {id(x) for x in rank[:cap]}
        lines.insert(-1, f"- {len(snds) - cap} minor sound lines (short silences, weak tags) are not listed.")
        snds = [x for x in snds if id(x) in keep_ids]
    for snd in snds:
        rows.append(_sound_line(snd))
    voices = sorted({c["voice"] for c in s.speech if c.get("voice")})
    tag = len(voices) > 1          # one voice throughout: say it once below, not on every line
    if len(s.speech) <= 150:
        for c in s.speech:
            v = f"[voice {c['voice']}] " if tag and c.get("voice") else ""
            rows.append((c["start"], f"{ts(c['start'])}-{ts(c['end']):<7} say {v}\"{c['text']}\""))
    else:
        # A long talk: one line per minute here, every word in transcript.txt (grep it).
        mins: dict[int, list] = {}
        for c in s.speech:
            mins.setdefault(int(c["start"] // 60), []).append(c)
        for m, cs in sorted(mins.items()):
            words = " ".join(c["text"] for c in cs).split()
            vs = sorted({c["voice"] for c in cs if c.get("voice")})
            v = f"[voices {', '.join(vs)}] " if tag and vs else ""
            rows.append((m * 60.0, f"{ts(m * 60)}-{ts(m * 60 + 60):<7} say {v}\"{' '.join(words[:20])}{' ...' if len(words) > 20 else ''}\""))
    rows.sort(key=lambda r: r[0])
    if len(s.speech) > 150:
        lines.insert(-1, f"- Speech is summarised one line per minute below; every line with times is in transcript.txt "
                         f"({len(s.speech)} lines): grep it.")
    lines.append("## Timeline")
    lines.append("```")
    lines += [r[1] for r in rows] or ["(nothing changed in picture or sound)"]
    lines.append("```")
    if len(voices) == 1:
        lines += ["", f"One voice throughout ({voices[0]}); no change of speaker was found."]
    elif voices:
        lines += ["", f"Voices: {'; '.join(voices)}. Letters group lines by speaker (a speaker-embedding model; "
                      "right on every real-speech bench dialogue); they are not names. Male/female comes from the "
                      "sound tagger. Lines under 0.8 s carry no letter."]
    return "\n".join(lines) + "\n"


def see(src: str, force: bool = False, asr_mode: str = "auto") -> Path:
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
    if (out / "card.md").exists() and not force:
        usage("see", key=out.name, cached=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    polite.enter()
    log(f"sensing {label} -> {out}")
    s, g = sense(path, out, asr_mode, subs)
    strip_watermarks(s)
    t = time.perf_counter()
    flows = flow_sheets(str(path), s, g, out)
    refs = event_sheets(str(path), s, out)
    s.timings["images"] = round(time.perf_counter() - t, 1)
    (out / "sensed.json").write_text(json.dumps({"framesift": __version__, "card": CARD_VERSION, **asdict(s)},
                                                indent=1, ensure_ascii=False, default=str))
    (out / "source.txt").write_text(str(path) + "\n")
    if s.speech:
        (out / "transcript.txt").write_text("".join(f"[{ts(c['start'])}-{ts(c['end'])}] {c['text']}\n" for c in s.speech))
    # The card marks the cache entry complete, so it is written last and atomically.
    tmp = out / "card.md.tmp"
    tmp.write_text(card(s, flows, refs, label))
    tmp.replace(out / "card.md")
    usage("see", key=out.name, cached=False, polite=polite.active(), duration=round(s.duration, 1), content=s.content,
          events=len(s.events), sounds=len(s.sounds), speech=len(s.speech), timings=s.timings)
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


def look(src: str, span: str, n: int = 8, width: int = TILE_W) -> Path:
    """Frames evenly spread over a span, with the sound drawn underneath on the same time axis."""
    path = Path(fetch(src)[0] if URL.match(src) else Path(src).expanduser().resolve())
    info = decode.probe(str(path))
    a, b = parse_span(span, info.duration)
    out = out_dir(path) / "look"
    out.mkdir(parents=True, exist_ok=True)
    if not info.has_video:              # audio only: the sound strip is the whole picture
        sheet = Image.new("RGB", (1280, 160), "black")
        _sound_strip(sheet, path, a, b, [a, b], 0, 160)
        name = f"look_{a:09.2f}-{b:09.2f}.jpg"
        usage("look", key=out.parent.name, start=round(a, 2), end=round(b, 2), n=0)
        sheet.save(out / name, quality=88)
        return out / name
    n = max(1, min(n, int((b - a) * min(info.fps or 30, 30)) or 1))
    times = [a + (b - a) * (i + 0.5) / n for i in range(n)]
    files = _frames_at(str(path), times, out / "frames", 768 if n <= 2 else 512)
    cols = 1 if n == 1 else 2 if n <= 4 else 4
    tw = 1024 if n == 1 else 640 if n <= 4 else width
    th = round(tw * (info.height or 9) / (info.width or 16))
    rows = (n + cols - 1) // cols
    strip_h = 120 if info.has_audio else 0
    sheet = Image.new("RGB", (cols * tw, rows * th + strip_h), "black")
    for i, (t, f) in enumerate(zip(times, files)):
        tile = _fit(Image.open(f), tw, th)
        _label(tile, f"{t:.2f}s", 20)
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    if info.has_audio:
        _sound_strip(sheet, path, a, b, times, rows * th, strip_h)
    name = f"look_{a:09.2f}-{b:09.2f}.jpg"
    usage("look", key=out.parent.name, start=round(a, 2), end=round(b, 2), n=n)
    sheet.save(out / name, quality=88)
    return out / name


def _sound_strip(sheet: Image.Image, path: Path, a: float, b: float, times: list[float], y0: int, h: int) -> None:
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
            d.line((x, mid - v * (h - 50), x, mid), fill=(90, 160, 255) if v > 0.08 else (60, 60, 70))
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
        said = [c for c in js.get("speech", []) if c["end"] >= a and c["start"] <= b]
        for k, c in enumerate(said):
            x0 = max(0, (c["start"] - a) / (b - a) * W)
            x1 = (said[k + 1]["start"] - a) / (b - a) * W if k + 1 < len(said) else W
            text = f"\"{c['text']}\""
            while len(text) > 4 and d.textlength(text, font=f) > max(x1 - x0 - 8, 40) * (2 if k % 2 else 1.6):
                text = text[:-5] + "...\""          # fit before the next line starts (rows alternate)
            d.text((x0 + 4, y0 + 12 + 18 * (k % 2)), text, fill=(180, 255, 180), font=f)
    d.text((W - 300, y0 + 12), f"sound {a:.1f}-{b:.1f}s (orange = short sounds)", fill=(150, 150, 150), font=f)
