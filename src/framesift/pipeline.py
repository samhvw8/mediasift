"""analyze(): video in, segments + keyframes + transcript out."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, audio, decode, keyframes, segment, transcript, visual

V_WIN = 1.5   # seconds each side for visual novelty
A_WIN = 2.0   # seconds each side for audio novelty


@dataclass
class Params:
    fps: float = 5.0             # analysis rate (tiny frames; cheap)
    fast: bool = False           # decode codec I-frames only instead of resampling
    min_seg: float = 2.0
    max_seg: float = 90.0
    sensitivity: float = 4.0     # robust-z threshold for soft (non-cut) boundaries
    cut_threshold: float = 0.12  # absolute frame distance for a hard cut
    drift: float | None = None   # change from last kept frame that earns a new keyframe (None: by content)
    content: str = "auto"        # auto | screen | camera
    max_frames: int = 80         # budget of unique keyframe images (0 = unlimited)
    width: int = 768             # extracted keyframe width
    sheets: bool = True
    transcript: str | None = None
    asr: bool = False
    asr_backend: str = "auto"    # auto | parakeet-mlx | mlx-whisper | faster-whisper
    asr_model: str | None = None # default depends on the backend
    lang: str | None = None      # ISO 639-1; detected when missing
    snap: float = 1.0            # move soft boundaries into a pause this close


@dataclass
class Result:
    manifest: dict
    out_dir: Path
    timings: dict = field(default_factory=dict)
    track: "visual.VisualTrack | None" = None   # for evaluation; not serialized


def log(msg: str) -> None:
    print(f"framesift: {msg}", file=sys.stderr)


def fmt_ts(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m):02d}:{s:04.1f}"


def _visual_track(path: str, info: decode.MediaInfo, p: Params, pk: decode.Packets | None):
    W, H = visual.ANALYSIS_W, visual.ANALYSIS_H
    if not p.fast:
        return visual.build_track(decode.video_frames(path, p.fps, W, H))
    assert pk is not None
    n = int(pk.is_key.sum())
    if n < max(4, info.duration / 30):
        # Screen recordings and some VP9/AV1 encodes carry almost no I-frames.
        log(f"--fast: only {n} I-frames in {info.duration:.0f}s, falling back to full decode")
        p.fast = False
        return visual.build_track(decode.video_frames(path, p.fps, W, H))
    return visual.build_track(decode.keyframe_frames(path, W, H))


def analyze(path: str, out_dir: str, p: Params) -> Result:
    decode.require_ffmpeg()
    t0 = time.perf_counter()
    timings = {}
    info = decode.probe(path)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    pk = decode.packets(path) if info.has_video else None
    track = _visual_track(path, info, p, pk) if info.has_video else None
    timings["visual"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    pcm = decode.audio_pcm(path, audio.SR) if info.has_audio else None
    atrack = audio.build_track(pcm) if pcm is not None else None
    timings["audio"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    v_nov = visual.novelty(track, V_WIN) if track is not None and len(track) > 1 else None
    a_nov = audio.novelty(atrack, A_WIN) if atrack is not None else None
    step = 1.0 / p.fps if not p.fast else 0.2
    sig = segment.build_signals(info.duration, step, track.times if track is not None else None, v_nov,
                                atrack, a_nov, pk)
    cuts = visual.detect_cuts(track, p.cut_threshold, 6.0) if track is not None and not p.fast else []
    if p.fast and track is not None:
        # Consecutive I-frames far apart visually: the later one is the cut.
        cuts = [(float(track.times[i]), float(track.step[i])) for i in range(1, len(track))
                if track.step[i] >= p.cut_threshold]
    switches: list[float] = []
    if cuts and track is not None:
        recurring = visual.camera_switches(track, [t for t, _ in cuts])
        switches = [c[0] for c, r in zip(cuts, recurring) if r]
        cuts = [c for c, r in zip(cuts, recurring) if not r]
    scenecuts = segment.codec_scenecuts(pk) if pk is not None else []
    if switches:
        # The encoder flags camera switches too; they are not story changes either.
        sw = np.array(switches)
        scenecuts = [t for t in scenecuts if np.abs(sw - t).min() > 1.0]
        # Picture novelty spikes at every switch too; silence it there so a
        # soft boundary needs other evidence (sound, a genuinely new picture).
        near = np.abs(sig.grid[:, None] - sw[None, :]).min(axis=1) <= V_WIN if len(sig.grid) * len(sw) < 5e7 \
            else np.array([np.abs(sw - g).min() <= V_WIN for g in sig.grid])
        sig.visual_z[near] = 0.0
    pauses = atrack.pauses if atrack is not None else []
    bounds = segment.find_boundaries(sig, cuts, scenecuts, pauses, info.duration, sensitivity=p.sensitivity,
                                     min_seg=p.min_seg, max_seg=p.max_seg, snap=p.snap)
    edges = [0.0] + [b.t for b in bounds] + [info.duration]
    spans = list(zip(edges[:-1], edges[1:]))
    starts = [{"start"}] + [sorted(b.reasons) for b in bounds]
    timings["segment"] = time.perf_counter() - t2

    # Per-segment evidence, also used to rank anchors under a tight budget.
    seg_stats = []
    def interior(s: float, e: float, pad: float) -> np.ndarray:
        # Novelty windows straddle boundaries; measure away from the edges.
        mid, half = (s + e) / 2, max((e - s) / 2 - pad, sig.step)
        m = (sig.grid >= mid - half) & (sig.grid < mid + half)
        if not m.any():
            m[min(len(sig.grid) - 1, int(mid / sig.step))] = True
        return m

    for s, e in spans:
        m = (sig.grid >= s) & (sig.grid < e)
        if not m.any():
            m[min(len(sig.grid) - 1, int(s / sig.step))] = True
        seg_stats.append({
            "visual_change": round(float(sig.visual_z[interior(s, e, V_WIN)].mean()), 2),
            "audio_change": round(float(sig.audio_z[interior(s, e, A_WIN)].mean()), 2),
            "speech_ratio": round(float(1.0 - sig.silent[m].mean()), 2) if atrack is not None else None,
            "activity": round(float(sig.activity[m].mean()), 2) if pk is not None else None,
        })

    t3 = time.perf_counter()
    kfs: list[keyframes.Keyframe] = []
    content, flat = None, None
    if track is not None:
        is_screen, flat = visual.looks_like_screen(track)
        content = p.content if p.content != "auto" else ("screen" if is_screen else "camera")
        # Small text changes matter on a screen; on camera the same area is noise and motion.
        drift = p.drift if p.drift is not None else (0.06 if content == "screen" else 0.12)
        screen = content == "screen"
        kfs = keyframes.select(track, spans, drift=drift, area_weight=1.0 if screen else 0.5,
                               min_gap_s=0.4 if screen else 1.0)
        keyframes.mark_duplicates(track, kfs, screen=screen)
        weight = {i: 1.0 + st["visual_change"] + st["audio_change"] + (st["activity"] or 0) for i, st in enumerate(seg_stats)}
        before = sum(k.duplicate_of is None for k in kfs)
        kfs = keyframes.apply_budget(kfs, p.max_frames, weight)
        after = sum(k.duplicate_of is None for k in kfs)
        if after < before:
            log(f"budget: kept {after} of {before} candidate keyframes (raise --max-frames for more)")
    names: dict[int, str] = {}
    jobs = []
    for n, k in enumerate(kf for kf in kfs if kf.duplicate_of is None):
        name = f"frames/f{n + 1:03d}_{k.t:08.2f}s.jpg"
        names[id(k)] = name
        jobs.append((k.t, str(out / name)))
    with ThreadPoolExecutor(8) as ex:  # each ffmpeg seek is mostly single-threaded
        list(ex.map(lambda j: decode.extract_frame(path, j[0], j[1], p.width), jobs))
    timings["extract"] = time.perf_counter() - t3

    texts = [""] * len(spans)
    cues: list[transcript.Cue] = []
    asr_info: dict | None = None
    if p.transcript:
        cues = transcript.load(p.transcript)
    elif p.asr and pcm is not None:
        from . import asr
        t_asr = time.perf_counter()
        ar = asr.transcribe(pcm, p.asr_backend, p.asr_model, p.lang)
        cues, asr_info = ar.cues, {"backend": ar.backend, "model": ar.model, "language": ar.language,
                                   "seconds": ar.seconds}
        timings["asr"] = time.perf_counter() - t_asr
    if cues:
        texts = transcript.assign(cues, spans)

    segments = []
    for i, ((s, e), reasons, st) in enumerate(zip(spans, starts, seg_stats)):
        kf_out = []
        for k in kfs:
            if k.segment != i:
                continue
            if k.duplicate_of is not None:
                kf_out.append({"t": round(k.t, 2), "same_as": names[id(k.duplicate_of)]})
            else:
                kf_out.append({"t": round(k.t, 2), "file": names[id(k)],
                               "why": "anchor" if k.anchor else f"drift {k.drift:.2f}"})
        n_sw = sum(1 for t in switches if s <= t < e)
        segments.append({"id": i + 1, "start": round(s, 2), "end": round(e, 2),
                         "start_ts": fmt_ts(s), "end_ts": fmt_ts(e),
                         "reasons": sorted(reasons), **st, "camera_switches": n_sw, "keyframes": kf_out, "transcript": texts[i]})

    manifest = {
        "framesift": __version__,
        "source": str(Path(path).resolve()),
        "duration": round(info.duration, 2),
        "media": {"width": info.width, "height": info.height, "fps": round(info.fps, 3),
                  "has_video": info.has_video, "has_audio": info.has_audio},
        "params": {k: v for k, v in asdict(p).items() if k not in ("transcript",)},
        "content": {"type": content, "identical_frames": round(flat, 3) if flat is not None else None},
        "asr": asr_info,
        "transcript": [[round(c.start, 2), round(c.end, 2), c.text] for c in cues],
        "stats": {
            "segments": len(segments),
            "keyframe_images": len(jobs),
            "analysed_frames": len(track) if track is not None else 0,
            "source_frames_approx": int(info.duration * info.fps),
            "hard_cuts": len(cuts),
            "camera_switches": len(switches),
            "codec_scenecuts": len(scenecuts),
            "pauses": len(pauses),
            "transcript_cues": len(cues),
        },
        "segments": segments,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    np.savez_compressed(out / "signals.npz", grid=sig.grid, visual_z=sig.visual_z, audio_z=sig.audio_z,
                        activity=sig.activity, silent=sig.silent)
    if p.sheets and jobs:
        from .render import contact_sheets
        manifest["sheets"] = contact_sheets(out, manifest)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    from .render import storyboard
    (out / "storyboard.md").write_text(storyboard(manifest))
    timings["total"] = time.perf_counter() - t0
    return Result(manifest, out, timings, track)
