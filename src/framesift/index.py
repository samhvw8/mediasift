"""`framesift index OUT`: embed an analysis so an agent can search it.

Builds OUT/vectors.npz (and OUT/index/ probe images):
  frames  every keyframe plus a uniform probe every PROBE_STEP seconds (at
          most MAX_PROBES), so a search can land on moments no keyframe kept;
  speech  transcript cues merged into windows of about WINDOW_S seconds.
Both are embedded by EmbeddingGemma 2 into one space, stored at 768 dims as
float16, with the model identity so a later query from another model fails
instead of comparing incompatible vectors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__, decode

PROBE_STEP = 1.0     # seconds between probe frames
MAX_PROBES = 600     # longer videos get a wider step
PROBE_W = 512        # the 70-token vision budget reads ~530x300 px; more is wasted
WINDOW_S = 8.0       # speech window length


def log(msg: str) -> None:
    import sys
    print(f"framesift: {msg}", file=sys.stderr)


def speech_windows(cues: list, window: float = WINDOW_S) -> list[tuple[float, float, str]]:
    """Merge consecutive cues until a window spans `window` seconds; a long gap
    (a pause of 3 s or more) also closes it."""
    out, cur = [], []
    for c in cues:
        if cur and (c[0] - cur[-1][1] >= 3.0 or c[1] - cur[0][0] > window * 1.5):
            out.append(cur)
            cur = []
        cur.append(c)
        if cur[-1][1] - cur[0][0] >= window:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [(w[0][0], w[-1][1], " ".join(c[2].strip() for c in w).strip()) for w in out if any(c[2].strip() for c in w)]


def probe_times(duration: float) -> tuple[float, int]:
    step = max(PROBE_STEP, duration / MAX_PROBES)
    return step, max(1, int(duration / step))


def build(out_dir: str, emb) -> Path:
    out = Path(out_dir)
    m = json.loads((out / "manifest.json").read_text())
    if not m["media"]["has_video"]:
        raise SystemExit("index needs a video stream")
    t0 = time.perf_counter()
    ident = emb.identity()

    # Probe frames, decoded once at index width and kept for result grids.
    step, n = probe_times(m["duration"])
    w = PROBE_W
    h = max(2, round(m["media"]["height"] * w / m["media"]["width"] / 2) * 2)
    pdir = out / "index"
    pdir.mkdir(exist_ok=True)
    for old in pdir.glob("p*.jpg"):
        old.unlink()
    frames: list[tuple[float, str, str]] = []
    for i, (_, rgb) in enumerate(decode.video_frames(m["source"], 1.0 / step, w, h)):
        if i >= n:
            break
        t = i * step
        name = f"index/p{t:08.2f}s.jpg"
        Image.fromarray(rgb).save(out / name, quality=85)
        frames.append((t, name, "probe"))
    for seg in m["segments"]:
        for kf in seg["keyframes"]:
            if "file" in kf:
                frames.append((kf["t"], kf["file"], "key"))
    frames.sort()
    t1 = time.perf_counter()
    log(f"index: {len(frames)} frames ({n} probes every {step:.1f}s + keyframes) decoded in {t1 - t0:.1f}s")

    fvec = emb.images([str(out / f) for _, f, _ in frames])
    t2 = time.perf_counter()
    log(f"index: embedded {len(frames)} frames in {t2 - t1:.1f}s ({len(frames) / max(t2 - t1, 1e-6):.1f}/s)")

    wins = speech_windows(m.get("transcript", []))
    tvec = emb.texts([w[2] for w in wins], "document") if wins else np.zeros((0, fvec.shape[1]), np.float32)
    if wins:
        log(f"index: embedded {len(wins)} speech windows in {time.perf_counter() - t2:.1f}s")
    else:
        log("index: no transcript, speech search disabled (run analyze with --asr or --transcript)")

    meta = {**ident, "framesift": __version__, "probe_step": step, "window_s": WINDOW_S,
            "seconds": round(time.perf_counter() - t0, 1)}
    np.savez_compressed(
        out / "vectors.npz",
        frame_t=np.array([f[0] for f in frames], np.float32),
        frame_file=np.array([f[1] for f in frames]),
        frame_kind=np.array([f[2] for f in frames]),
        frame_vec=fvec.astype(np.float16),
        text_start=np.array([w[0] for w in wins], np.float32),
        text_end=np.array([w[1] for w in wins], np.float32),
        text=np.array([w[2] for w in wins]),
        text_vec=tvec.astype(np.float16),
        meta=np.array(json.dumps(meta)),
    )
    return out / "vectors.npz"
