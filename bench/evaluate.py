"""Benchmark record: run mediasift on a set of videos and measure what it buys.

For each video:
- timings per stage, segment / keyframe / transcript counts;
- coverage: of the video's distinct moments (one frame per second), the share
  that some chosen image represents (content distance below the drift
  threshold mediasift itself uses), for mediasift and for uniform sampling
  with the *same number of images*;
- agent cost: image tokens for mediasift's frames and contact sheets versus
  1 fps sampling (Claude vision: about width*height/750 tokens per image).

Usage: python bench/evaluate.py OUT.json video[::srt] ... [--asr]
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from mediasift import __version__, visual
from mediasift.pipeline import Params, analyze


def tokens(w: int, h: int) -> int:
    return round(w * h / 750)


def coverage(track, chosen: list[int], thr: float, area_weight: float) -> float:
    """Share of 1-fps probe frames within `thr` of at least one chosen frame."""
    dt = float(np.median(np.diff(track.times)))
    step = max(1, round(1.0 / dt))
    probes = range(0, len(track), step)
    hit = 0
    for i in probes:
        if any(visual.content_distance(track, i, j, area_weight) <= thr for j in chosen):
            hit += 1
    return hit / len(probes)


def nearest(track, t: float) -> int:
    return int(np.clip(np.searchsorted(track.times, t), 0, len(track) - 1))


def run(video: str, srt: str | None, asr: bool, out_root: Path) -> dict:
    name = Path(video).stem
    t0 = time.perf_counter()
    r = analyze(video, str(out_root / name), Params(transcript=srt, asr=asr and not srt))
    wall = time.perf_counter() - t0
    m, tr = r.manifest, r.track
    content = m["content"]["type"]
    thr, aw = (0.06, 1.0) if content == "screen" else (0.12, 0.5)
    kf_all = [k for s in m["segments"] for k in s["keyframes"]]
    uniq = [nearest(tr, k["t"]) for k in kf_all if "file" in k]
    n = len(uniq)
    uni = [nearest(tr, (i + 0.5) * m["duration"] / n) for i in range(n)]
    segs = m["segments"]
    durs = [s["end"] - s["start"] for s in segs]
    w, h = 768, round(768 * m["media"]["height"] / max(m["media"]["width"], 1))
    sheets = [str(out_root / name / s) for s in m.get("sheets", [])]
    sheet_tokens = 0
    for sp in sheets:
        from PIL import Image
        sw, sh = Image.open(sp).size
        sheet_tokens += tokens(sw, sh)
    fps1 = int(m["duration"])
    return {
        "video": name,
        "duration": m["duration"],
        "content": content,
        "timings": {k: round(v, 1) for k, v in r.timings.items()} | {"wall": round(wall, 1)},
        "asr": m.get("asr"),
        "segments": len(segs),
        "segment_median_s": round(float(np.median(durs)), 1),
        "hard_cuts": m["stats"]["hard_cuts"],
        "camera_switches": m["stats"].get("camera_switches", 0),
        "keyframes": n,
        "keyframe_refs": sum(1 for k in kf_all if "same_as" in k),
        "segments_with_image": sum(1 for s in segs if s["keyframes"]),
        "segments_with_text": sum(1 for s in segs if s["transcript"]),
        "transcript_cues": m["stats"]["transcript_cues"],
        "coverage_mediasift": round(coverage(tr, uniq, thr, aw), 3),
        "coverage_uniform_same_n": round(coverage(tr, uni, thr, aw), 3),
        "tokens_mediasift_frames": n * tokens(w, h),
        "tokens_mediasift_sheets": sheet_tokens,
        "tokens_uniform_1fps": fps1 * tokens(w, h),
        "frames_uniform_1fps": fps1,
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    asr = "--asr" in sys.argv
    out = Path(args[0])
    rows = []
    for spec in args[1:]:
        video, _, srt = spec.partition("::")
        print(f"== {video}", file=sys.stderr)
        rows.append(run(video, srt or None, asr, out.parent / "runs"))
        print(json.dumps(rows[-1]), file=sys.stderr)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if sys.platform == "linux" else ""
    cpu = subprocess.run(["sh", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//' || sysctl -n machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    out.write_text(json.dumps({"mediasift": __version__, "host": platform.node(), "cpu": cpu, "gpu": gpu,
                               "at": time.strftime("%Y-%m-%d %H:%M"), "rows": rows}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
