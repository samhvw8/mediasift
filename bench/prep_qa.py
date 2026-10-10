"""Prepare the question-answering eval: one folder per video per condition.

  oracle/   dense labelled frame sheets + full timed transcript (question writer only)
  A/        transcript only
  B/        transcript + N uniformly spaced frames (N = framesift's image count)
  C/        framesift storyboard.md + contact sheets + frames
  D/        C + meaning.md

Usage: python bench/prep_qa.py EVAL_DIR RUNS_DIR video.mp4[::oracle_seconds[::oracle_step]] ...
oracle_seconds limits the oracle (and so the questions) to the first part of a long video.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W, COLS, PER_SHEET = 384, 3, 12


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def grab(video: str, times: list[float], dest: Path, width: int = 768) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for t in times:
        p = dest / f"t{t:08.2f}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                        "-vf", f"scale='min({width},iw)':-2", "-q:v", "4", str(p)], check=True)
        out.append(p)
    return out


def sheets(frames: list[tuple[float, Path]], dest: Path, prefix: str) -> list[str]:
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    names = []
    for n in range(0, len(frames), PER_SHEET):
        chunk = frames[n:n + PER_SHEET]
        ims = [Image.open(p).convert("RGB") for _, p in chunk]
        ims = [im.resize((TILE_W, round(im.height * TILE_W / im.width))) for im in ims]
        th = max(i.height for i in ims)
        sheet = Image.new("RGB", (COLS * TILE_W, ((len(ims) + COLS - 1) // COLS) * th), "black")
        d = ImageDraw.Draw(sheet)
        for i, ((t, _), im) in enumerate(zip(chunk, ims)):
            x, y = (i % COLS) * TILE_W, (i // COLS) * th
            sheet.paste(im, (x, y))
            d.rectangle((x, y, x + 90, y + 26), fill="black")
            d.text((x + 5, y + 3), fmt(t), fill="yellow", font=font)
        name = f"{prefix}_{n // PER_SHEET + 1:02d}.jpg"
        sheet.save(dest / name, quality=82)
        names.append(name)
    return names


def transcript_txt(m: dict, limit: float | None = None) -> str:
    return "\n".join(f"[{fmt(a)}-{fmt(b)}] {t}" for a, b, t in m.get("transcript", []) if limit is None or a < limit)


def main() -> None:
    ev, runs = Path(sys.argv[1]), Path(sys.argv[2])
    for spec in sys.argv[3:]:
        parts = spec.split("::")
        video = parts[0]
        name = Path(video).stem
        run = runs / name
        m = json.loads((run / "manifest.json").read_text())
        dur = m["duration"]
        oracle_s = float(parts[1]) if len(parts) > 1 else dur
        step = float(parts[2]) if len(parts) > 2 else 2.0
        root = ev / name
        if root.exists():
            shutil.rmtree(root)
        tx = transcript_txt(m)

        o = root / "oracle"
        o.mkdir(parents=True)
        ts = [i * step + step / 2 for i in range(int(oracle_s / step))]
        frames = list(zip(ts, grab(video, ts, o / "_frames", 480)))
        sheets(frames, o, "dense")
        shutil.rmtree(o / "_frames")
        (o / "transcript.txt").write_text(transcript_txt(m, oracle_s))

        (root / "A").mkdir()
        (root / "A" / "transcript.txt").write_text(tx)

        b = root / "B"
        b.mkdir()
        n = m["stats"]["keyframe_images"]
        ut = [(i + 0.5) * dur / n for i in range(n)]
        sheets(list(zip(ut, grab(video, ut, b / "_frames"))), b, "uniform")
        shutil.rmtree(b / "_frames")
        (b / "transcript.txt").write_text(tx)

        c = root / "C"
        c.mkdir()
        shutil.copy(run / "storyboard.md", c)
        for s in m.get("sheets", []):
            shutil.copy(run / s, c)
        shutil.copytree(run / "frames", c / "frames")

        if (run / "meaning.md").exists():
            shutil.copytree(c, root / "D")
            shutil.copy(run / "meaning.md", root / "D")
        (root / "info.json").write_text(json.dumps({"video": name, "duration": dur, "oracle_seconds": oracle_s,
                                                    "oracle_step": step, "images": n}, indent=2))
        print(f"{name}: oracle {len(ts)} frames, B/C {n} images")


if __name__ == "__main__":
    main()
