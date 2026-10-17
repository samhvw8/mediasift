"""Real-footage round of the sense bench: temporal questions written from dense frames.

    uv run python bench/sense/real.py oracle [NAMES]    # 2 fps oracle sheets + transcript, writer briefs
    uv run python bench/sense/real.py arms ROUND        # arm folders (B uniform 1 fps, C mediasift see) + briefs

Synthetic clips have exact answers but were designed by the same hand as the
sensor. This round uses real videos and a writer agent that sees only a dense
oracle (every 0.5 s) and the transcript, never mediasift output. No agent can
hear, so audio truth is limited to speech timing; non-speech sound stays in the
synthetic round.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
VIDEOS = ["big_buck_bunny", "tears_of_steel", "python_100s", "cs50_lecture",
          # phone video (portrait), as daily use brings: an English talking-head reel, a Vietnamese TikTok
          "reel_en", "tiktok_vi"]
FONT = "/System/Library/Fonts/Helvetica.ttc"

WRITER = """You write a test of how well someone understands a video they could not watch. You are the only
one who sees the full material: {oracle} holds contact sheets with a frame every 0.5 s (each tile labelled
with its time in seconds), and {transcript} is the timed speech. Read ALL of the sheets in order and the transcript.

Write 8 questions about {video} that need understanding of TIME in the video, not just a single picture:
- order of events or steps ("what happens right after X", "which came first"),
- motion and direction ("which way does X move", "where does X go"),
- counting repeated actions or appearances ("how many times does X"),
- duration or brief events ("what appears briefly when ...", "for about how long"),
- picture and speech together ("what is on screen when the speaker says '...'").
Each answer must be unambiguous and checkable from the sheets and transcript; avoid anything you are unsure
of in a 0.5 s sampling (do not ask about events shorter than about 1 s unless clearly visible in two tiles).
Prefer questions whose answer is NOT given away by the transcript alone. Use plain descriptions of what is
visible, never tile numbers or sheet names. Mix the five kinds.

Write JSON (nothing else) to {out}:
{{"video": "{video}", "questions": [{{"id": "q01", "type": "sequence|motion|count|duration|deixis",
 "question": "...", "answer": "...", "accept": ["...", "..."], "evidence": [[start_s, end_s]]}}, ...]}}
Then reply with one line: WROTE 8
"""

MATERIAL = {
    "B": "the folder {dir}: transcript.txt (timed speech) and contact sheets sheet_*.jpg, each holding 12 frames "
         "sampled every 1 s, labelled with their time.",
    "C": "the folder {dir}: card.md (read it first) and the images it names (plus transcript.txt when present). "
         "You may also run `mediasift look {video} <start>-<end>` to see any time span of the video; "
         "it prints the path of an image to read.",
}


def sheets(video: Path, out: Path, fps: float, tile_w: int) -> int:
    tmp = out / "_f"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={fps},scale={tile_w}:-2",
                    str(tmp / "%05d.jpg")], check=True)
    files = sorted(tmp.glob("*.jpg"))
    f = ImageFont.truetype(FONT, 15)
    for s in range(0, len(files), 12):
        tiles = [Image.open(p) for p in files[s:s + 12]]
        w, h = tiles[0].size
        sheet = Image.new("RGB", (4 * w, 3 * h), "black")
        for k, im in enumerate(tiles):
            t = (s + k + 0.5) / fps
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 64, 20], fill="black")
            d.text((3, 2), f"{t:.1f}s", font=f, fill="yellow")
            sheet.paste(im, ((k % 4) * w, (k // 4) * h))
        sheet.save(out / f"sheet_{s // 12 + 1:03d}.jpg", quality=85)
    shutil.rmtree(tmp)
    return len(files)


def see_dir(video: Path) -> Path:
    r = subprocess.run([str(Path.home() / ".local/bin/mediasift"), "see", str(video)], capture_output=True, text=True,
                       check=True)
    return Path(r.stdout.strip().splitlines()[-1].split(" in ", 1)[1].rstrip(")"))


def transcript(video: Path) -> str:
    js = json.loads((see_dir(video) / "sensed.json").read_text())
    lines = [f"[{c['start']:.1f}-{c['end']:.1f}] {c['text']}" for c in js["speech"]]
    return "\n".join(lines) + "\n" if lines else "(no speech)\n"


def oracle(names: list[str] | None = None) -> None:
    base = ROOT / "real"
    (base / "prompts").mkdir(parents=True, exist_ok=True)
    for v in names or VIDEOS:
        video = REPO / "samples" / f"{v}.mp4"
        od = base / v / "oracle"
        if od.exists():
            shutil.rmtree(od)
        od.mkdir(parents=True)
        n = sheets(video, od, 2.0, 320)
        (base / v / "transcript.txt").write_text(transcript(video))
        (base / "prompts" / f"writer.{v}.md").write_text(WRITER.format(
            oracle=od, transcript=base / v / "transcript.txt", video=v, out=base / v / "key.json"))
        print(f"{v}: {n} oracle frames")


def arms(rnd: str) -> None:
    sys.path.insert(0, str(ROOT))
    from prep import ANSWER, GRADE, uniform_sheets
    base = ROOT / "real"
    prompts = base / rnd / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for v in VIDEOS:
        video = REPO / "samples" / f"{v}.mp4"
        key = json.loads((base / v / "key.json").read_text())
        qdir = base / rnd / v
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "questions.json").write_text(json.dumps(
            {"video": v, "questions": [{"id": q["id"], "question": q["question"]} for q in key["questions"]]}, indent=1))
        for arm in "BC":
            adir = qdir / arm
            if adir.exists():
                shutil.rmtree(adir)
            if arm == "B":
                adir.mkdir()
                uniform_sheets(video, adir)
                shutil.copy(base / v / "transcript.txt", adir / "transcript.txt")
            else:
                src = see_dir(video)
                adir.mkdir()
                for f in src.iterdir():
                    if f.suffix in (".md", ".jpg", ".txt") and f.name != "source.txt":
                        shutil.copy(f, adir / f.name)
            out = base / rnd / "answers" / f"{v}.{arm}.json"
            out.parent.mkdir(exist_ok=True)
            (prompts / f"answer.{v}.{arm}.md").write_text(ANSWER.format(
                questions=qdir / "questions.json", material=MATERIAL[arm].format(dir=adir, video=video), out=out, arm=arm))
        files = "\n".join(f"- {base / rnd / 'answers' / f'{v}.{a}.json'}" for a in "BC")
        (prompts / f"grade.{v}.md").write_text(GRADE.format(key=base / v / "key.json", files=files,
                                                            out=base / rnd / "answers" / f"{v}.grades.BC.json", video=v))
    print(f"briefs in {prompts}")


if __name__ == "__main__":
    {"oracle": lambda: oracle(sys.argv[2:] or None), "arms": lambda: arms(sys.argv[2])}[sys.argv[1]]()
