"""Build arm folders and agent briefs for the sense bench.

    uv run python bench/sense/prep.py ROUND ARMS     # e.g. r1 ABC

Arms (each a folder the answerer may read, nothing else):
  A  mediasift v0.2: storyboard.md + contact sheets + frames/ (bench/sense/runs/v02/<clip>)
  B  prior-art shape: uniform 1 fps contact sheets (12 per image, time-labelled) + transcript.txt
  C  mediasift see: card.md + its images, and `mediasift look` on the clip (bench/sense/runs/see/<clip>)

Writes bench/sense/<ROUND>/<clip>/questions.json (no answers), arm folders under
bench/sense/<ROUND>/<clip>/<arm>/, and briefs under bench/sense/<ROUND>/prompts/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CLIPS = ["ui_flow", "motion", "count", "audio_scene", "deixis"]
FONT = "/System/Library/Fonts/Helvetica.ttc"

MATERIAL = {
    "A": "the folder {dir}: storyboard.md (read it first), the contact sheets sheet_*.jpg and frames/*.jpg.",
    "B": "the folder {dir}: transcript.txt (timed speech) and contact sheets sheet_*.jpg, each holding 12 frames "
         "sampled every 1 s, labelled with their time.",
    "C": "the folder {dir}: card.md (read it first) and the images it names. You may also run the command "
         "`mediasift look {video} <start>-<end>` (and other `mediasift look` options listed in card.md) "
         "to see any time span of the video; it prints the path of an image to read.",
}

ANSWER = """You are tested on how well you understand a video from limited material. You cannot watch or hear
the video; you only get the material below. Answer every question in {questions} using ONLY that material.
Do not open any other file in the repository (never open *.key.json, other rounds or other arms' folders,
the source code, or the .mp4 directly), and do not use the web.

YOUR MATERIAL:
{material}

Rules:
- Answer each question with a short answer (a word, number, name or short phrase).
- If the material does not let you know, give your best guess and set "sure": false. Never leave one blank.
- Keep a count of every image you read (each Read of a .jpg or .png counts as one image).

Write JSON (nothing else) to {out}:
{{"arm": "{arm}", "images_read": <int>, "answers": [{{"id": "q01", "answer": "...", "sure": true}}, ...]}}
Then reply with one line: ANSWERED <n> IMAGES <images_read>
"""

GRADE = """You grade answers in a video benchmark. Read {key} (questions, reference answers and accepted variants)
and the answer files listed below. For each arm and question, decide correct (1) or wrong (0).
An answer is correct when it means the same as the reference or an accepted variant; ignore case,
articles and small spelling differences; a number must match; a list must have the same items in the same
order; extra detail is fine if it does not contradict the reference; a vague answer that would fit many
videos is wrong.

Answer files:
{files}

Write JSON (nothing else) to {out}:
{{"video": "{video}", "grades": {{"<arm>": {{"q01": 1, ...}}, ...}}, "notes": {{"<arm>/q01": "only if borderline"}}}}
Then reply with one line: GRADED
"""


def uniform_sheets(video: Path, out: Path) -> None:
    """1 fps frames at 320 px, 12 per sheet (4 x 3), each tile labelled with its time."""
    tmp = out / "_f"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", "fps=1,scale=320:-2",
                    str(tmp / "%04d.jpg")], check=True)
    files = sorted(tmp.glob("*.jpg"))
    f = ImageFont.truetype(FONT, 16)
    for s in range(0, len(files), 12):
        tiles = [Image.open(p) for p in files[s:s + 12]]
        w, h = tiles[0].size
        sheet = Image.new("RGB", (4 * w, 3 * h), "black")
        for k, im in enumerate(tiles):
            t = s + k + 0.5  # fps filter takes the frame at the middle of each second
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 70, 22], fill="black")
            d.text((4, 2), f"{t:.1f}s", font=f, fill="yellow")
            sheet.paste(im, ((k % 4) * w, (k // 4) * h))
        sheet.save(out / f"sheet_{s // 12 + 1:02d}.jpg", quality=85)
    shutil.rmtree(tmp)


def transcript(manifest: Path) -> str:
    cues = json.loads(manifest.read_text()).get("transcript", [])
    lines = [f"[{s:.1f}-{e:.1f}] {t}" for s, e, t in cues]
    return "\n".join(lines) + "\n" if lines else "(no speech recognised)\n"


def main(rnd: str, arms: str) -> None:
    base = ROOT / rnd
    prompts = base / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        key = json.loads((ROOT / "clips" / f"{clip}.key.json").read_text())
        cdir = base / clip
        cdir.mkdir(parents=True, exist_ok=True)
        qs = [{"id": q["id"], "question": q["question"]} for q in key["questions"]]
        (cdir / "questions.json").write_text(json.dumps({"video": clip, "questions": qs}, indent=1))
        video = ROOT / "clips" / f"{clip}.mp4"
        for arm in arms:
            adir = cdir / arm
            if adir.exists():
                shutil.rmtree(adir)
            if arm == "A":
                shutil.copytree(ROOT / "runs" / "v02" / clip, adir,
                                ignore=shutil.ignore_patterns("manifest.json", "signals.npz"))
            elif arm == "B":
                adir.mkdir()
                uniform_sheets(video, adir)
                (adir / "transcript.txt").write_text(transcript(ROOT / "runs" / "v02" / clip / "manifest.json"))
            elif arm == "C":
                shutil.copytree(ROOT / "runs" / "see" / clip, adir)
            out = base / "answers" / f"{clip}.{arm}.json"
            out.parent.mkdir(exist_ok=True)
            (prompts / f"answer.{clip}.{arm}.md").write_text(ANSWER.format(
                questions=cdir / "questions.json", material=MATERIAL[arm].format(dir=adir, video=video),
                out=out, arm=arm))
        files = "\n".join(f"- {base / 'answers' / f'{clip}.{a}.json'}" for a in arms)
        (prompts / f"grade.{clip}.{arms}.md").write_text(GRADE.format(
            key=ROOT / "clips" / f"{clip}.key.json", files=files, out=base / "answers" / f"{clip}.grades.{arms}.json",
            video=clip))
    print(f"{base}: {len(CLIPS)} clips x arms {arms}; briefs in {prompts}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
