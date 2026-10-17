"""Write the sub-agent briefs for the agent QA bench (see bench/qa/README.md).

  python bench/qa_prompts.py writer                 one question writer per video
  python bench/qa_prompts.py answer ROUND [ARMS]    one answerer per video x arm (default ABCF)
  python bench/qa_prompts.py grade ROUND [ARMS]     one grader per video

Briefs go to bench/qa/prompts/; answers and grades to bench/qa/answers_<ROUND>/.
Each brief is self-contained: run it as `Read <brief> and follow it exactly`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa"
RUNS = ROOT / "results" / "runs_v03"
FS = ROOT.parent / ".venv" / "bin" / "mediasift"
VIDEOS = ["python_100s", "cs50_lecture", "tears_of_steel", "big_buck_bunny"]

MATERIAL = {
    "A": "The transcript only: {qa}/{v}/A/transcript.txt",
    "B": "Everything in {qa}/{v}/B/ : uniform_*.jpg contact sheets (frames evenly spaced through the video, "
         "tiles labelled with time) and transcript.txt.",
    "C": "Everything in {qa}/{v}/C/ : storyboard.md (segments, transcript per segment), the sheet_*.jpg contact "
         "sheets it lists (12 labelled keyframes each), and frames/*.jpg if you need full resolution.",
    "F": ("The text file {qa}/{v}/C/storyboard.md (segments, times, transcript) and a search tool over the video. "
          "You may NOT open the sheet_*.jpg or frames/*.jpg files that the storyboard mentions. You see pictures "
          "only through these commands, run with Bash:\n"
          "  {fs} find {runs}/{v} \"<what you want to see>\"        # top 6 distinct moments as one grid; prints "
          "the grid path and the speech at each moment\n"
          "  {fs} find {runs}/{v} \"<text>\" --between 60-120 -k 4  # limit to a time range / fewer tiles\n"
          "  {fs} look {runs}/{v} --at 12.5,40,41.5                 # exact moments as one grid of small tiles; a "
          "single moment (--at 80) comes at full width, to read small text\n"
          "Each command prints the path of a grid image; Read that image to see it. Describe what would be visible "
          "in the query (e.g. 'chart ranking programming languages', 'man with name caption'), not the question "
          "itself. Use as few grids as you need; read the storyboard text first, since many answers are in the "
          "speech."),
}


def fill(template: str, **kw: str) -> str:
    s = (QA / "prompts" / template).read_text()
    for k, v in kw.items():
        s = s.replace("{{" + k + "}}", v)
    if "{{" in s:
        raise SystemExit(f"{template}: unfilled placeholder in brief")
    return s


def main() -> None:
    what = sys.argv[1]
    out = QA / "prompts"
    if what == "writer":
        for v in VIDEOS:
            (out / f"writer.{v}.md").write_text(fill("writer.md", VIDEO=v, ORACLE=str(QA / v / "oracle"),
                                                     OUT=str(QA / v / "key.json")))
        return
    rnd, arms = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "ABCF")
    ans = QA / f"answers_{rnd}"
    ans.mkdir(exist_ok=True)
    for v in VIDEOS:
        if what == "answer":
            for a in arms:
                m = MATERIAL[a].format(qa=QA, v=v, fs=FS, runs=RUNS)
                (out / f"answer_{rnd}.{v}.{a}.md").write_text(
                    fill("answer.md", MATERIAL=m, QUESTIONS=str(QA / v / "questions.json"), ARM=a,
                         OUT=str(ans / f"{v}.{a}.json")))
        elif what == "grade":
            files = "\n".join(f"- {ans}/{v}.{a}.json  (arm {a})" for a in arms)
            (out / f"grade_{rnd}.{v}.md").write_text(
                fill("grade.md", KEY=str(QA / v / "key.json"), FILES=files, OUT=str(ans / f"{v}.grades.json"), VIDEO=v))
    print(f"wrote briefs to {out}")


if __name__ == "__main__":
    main()
