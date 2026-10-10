"""Summarise the agent QA bench: accuracy and image tokens per arm.

  A  transcript only                      (text ceiling)
  B  transcript + uniform contact sheets  (same image count as C)
  C  storyboard + framesift contact sheets (today's default)
  F  storyboard + `find` / `look` grids on demand

Reads bench/qa/<video>/key.json, answers/<video>.<arm>.json and
answers/<video>.grades.json (written by the grader). Image tokens use the
Claude estimate width*height/750 (after its 1568 px resize) on the actual
files of each arm; for F, the sum over the grids the agent asked for.

Usage: python bench/qa_report.py QA_DIR RUNS_DIR [ANSWERS_SUBDIR [GRIDS_DIR]]
  ANSWERS_SUBDIR  default "answers"; GRIDS_DIR holds <video>/find|look (default RUNS_DIR)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ARMS = "ABCF"


def tokens(f: Path) -> int:
    """Claude scales an image whose long side exceeds 1568 px before counting."""
    w, h = Image.open(f).size
    scale = min(1.0, 1568 / max(w, h))
    return round(w * h * scale * scale / 750)


def mean_tokens(files: list[Path]) -> float:
    return sum(map(tokens, files)) / len(files) if files else 0.0


def main() -> None:
    qa, runs = Path(sys.argv[1]), Path(sys.argv[2])
    sub = sys.argv[3] if len(sys.argv) > 3 else "answers"
    grids = Path(sys.argv[4]) if len(sys.argv) > 4 else runs
    tot = {a: {"ok": 0, "n": 0, "img": 0, "tok": 0.0, "by": {}} for a in ARMS}
    print(f"{'video':16s} " + "  ".join(f"{a}: acc  imgs  ktok" for a in ARMS))
    for gf in sorted(qa.glob(f"{sub}/*.grades.json")):
        v = gf.name.split(".")[0]
        key = {q["id"]: q for q in json.loads((qa / v / "key.json").read_text())["questions"] if not q.get("invalid")}
        grades = json.loads(gf.read_text())["grades"]
        per = {
            "A": [],
            "B": sorted((qa / v / "B").glob("uniform_*.jpg")),
            "C": sorted((qa / v / "C").glob("sheet_*.jpg")),
            "F": sorted((grids / v).glob("find/[0-9]*.jpg")) + sorted((grids / v).glob("look/[0-9]*.jpg")),
        }
        cells = []
        for a in ARMS:
            g = grades.get(a, {})
            af = qa / sub / f"{v}.{a}.json"
            if not af.exists():
                cells.append(f"{'-':>15s}")
                continue
            ans = json.loads(af.read_text())
            n_img = int(ans.get("images_read", 0))
            # F: every grid it asked for is an image it read, so count them exactly;
            # the others: images read x the mean size of that arm's sheets.
            tok = sum(map(tokens, per[a])) if a == "F" and per[a] else n_img * mean_tokens(per[a])
            g = {qid: x for qid, x in g.items() if qid in key}  # drop questions marked invalid
            ok = sum(int(x) for x in g.values())
            t = tot[a]
            t["ok"] += ok
            t["n"] += len(key)
            t["img"] += n_img
            t["tok"] += tok
            for qid, x in g.items():
                ty = key[qid]["type"]
                b = t["by"].setdefault(ty, [0, 0])
                b[0] += int(x)
                b[1] += 1
            cells.append(f"{ok:2d}/{len(key)} {n_img:4d} {tok / 1000:5.1f}")
        print(f"{v:16s} " + "  ".join(cells))
        log = next((f for f in (grids / v / "queries.jsonl", qa / sub / f"{v}.queries.jsonl") if f.exists()), None)
        if log:
            ops = [json.loads(line)["op"] for line in log.read_text().splitlines()]
            print(f"{'':16s} F used find x{ops.count('find')}, look x{ops.count('look')}")
    print()
    for a in ARMS:
        t = tot[a]
        if not t["n"]:
            continue
        by = "  ".join(f"{ty} {b[0]}/{b[1]}" for ty, b in sorted(t["by"].items()))
        per_ok = t["tok"] / t["ok"] if t["ok"] else float("nan")
        print(f"{a}: {t['ok']}/{t['n']} correct ({t['ok'] / t['n']:.0%}) · {t['img']} images · "
              f"{t['tok'] / 1000:.1f}k image tokens · {per_ok:.0f} tokens per correct · {by}")


if __name__ == "__main__":
    main()
