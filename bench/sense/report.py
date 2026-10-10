"""Accuracy per arm and question type for one sense-bench round.

    uv run python bench/sense/report.py r1          # synthetic round
    uv run python bench/sense/report.py real/r2     # real-footage round
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(rnd: str) -> None:
    base = ROOT / rnd / "answers"
    by_type = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    total = defaultdict(lambda: [0, 0])
    images = defaultdict(int)
    rows = []
    for g in sorted(base.glob("*.grades.*json")):
        clip = g.name.split(".")[0]
        kf = ROOT / "clips" / f"{clip}.key.json"
        if not kf.exists():                       # real round: bench/sense/real/<video>/key.json
            kf = ROOT / "real" / clip / "key.json"
        key = {q["id"]: q for q in json.loads(kf.read_text())["questions"]}
        grades = json.loads(g.read_text())["grades"]
        for arm, gs in grades.items():
            ans = json.loads((base / f"{clip}.{arm}.json").read_text())
            images[arm] += ans.get("images_read", 0)
            given = {a["id"]: a["answer"] for a in ans["answers"]}
            for qid, ok in gs.items():
                ty = key[qid]["type"]
                by_type[ty][arm][0] += ok
                by_type[ty][arm][1] += 1
                total[arm][0] += ok
                total[arm][1] += 1
                rows.append((clip, qid, ty, arm, ok, given.get(qid, ""), key[qid]["answer"]))
    arms = sorted(total)
    print("type".ljust(10) + "".join(a.rjust(9) for a in arms))
    for ty in sorted(by_type):
        print(ty.ljust(10) + "".join(f"{by_type[ty][a][0]}/{by_type[ty][a][1]}".rjust(9) for a in arms))
    print("TOTAL".ljust(10) + "".join(f"{total[a][0]}/{total[a][1]}".rjust(9) for a in arms))
    print("images".ljust(10) + "".join(str(images[a]).rjust(9) for a in arms))
    if "-v" in sys.argv:
        for r in rows:
            if not r[4]:
                print(f"  x {r[0]}.{r[1]} [{r[2]}] {r[3]}: {r[5]!r}  (key: {r[6]!r})")


if __name__ == "__main__":
    main(sys.argv[1])
