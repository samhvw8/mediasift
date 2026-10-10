"""Score an existing manifest's keyframe choice with the same coverage metric
as evaluate.py, so an older version's output can be compared on equal terms.

Usage: python bench/score_manifest.py OUT.json video::manifest.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluate import coverage, nearest
from framesift import decode, visual


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for spec in sys.argv[2:]:
        video, _, man = spec.partition("::")
        m = json.loads(Path(man).read_text())
        tr = visual.build_track(decode.video_frames(video, 5.0, visual.ANALYSIS_W, visual.ANALYSIS_H))
        screen, _ = visual.looks_like_screen(tr)
        thr, aw = (0.06, 1.0) if screen else (0.12, 0.5)
        kfs = [k for s in m["segments"] for k in s["keyframes"]]
        chosen = [nearest(tr, k["t"]) for k in kfs if "file" in k]
        n = len(chosen)
        uni = [nearest(tr, (i + 0.5) * m["duration"] / n) for i in range(n)]
        segs = m["segments"]
        rows.append({
            "video": Path(video).stem, "framesift": m.get("framesift"), "segments": len(segs),
            "hard_cuts": m["stats"]["hard_cuts"], "keyframes": n,
            "keyframe_refs": sum(1 for k in kfs if "same_as" in k),
            "segments_with_image": sum(1 for s in segs if s["keyframes"]),
            "segments_with_text": sum(1 for s in segs if s["transcript"]),
            "coverage": round(coverage(tr, chosen, thr, aw), 3),
            "coverage_uniform_same_n": round(coverage(tr, uni, thr, aw), 3),
        })
        print(json.dumps(rows[-1]), file=sys.stderr)
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
