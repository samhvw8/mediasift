"""Retrieval check for `framesift find`, no LLM in the loop.

For every question in bench/qa/<video>/key.json whose answer is visible
("image" or "both"), run find with the question text and count a hit when one
of the top-k moments falls inside an evidence span. The same selection with
random scores gives the chance level for that video and k.

Variants:
  shipped       what `find` ranks by (query.frame_scores)
  visual        frame score only
  hybW          frame score + W x score of the speech window covering the frame
  dim256        both sides truncated to 256 dims and renormalised

Usage: python bench/search_eval.py RUNS_DIR QA_DIR [video ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from framesift.embed import Embedder
from framesift.query import check_identity, frame_scores, load, select

KS = (1, 3, 6)
PAD = 1.0


def hit(ts: list[float], spans: list[list[float]]) -> bool:
    return any(a - PAD <= t <= b + PAD for t in ts for a, b in spans)


def speech_per_frame(ix, tsc: np.ndarray) -> np.ndarray:
    out = np.zeros(len(ix.frame_t), np.float32)
    for i, t in enumerate(ix.frame_t):
        m = (ix.text_start - 1 <= t) & (t <= ix.text_end + 1)
        out[i] = max(tsc[m].max(), 0.0) if m.any() else 0.0
    return out


def renorm(x: np.ndarray, d: int) -> np.ndarray:
    y = x[..., :d]
    return y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-9)


def main() -> None:
    with Embedder(vision=False) as emb:
        run(emb)


def run(emb) -> None:
    runs, qa = Path(sys.argv[1]), Path(sys.argv[2])
    videos = sys.argv[3:] or sorted(p.name for p in qa.iterdir() if (p / "key.json").exists())
    rng = np.random.default_rng(0)
    totals: dict[str, dict[int, list[int]]] = {}
    for v in videos:
        key = json.loads((qa / v / "key.json").read_text())
        qs = [q for q in key["questions"] if q["type"] in ("image", "both") and not q.get("invalid")]
        ix = load(str(runs / v))
        check_identity(ix, emb.identity())
        Q = emb.texts([q["question"] for q in qs], "query")
        allowed = np.ones(len(ix.frame_t), bool)
        variants = {
            "shipped": lambda q: frame_scores(ix, q),
            "visual": lambda q: ix.frame_vec @ q,
            **{f"hyb{w}": (lambda q, w=w: ix.frame_vec @ q + w * speech_per_frame(ix, ix.text_vec @ q)
                           if len(ix.text) else ix.frame_vec @ q) for w in (0.25, 1.0)},
            "dim256": lambda q: renorm(ix.frame_vec, 256) @ renorm(q, 256),
        }
        rows = {name: {k: 0 for k in KS} for name in [*variants, "random"]}
        for qi, q in enumerate(qs):
            for name, f in variants.items():
                picks = select(f(Q[qi]), ix.frame_t, ix.frame_vec, max(KS), allowed)
                for k in KS:
                    rows[name][k] += hit([float(ix.frame_t[i]) for i in picks[:k]], q["evidence"])
            r = {k: 0.0 for k in KS}
            for _ in range(200):
                picks = select(rng.random(len(ix.frame_t)), ix.frame_t, ix.frame_vec, max(KS), allowed)
                for k in KS:
                    r[k] += hit([float(ix.frame_t[i]) for i in picks[:k]], q["evidence"]) / 200
            for k in KS:
                rows["random"][k] += r[k]
        print(f"\n{v}: {len(qs)} visual questions")
        for name, row in rows.items():
            print(f"  {name:8s} " + "  ".join(f"hit@{k} {row[k]:5.1f}/{len(qs)}" for k in KS))
            t = totals.setdefault(name, {k: [0, 0] for k in KS})
            for k in KS:
                t[k][0] += row[k]
                t[k][1] += len(qs)
    print("\nall videos")
    for name, t in totals.items():
        print(f"  {name:8s} " + "  ".join(f"hit@{k} {t[k][0] / t[k][1]:.2f}" for k in KS))


if __name__ == "__main__":
    main()
