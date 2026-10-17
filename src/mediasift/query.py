"""Search an indexed analysis: the agent asks, mediasift answers with one grid.

  find  rank frames by meaning against a text query (or against the frame at
        a moment, `--like T`), keep the top k that are not near-duplicates,
        and return them as one labelled grid plus the best speech matches;
  look  a grid of exactly the moments asked for, decoded at full width.

Every call is appended to OUT/queries.jsonl, so real agent questions become
benchmark material.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import decode
from .render import grid

MIN_SEP_S = 2.0       # results closer than this in time are one moment
DUP_COS = 0.95        # results more similar than this are one picture
SPEECH_W = 0.5        # weight of the speech around a frame in its score (bench: 0.5 and 1.0 tie, 0.25 is lower)


@dataclass
class Index:
    out: Path
    manifest: dict
    meta: dict
    frame_t: np.ndarray
    frame_file: np.ndarray
    frame_vec: np.ndarray      # float32, unit rows
    text_start: np.ndarray
    text_end: np.ndarray
    text: np.ndarray
    text_vec: np.ndarray


def load(out_dir: str) -> Index:
    out = Path(out_dir)
    f = out / "vectors.npz"
    if not f.exists():
        raise SystemExit(f"{f} not found: run `mediasift index {out}` first")
    z = np.load(f)
    return Index(out, json.loads((out / "manifest.json").read_text()), json.loads(str(z["meta"])),
                 z["frame_t"], z["frame_file"], z["frame_vec"].astype(np.float32),
                 z["text_start"], z["text_end"], z["text"], z["text_vec"].astype(np.float32))


def check_identity(ix: Index, ident: dict) -> None:
    """Vectors from another model live in another space. Comparing them returns
    plausible nonsense, so refuse. Queries are text, and text vectors agree
    across engines (PyTorch, llama.cpp Metal and CUDA: cosine >= 0.999), so the
    engine and image budget of the index do not have to match the query's."""
    for k in ("model", "dim"):
        if ix.meta.get(k) != ident.get(k):
            raise SystemExit(f"vectors.npz was built with {k}={ix.meta.get(k)!r} but the embedder has "
                             f"{ident.get(k)!r}: re-run `mediasift index {ix.out}`")


def fmt(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def segment_at(m: dict, t: float) -> int:
    for seg in m["segments"]:
        if seg["start"] <= t < seg["end"]:
            return seg["id"]
    return m["segments"][-1]["id"] if m["segments"] else 0


def speech_at(m: dict, t: float, pad: float = 2.0) -> str:
    return " ".join(c[2].strip() for c in m.get("transcript", []) if c[0] - pad <= t <= c[1] + pad)


def keyframe_near(m: dict, t: float, tol: float = 1.0) -> str | None:
    best = None
    for seg in m["segments"]:
        for kf in seg["keyframes"]:
            if "file" in kf and abs(kf["t"] - t) <= tol and (best is None or abs(kf["t"] - t) < best[0]):
                best = (abs(kf["t"] - t), kf["file"])
    return best[1] if best else None


def parse_span(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    a, b = s.split("-", 1)
    return float(a), float(b)


def frame_scores(ix: Index, q: np.ndarray, speech: bool = True) -> np.ndarray:
    """Cosine of each frame to the query, plus SPEECH_W x the best cosine of
    the speech heard within 1 s of it. A frame shown while the narrator talks
    about the query is more likely the moment asked for: on the 4 bench videos
    this lifted hit@3 from 0.80 to 0.87 and was never worse on any video."""
    s = ix.frame_vec @ q
    if speech and len(ix.text):
        ts = ix.text_vec @ q
        near = (ix.text_start[None, :] - 1 <= ix.frame_t[:, None]) & (ix.frame_t[:, None] <= ix.text_end[None, :] + 1)
        s = s + SPEECH_W * np.where(near, ts[None, :], 0.0).max(axis=1).clip(min=0.0)
    return s


def select(scores: np.ndarray, times: np.ndarray, vecs: np.ndarray, k: int,
           allowed: np.ndarray) -> list[int]:
    """Greedy top-k that skips results within MIN_SEP_S of, or near-identical
    to, one already chosen: k distinct moments, not one moment k times."""
    chosen: list[int] = []
    for i in np.argsort(-scores):
        if not allowed[i]:
            continue
        if any(abs(times[i] - times[j]) < MIN_SEP_S or float(vecs[i] @ vecs[j]) > DUP_COS for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen


def _next_name(d: Path) -> Path:
    d.mkdir(exist_ok=True)
    return d / f"{len(list(d.glob('[0-9][0-9][0-9].jpg'))) + 1:03d}.jpg"


def _log(ix: Index, rec: dict) -> None:
    with open(ix.out / "queries.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}, ensure_ascii=False) + "\n")


def find(ix: Index, emb, query: str | None = None, like: float | None = None, k: int = 6,
         between: tuple[float, float] | None = None, speech_k: int = 3) -> dict:
    t0 = time.perf_counter()
    if (query is None) == (like is None):
        raise ValueError("give a text query or --like, not both")
    if query is not None:
        check_identity(ix, emb.identity())
    allowed = np.ones(len(ix.frame_t), bool)
    if between:
        allowed &= (ix.frame_t >= between[0]) & (ix.frame_t <= between[1])
    if query is not None:
        q = emb.texts([query], "query")[0]
        scores = frame_scores(ix, q)
    else:
        i = int(np.argmin(np.abs(ix.frame_t - like)))
        q = ix.frame_vec[i]
        allowed &= np.abs(ix.frame_t - like) > 3.0
        scores = frame_scores(ix, q, speech=False)
    picks = select(scores, ix.frame_t, ix.frame_vec, k, allowed)
    hits = []
    for r, i in enumerate(picks, 1):
        t = float(ix.frame_t[i])
        hits.append({"rank": r, "t": round(t, 2), "score": round(float(scores[i]), 3),
                     "segment": segment_at(ix.manifest, t),
                     "image": str(ix.frame_file[i]), "keyframe": keyframe_near(ix.manifest, t),
                     "speech": speech_at(ix.manifest, t)})
    speech = []
    if query is not None and len(ix.text):
        ts = ix.text_vec @ q
        ok = np.ones(len(ts), bool)
        if between:
            ok &= (ix.text_end >= between[0]) & (ix.text_start <= between[1])
        for i in [j for j in np.argsort(-ts) if ok[j]][:speech_k]:
            speech.append({"start": round(float(ix.text_start[i]), 2), "end": round(float(ix.text_end[i]), 2),
                           "score": round(float(ts[i]), 3), "text": str(ix.text[i])})
    sheet = None
    if hits:
        sheet = grid([(ix.out / h["image"], f"#{h['rank']} {fmt(h['t'])} S{h['segment']:02d}") for h in hits],
                     _next_name(ix.out / "find"))
    res = {"query": query, "like": like, "k": k, "between": between, "hits": hits, "speech": speech,
           "grid": str(sheet) if sheet else None, "seconds": round(time.perf_counter() - t0, 3)}
    _log(ix, {"op": "find", **{k_: v for k_, v in res.items() if k_ != "speech"}})
    return res


def look(ix: Index, times: list[float], width: int = 768) -> dict:
    t0 = time.perf_counter()
    d = ix.out / "look"
    d.mkdir(exist_ok=True)
    dur = ix.manifest["duration"]
    times = [min(max(t, 0.0), max(dur - 0.05, 0.0)) for t in times]
    files = [d / f"t{t:08.2f}s.jpg" for t in times]
    todo = [(t, f) for t, f in zip(times, files) if not f.exists()]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: decode.extract_frame(ix.manifest["source"], j[0], str(j[1]), width), todo))
    hits = [{"t": round(t, 2), "segment": segment_at(ix.manifest, t), "image": str(f.relative_to(ix.out)),
             "speech": speech_at(ix.manifest, t)} for t, f in zip(times, files)]
    # One moment comes at full width, to read small text; several come as a
    # contact-sheet grid. On the bench, full-width tiles for 2-4 moments let
    # agents spend 2x the image tokens for no accuracy gain.
    cols, tile_w = (1, width) if len(hits) == 1 else (3, 384)
    sheet = grid([(ix.out / h["image"], f"{fmt(h['t'])} S{h['segment']:02d}") for h in hits], _next_name(d),
                 cols=cols, tile_w=tile_w)
    res = {"at": times, "hits": hits, "grid": str(sheet),
           "seconds": round(time.perf_counter() - t0, 3)}
    _log(ix, {"op": "look", **res})
    return res


def render_find(res: dict) -> str:
    what = f'"{res["query"]}"' if res["query"] is not None else f"frames like {fmt(res['like'])}"
    lines = [f"# find {what}" + (f" in {fmt(res['between'][0])}-{fmt(res['between'][1])}" if res["between"] else ""),
             ""]
    if not res["hits"]:
        lines.append("no frames in range")
        return "\n".join(lines)
    lines.append(f"grid: {res['grid']}  (tiles labelled #rank time segment; ranked by meaning, so the "
                 "top tile is the best guess, not proof: check it. For small text use `look` on its time)")
    lines.append("")
    for h in res["hits"]:
        kf = f" · keyframe `{h['keyframe']}`" if h["keyframe"] else ""
        lines.append(f"- #{h['rank']} {fmt(h['t'])} S{h['segment']:02d} score {h['score']}{kf}")
        if h["speech"]:
            lines.append(f"  > {h['speech'][:240]}")
    if res["speech"]:
        lines += ["", "speech matches:"]
        for s in res["speech"]:
            lines.append(f"- {fmt(s['start'])}-{fmt(s['end'])} score {s['score']}: {s['text'][:300]}")
    return "\n".join(lines)


def render_look(res: dict) -> str:
    lines = [f"grid: {res['grid']}", ""]
    for h in res["hits"]:
        lines.append(f"- {fmt(h['t'])} S{h['segment']:02d} `{h['image']}`")
        if h["speech"]:
            lines.append(f"  > {h['speech'][:240]}")
    return "\n".join(lines)
