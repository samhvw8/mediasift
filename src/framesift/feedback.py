"""Corrections to the card's sound names, and what they teach.

An agent that checks a card line with `framesift look` and finds it wrong runs
`framesift fix VIDEO TIME "what it is"`. Each fix goes to <cache>/feedback.jsonl with the
tagger label the card showed there. A label corrected the same way at least MIN_FIXES times,
by at least AGREE of its fixes, becomes a learned name: the card then shows both
("throat clearing (usually: cough)"), never silently replacing what the tagger heard.
Learned names apply to cards sensed after the fixes (`see --force` refreshes a cached card).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import sound

MIN_FIXES = 3
AGREE = 0.75


def log_path() -> Path:
    return sound.cache_root() / "feedback.jsonl"


def _was(card_dir: Path, t: float) -> str | None:
    """The sound label the card showed at time t (within half a second), or None."""
    sounds = json.loads((card_dir / "sensed.json").read_text()).get("sounds", [])
    best = None
    for s in sounds:
        if s["kind"] == "silence":
            continue
        d = (min(abs(t - x) for x in s["times"]) if s["kind"] == "hits" and s["times"]
             else 0.0 if s["start"] <= t <= s["end"] else min(abs(t - s["start"]), abs(t - s["end"])))
        if d <= 0.5 and (best is None or d < best[0]):
            best = (d, s["label"])
    return best[1] if best else None


def record(card_dir: Path, t: float, says: str) -> dict:
    card = json.loads((card_dir / "sensed.json").read_text()).get("card")
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": card_dir.name, "card": card, "at": round(t, 2),
           "was": _was(card_dir, t), "says": says.strip().lower()}
    f = log_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    learned.cache_clear()
    return row


def rows() -> list[dict]:
    f = log_path()
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


@lru_cache(maxsize=1)
def learned() -> dict[str, str]:
    by: dict[str, Counter] = {}
    for r in rows():
        if r.get("was"):
            by.setdefault(r["was"], Counter())[r["says"]] += 1
    out = {}
    for was, c in by.items():
        says, n = c.most_common(1)[0]
        if n >= MIN_FIXES and n / sum(c.values()) >= AGREE and says != was.lower():
            out[was] = says
    return out


def report() -> str:
    rs = rows()
    if not rs:
        return f"no corrections yet ({log_path()})"
    pairs = Counter((r.get("was") or "(card heard nothing)", r["says"]) for r in rs)
    lines = [f"{len(rs)} corrections in {log_path()}", ""]
    lines += [f"{n:3d}x  {was} -> {says}" for (was, says), n in pairs.most_common()]
    got = learned()
    lines += ["", "learned names (on new cards): " + (", ".join(f"{a} -> {b}" for a, b in got.items()) or "none yet")]
    return "\n".join(lines)
