"""Speech: load an existing transcript or run local ASR if it is installed."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    start: float
    end: float
    text: str


_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mi, se, frac = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(frac.ljust(3, "0")) / 1000


def load(path: str) -> list[Cue]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
        items = data.get("segments", data) if isinstance(data, dict) else data
        return [Cue(float(c["start"]), float(c["end"]), str(c["text"]).strip()) for c in items]
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->")
                text = " ".join(re.sub(r"<[^>]+>", "", t) for t in lines[i + 1:]).strip()
                if text:
                    cues.append(Cue(_ts(a), _ts(b), text))
                break
    return cues


def assign(cues: list[Cue], segments: list[tuple[float, float]]) -> list[str]:
    """A cue goes to every segment it substantially overlaps (1.5 s or 30% of
    the cue, whichever is smaller), else to the one it overlaps most.
    Midpoint-only assignment left short segments silent while someone was
    talking over them; a thin edge overlap should not duplicate text."""
    out: list[list[str]] = [[] for _ in segments]
    for c in cues:
        span = max(c.end - c.start, 1e-6)
        ov = [max(0.0, min(c.end, e) - max(c.start, s)) for s, e in segments]
        hits = [i for i, o in enumerate(ov) if o >= min(1.5, 0.3 * span) and o > 0]
        if not hits:
            best = max(range(len(segments)), key=lambda i: ov[i])
            hits = [best] if ov[best] > 0 or c.start >= segments[-1][0] else []
        for i in hits:
            out[i].append(c.text)
    return [" ".join(t) for t in out]
