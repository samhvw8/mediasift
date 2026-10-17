"""Pick the few frames worth looking at, codec style.

A video encoder stores an I-frame, then P-frames that only encode the delta,
and inserts a new I-frame once the delta is too costly. We do the same with
meaning instead of bits: keep an anchor frame, measure drift from *the last
kept frame* (not from the previous frame, so slow pans and slides that build
up bullet by bullet still add up), and keep a new frame once drift crosses a
threshold. A static talking head costs one frame; a busy screen recording
costs one per visible change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .visual import VisualTrack, changed_area, content_distance, pair_distance


@dataclass
class Keyframe:
    idx: int            # index into the visual track
    t: float
    segment: int
    drift: float        # distance from the previous kept frame (1.0 for anchors)
    anchor: bool
    duplicate_of: "Keyframe | None" = None   # an earlier kept frame this one repeats


def _stable_pick(track: VisualTrack, lo: int, hi: int) -> int:
    """Within [lo, hi) the frame least in motion, tie-broken by sharpness,
    so we do not keep a mid-transition or motion-blurred frame."""
    if hi - lo <= 1:
        return lo
    nxt = np.append(track.step[lo + 1:hi + 1], track.step[hi - 1] if hi < len(track) else 0.0)[: hi - lo]
    motion = track.step[lo:hi] + nxt
    sharp = track.sharp[lo:hi] / (track.sharp[lo:hi].max() + 1e-9)
    return lo + int(np.argmin(motion - 0.02 * sharp))


def select(track: VisualTrack, segments: list[tuple[float, float]], *, drift: float,
           area_weight: float = 1.0, min_gap_s: float = 0.4, settle_s: float = 0.6) -> list[Keyframe]:
    dt = float(np.median(np.diff(track.times))) if len(track) > 1 else 1.0
    settle = max(1, int(round(settle_s / dt)))
    gap = max(1, int(round(min_gap_s / dt)))  # like a codec's minimum GOP length
    out: list[Keyframe] = []
    for si, (s, e) in enumerate(segments):
        lo = int(np.searchsorted(track.times, s - 1e-6))
        hi = int(np.searchsorted(track.times, e - 1e-6))
        if hi <= lo:
            lo, hi = min(lo, len(track) - 1), min(lo, len(track) - 1) + 1
        first = _stable_pick(track, lo, min(hi, lo + max(settle, (hi - lo) // 4)))
        out.append(Keyframe(first, float(track.times[first]), si, 1.0, True))
        last, j = first, first + gap
        while j < hi:
            d = content_distance(track, j, last, area_weight)
            if d >= drift:
                k = _stable_pick(track, j, min(hi, j + settle))
                dk = content_distance(track, k, last, area_weight)
                if dk < drift:  # settled back (flash, bounce): not a real change
                    j += 1
                    continue
                out.append(Keyframe(k, float(track.times[k]), si, dk, False))
                last, j = k, k + gap
            else:
                j += 1
    return out


def mark_duplicates(track: VisualTrack, kfs: list[Keyframe], screen: bool = False) -> None:
    """A frame that looks like one already kept (the host cutting back to the
    same camera) is referenced, not shown again. On screens any visible change
    matters, so only near-identical frames count; on camera the same angle
    with a moved hand is the same shot (recurring shots measure < 0.04)."""
    for i, k in enumerate(kfs):
        for j in range(i - 1, -1, -1):  # nearest earlier first
            ref = kfs[j]
            if ref.duplicate_of is not None:
                continue
            if screen:
                same = content_distance(track, k.idx, ref.idx) <= 0.02
            else:
                same = (pair_distance(track, k.idx, ref.idx) <= 0.04
                        and changed_area(track, k.idx, ref.idx) <= 0.25)
            if same:
                k.duplicate_of = ref
                break


def apply_budget(kfs: list[Keyframe], budget: int, seg_weight: dict[int, float]) -> list[Keyframe]:
    """Keep at most `budget` unique frames. Drift frames go first (smallest
    change first). Then anchors, dropping the one whose loss hurts least:
    low segment weight *and* close neighbours, so one busy minute cannot eat
    the whole budget and leave an hour of the video without a picture."""
    unique = [k for k in kfs if k.duplicate_of is None]
    if budget <= 0 or len(unique) <= budget:
        return kfs
    drop: set[int] = set()
    extra = len(unique) - budget
    for k in sorted((k for k in unique if not k.anchor), key=lambda k: k.drift)[:extra]:
        drop.add(id(k))
    anchors = sorted((k for k in unique if k.anchor), key=lambda k: k.t)
    while len(unique) - len(drop) > budget and len(anchors) > 1:
        def cost(i: int) -> float:
            prev = anchors[i - 1].t if i > 0 else anchors[i].t - (anchors[1].t - anchors[0].t)
            nxt = anchors[i + 1].t if i + 1 < len(anchors) else anchors[i].t + (anchors[-1].t - anchors[-2].t)
            return (1.0 + seg_weight.get(anchors[i].segment, 0.0)) * (nxt - prev)
        i = min(range(len(anchors)), key=cost)
        drop.add(id(anchors.pop(i)))
    # A duplicate survives only while the frame it points at does.
    return [k for k in kfs if id(k) not in drop and (k.duplicate_of is None or id(k.duplicate_of) not in drop)]
