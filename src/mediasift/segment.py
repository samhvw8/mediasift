"""Fuse visual, audio and codec evidence into segment boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import AudioTrack
from .decode import Packets


@dataclass
class Boundary:
    t: float
    strength: float
    hard: bool
    reasons: set[str] = field(default_factory=set)


@dataclass
class Signals:
    """Every curve resampled onto one uniform grid (grid[i] = i * step)."""
    step: float
    grid: np.ndarray
    visual_z: np.ndarray
    audio_z: np.ndarray
    activity: np.ndarray    # codec bits spent on predicted frames, ~1 = typical
    silent: np.ndarray      # bool


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    scale = max(mad, 0.25 * float(x.std()), 1e-4)
    return np.clip((x - med) / scale, 0, None)


def build_signals(duration: float, step: float, v_times, v_nov, audio: AudioTrack | None,
                  a_nov, pk: Packets | None) -> Signals:
    grid = np.arange(0.0, duration, step)
    vz = robust_z(np.interp(grid, v_times, v_nov)) if v_times is not None and len(v_times) > 1 else np.zeros_like(grid)
    az = np.zeros_like(grid)
    silent = np.zeros(len(grid), bool)
    if audio is not None and len(audio.times):
        idx = np.clip(np.searchsorted(grid, audio.times, side="right") - 1, 0, len(grid) - 1)
        pooled = np.zeros(len(grid), np.float32)
        np.maximum.at(pooled, idx, a_nov)
        az = robust_z(pooled)
        sil_count = np.bincount(idx, weights=audio.silent.astype(float), minlength=len(grid))
        tot = np.bincount(idx, minlength=len(grid))
        silent = sil_count > 0.5 * np.maximum(tot, 1)
    activity = np.ones(len(grid), np.float32)
    if pk is not None and len(pk.times) and (~pk.is_key).any():
        p = ~pk.is_key
        idx = np.clip(np.searchsorted(grid, pk.times[p], side="right") - 1, 0, len(grid) - 1)
        bits = np.bincount(idx, weights=pk.sizes[p], minlength=len(grid))
        typical = np.median(bits[bits > 0]) if (bits > 0).any() else 1.0
        activity = (bits / typical).astype(np.float32)
    return Signals(step, grid, vz, az, activity, silent)


def codec_scenecuts(pk: Packets) -> list[float]:
    """I-frames that break the encoder's regular GOP cadence. Encoders (x264/x265
    `scenecut`, most hardware encoders) insert an unscheduled I-frame when the
    next frame cannot be predicted from the last one, which is a scene cut."""
    kt = pk.times[pk.is_key]
    if len(kt) < 4:
        return []
    gaps = np.diff(kt)
    gop = np.median(gaps)
    return [float(kt[i + 1]) for i, g in enumerate(gaps) if g < 0.8 * gop and kt[i + 1] > 0]


def _peaks(x: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    order = np.argsort(-x)
    taken = np.zeros(len(x), bool)
    out = []
    for i in order:
        if x[i] < threshold:
            break
        if taken[max(0, i - min_dist): i + min_dist + 1].any():
            continue
        taken[i] = True
        out.append(int(i))
    return sorted(out)


def _nearest_pause(t: float, pauses: list[tuple[float, float]], radius: float) -> float | None:
    best = None
    for s, e in pauses:
        mid = (s + e) / 2
        if s - radius <= t <= e + radius and (best is None or abs(mid - t) < abs(best - t)):
            best = mid
    return best


def find_boundaries(sig: Signals, cuts: list[tuple[float, float]], scenecuts: list[float],
                    pauses: list[tuple[float, float]], duration: float, *, sensitivity: float,
                    min_seg: float, max_seg: float, snap: float) -> list[Boundary]:
    bounds: list[Boundary] = [Boundary(t, 100.0 + s, True, {"visual_cut"}) for t, s in cuts]

    combined = sig.visual_z + sig.audio_z
    for i in _peaks(combined, sensitivity, max(1, int(min_seg / sig.step / 2))):
        t = float(sig.grid[i])
        reasons = set()
        if sig.audio_z[i] >= sensitivity / 2:
            reasons.add("audio_change")
        if sig.visual_z[i] >= sensitivity / 2:
            reasons.add("visual_change")
        if not reasons:
            reasons.add("joint_change")
        p = _nearest_pause(t, pauses, snap)
        if p is not None:
            t = p
            reasons.add("at_pause")
        bounds.append(Boundary(t, float(combined[i]), False, reasons))

    for t in scenecuts:
        near = [b for b in bounds if abs(b.t - t) <= 0.5]
        if near:
            for b in near:
                b.reasons.add("codec_scenecut")
        else:
            bounds.append(Boundary(t, sensitivity, False, {"codec_scenecut"}))

    bounds = [b for b in bounds if min_seg / 2 <= b.t <= duration - min_seg / 2]
    bounds = _merge_close(bounds, min_seg / 2, merge=True)
    bounds = _merge_close(bounds, min_seg, merge=False)
    return _split_long(bounds, sig, pauses, duration, min_seg, max_seg)


def _merge_close(bounds: list[Boundary], gap: float, merge: bool) -> list[Boundary]:
    """Repeatedly resolve the closest too-near pair: fold the weaker into the
    stronger (merge=True, evidence for the same moment) or just drop it."""
    bounds = sorted(bounds, key=lambda b: b.t)
    while True:
        gaps = [bounds[i + 1].t - bounds[i].t for i in range(len(bounds) - 1)]
        close = [i for i, g in enumerate(gaps) if g < gap]
        if not close:
            return bounds
        i = min(close, key=lambda k: gaps[k])
        a, b = bounds[i], bounds[i + 1]
        keep, lose = (a, b) if (a.strength, -a.t) >= (b.strength, -b.t) else (b, a)
        if merge:
            keep.reasons |= lose.reasons
            keep.hard = keep.hard or lose.hard
        bounds.remove(lose)


def _split_long(bounds, sig: Signals, pauses, duration, min_seg, max_seg) -> list[Boundary]:
    edges = [0.0] + [b.t for b in bounds] + [duration]
    out = list(bounds)
    for s, e in zip(edges[:-1], edges[1:]):
        out.extend(_split_span(s, e, sig, pauses, min_seg, max_seg))
    return sorted(out, key=lambda b: b.t)


def _split_span(s, e, sig: Signals, pauses, min_seg, max_seg) -> list[Boundary]:
    """Cut an over-long span into ceil(len / max_seg) near-equal pieces, each
    cut moved to the best nearby point (a pause, or the strongest change)
    within a quarter piece of its even-spacing target."""
    n = int(np.ceil((e - s) / max_seg))
    if n <= 1:
        return []
    piece = (e - s) / n
    score = sig.visual_z + sig.audio_z + 3.0 * sig.silent
    out = []
    for k in range(1, n):
        target = s + k * piece
        lo = np.searchsorted(sig.grid, max(s + min_seg, target - piece / 4))
        hi = np.searchsorted(sig.grid, min(e - min_seg, target + piece / 4))
        t = float(sig.grid[lo + int(np.argmax(score[lo:hi]))]) if hi > lo else target
        reasons = {"max_duration"}
        p = _nearest_pause(t, pauses, 1.0)
        if p is not None and abs(p - target) <= piece / 4:
            t = p
            reasons.add("at_pause")
        out.append(Boundary(t, 0.0, False, reasons))
    return out
