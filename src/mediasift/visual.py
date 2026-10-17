"""Visual track: tiny per-frame features and the change signals derived from them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

ANALYSIS_W, ANALYSIS_H = 128, 72  # decode size: cheap, still enough to see a new line of text appear
THUMB_W, THUMB_H = 32, 18         # 4x4-pixel blocks; stored per frame for distances
BLOCK_LUMA_T = 0.04               # block counts as changed past this brightness delta...
BLOCK_EDGE_T = 0.03               # ...or this edge-density delta
HIST_BINS = 64                    # 4x4x4 RGB histogram


@dataclass
class VisualTrack:
    times: np.ndarray    # (N,) seconds
    thumbs: np.ndarray   # (N, THUMB_H, THUMB_W) float32 gray in [0, 1]
    edges: np.ndarray    # (N, THUMB_H, THUMB_W) float32 edge density per block
    hists: np.ndarray    # (N, 64) float32, rows sum to 1
    sharp: np.ndarray    # (N,) Laplacian variance, higher = crisper
    step: np.ndarray     # (N,) distance to the previous sample (0 for the first)

    def __len__(self) -> int:
        return len(self.times)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) / 255.0


def _hist(rgb: np.ndarray) -> np.ndarray:
    q = rgb >> 6
    idx = (q[..., 0].astype(np.int32) << 4) | (q[..., 1].astype(np.int32) << 2) | q[..., 2]
    h = np.bincount(idx.ravel(), minlength=HIST_BINS).astype(np.float32)
    return h / h.sum()


def _sharpness(g: np.ndarray) -> float:
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def distance(thumb_a, hist_a, thumb_b, hist_b) -> np.ndarray:
    """Frame distance in [0, 1]: half colour-distribution change, half structural change.

    The histogram term ignores motion (a pan keeps the same colours), the
    thumbnail term catches layout changes with the same palette (a new slide).
    """
    h = np.abs(hist_a - hist_b).sum(axis=-1) / 2.0
    p = np.abs(thumb_a - thumb_b).mean(axis=(-2, -1))
    return 0.5 * h + 0.5 * p


def _pool(x: np.ndarray) -> np.ndarray:
    return x.reshape(THUMB_H, x.shape[0] // THUMB_H, THUMB_W, x.shape[1] // THUMB_W).mean(axis=(1, 3))


def build_track(frames: Iterable[tuple[float, np.ndarray]]) -> VisualTrack:
    times, thumbs, edges, hists, sharp = [], [], [], [], []
    for t, rgb in frames:
        g = _gray(rgb.astype(np.float32))
        grad = np.zeros_like(g)
        grad[:, :-1] += np.abs(np.diff(g, axis=1))
        grad[:-1, :] += np.abs(np.diff(g, axis=0))
        times.append(t)
        thumbs.append(_pool(g).astype(np.float32))
        edges.append(_pool(grad).astype(np.float32))
        hists.append(_hist(rgb))
        sharp.append(_sharpness(g))
    if not times:
        raise ValueError("no video frames decoded")
    th = np.stack(thumbs)
    hi = np.stack(hists)
    step = np.zeros(len(times), np.float32)
    step[1:] = distance(th[1:], hi[1:], th[:-1], hi[:-1])
    return VisualTrack(np.array(times), th, np.stack(edges), hi, np.array(sharp), step)


def pair_distance(track: VisualTrack, i: int, j: int) -> float:
    return float(distance(track.thumbs[i], track.hists[i], track.thumbs[j], track.hists[j]))


def _shift_min_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-block |a - b|, minimised over +-1 block shifts of b: a tiny motion
    search, like a codec's, so camera shake and small moves do not count as
    new content."""
    pb = np.pad(b, 1, mode="edge")
    h, w = a.shape
    return np.min([np.abs(a - pb[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)


def changed_area(track: VisualTrack, i: int, j: int) -> float:
    """Fraction of the screen whose content changed between frames i and j.
    Unlike a whole-frame mean, a new chat message or bullet point in 10% of
    the screen reads as 0.10 no matter how busy the rest of the frame is."""
    luma = _shift_min_delta(track.thumbs[i], track.thumbs[j]) > BLOCK_LUMA_T
    edge = _shift_min_delta(track.edges[i], track.edges[j]) > BLOCK_EDGE_T
    return float((luma | edge).mean())


def content_distance(track: VisualTrack, i: int, j: int, area_weight: float = 1.0) -> float:
    """What keyframe selection uses: whole-frame change or localized change,
    whichever is larger. On camera footage localized change is mostly motion,
    so callers down-weight it."""
    return max(pair_distance(track, i, j), area_weight * changed_area(track, i, j))


def _rolling(x: np.ndarray, half: int, fn) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = fn(x[max(0, i - half): i + half + 1])
    return out


def detect_cuts(track: VisualTrack, min_score: float, k: float, context_s: float = 5.0) -> list[tuple[float, float]]:
    """Hard cuts: a step distance that is both large in absolute terms and an
    outlier against the local baseline (so a shaky handheld shot does not cut
    every frame). Returns (time, score)."""
    s = track.step
    n = len(s)
    if n < 3:
        return []
    dt = float(np.median(np.diff(track.times))) or 1.0
    half = max(2, int(context_s / dt))
    med = _rolling(s, half, np.median)
    mad = _rolling(np.abs(s - med), half, np.median)
    cuts = []
    for i in range(1, n):
        if s[i] < min_score or s[i] < med[i] + k * (1.4826 * mad[i] + 0.01):
            continue
        if s[i] < s[max(1, i - 2): i + 3].max():
            continue
        # A flash frame spikes on entry and exit; a real cut changes what comes after.
        after = min(n - 1, i + max(1, int(0.4 / dt)))
        if pair_distance(track, i - 1, after) < 0.6 * min_score:
            continue
        cuts.append((float(track.times[i]), float(s[i])))
    return cuts


def novelty(track: VisualTrack, window_s: float) -> np.ndarray:
    """How different the next `window_s` looks from the previous `window_s`.
    Peaks on hard cuts and on gradual ones (fades, dissolves, slow scene drift)."""
    n = len(track)
    dt = float(np.median(np.diff(track.times))) if n > 1 else 1.0
    w = max(1, int(round(window_s / dt)))
    ct = np.concatenate([np.zeros((1, THUMB_H, THUMB_W), np.float64), np.cumsum(track.thumbs, axis=0, dtype=np.float64)])
    ch = np.concatenate([np.zeros((1, HIST_BINS), np.float64), np.cumsum(track.hists, axis=0, dtype=np.float64)])
    out = np.zeros(n, np.float32)
    for i in range(1, n):
        a, b = max(0, i - w), min(n, i + w)
        left_t = (ct[i] - ct[a]) / (i - a)
        right_t = (ct[b] - ct[i]) / (b - i)
        left_h = (ch[i] - ch[a]) / (i - a)
        right_h = (ch[b] - ch[i]) / (b - i)
        out[i] = distance(left_t, left_h, right_t, right_h)
    return out


def looks_like_screen(track: VisualTrack) -> tuple[bool, float]:
    """Screen content repeats itself exactly between changes; a camera never
    does, even on a tripod (sensor noise survives compression). Measured on
    real files: screen recordings 0.45-0.85 identical consecutive frames,
    camera footage (film, animation, a filmed lecture) 0.00-0.06. Flat area
    alone misfires on a dark stage."""
    identical = float((track.step[1:] < 0.001).mean()) if len(track) > 1 else 0.0
    return identical >= 0.25, identical


def camera_switches(track: VisualTrack, cut_times: list[float], lookback: int = 40,
                    same_shot: float = 0.06) -> list[bool]:
    """For each cut, is the shot it opens a camera angle seen in one of the last
    `lookback` shots? Interviews and podcasts cut host/guest/wide over and over;
    those cuts change the camera, not the story. Measured on a 1h45m two-camera
    podcast: 392 of 569 shots sit within 0.04 of a recent earlier shot, new
    content (B-roll, graphics) above 0.1."""
    edges = [float(track.times[0])] + list(cut_times) + [float(track.times[-1]) + 1e-3]
    reps = []
    for s, e in zip(edges[:-1], edges[1:]):
        lo, hi = int(np.searchsorted(track.times, s)), int(np.searchsorted(track.times, e))
        reps.append((lo + hi) // 2 if hi > lo else min(lo, len(track) - 1))
    out = []
    for i in range(1, len(reps)):
        near = min((pair_distance(track, reps[i], reps[j]) for j in range(max(0, i - lookback), i)), default=1.0)
        out.append(near <= same_shot)
    return out
