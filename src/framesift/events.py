"""Visual events: where the picture changed, how, and for how long.

A still-image model loses three things between sampled frames: brief events
(a toast shown for 0.7 s, a lamp that flashes), the order of small changes,
and motion. So instead of picking frames, this module reads every analysed
frame as a 32 x 18 grid of colour blocks and turns runs of change into events:

  change    a region changed and stayed changed (a dialog opens, text grows)
  blink     a region changed and changed back within the event (a flash)
  motion    something travelled across the frame (with its path)
  animating a region kept changing in place (a spinner, video playing)
  cut       the whole picture changed at once
  scene     the whole picture kept changing (camera move, busy footage)

Each event carries its region (where, how big, what colour), its before/after
frames, and links: a change that is undone later is reported as shown for N s.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

GW, GH = 32, 18                 # block grid
MERGE_GAP_S = 0.2               # active frames closer than this belong to one event
FULL = 0.6                      # share of the frame that counts as "whole picture"

COLOURS = {
    "black": (20, 20, 20), "white": (240, 240, 240), "gray": (128, 128, 128), "red": (210, 40, 40),
    "orange": (240, 140, 30), "yellow": (235, 220, 40), "green": (50, 160, 70), "cyan": (40, 200, 210),
    "blue": (50, 90, 220), "purple": (130, 60, 180), "pink": (240, 130, 180), "brown": (130, 80, 40),
}


DECODE_W, DECODE_H = 256, 144   # 8 x 8 pixels per block: a typed character or a spinner dot still shows


@dataclass
class Grid:
    """Per-frame colour blocks and how much each block changed since the previous frame."""
    times: np.ndarray          # (N,)
    blocks: np.ndarray         # (N, GH, GW, 3) float16 in [0, 1], block mean colour
    diff: np.ndarray           # (N, GH, GW) float16, mean colour change since the previous frame
    pix: np.ndarray            # (N, GH, GW) float16, largest local brightness change inside the block
    thr: float                 # change thresholds for this video: block mean ...
    pthr: float                # ... and largest local change
    identical: float           # share of frames identical to the previous one (screens repeat, cameras never)
    shift: np.ndarray | None = None   # (N, 3): content motion over the last ~0.5 s, in frame widths and
                                      # heights per second (+x right, +y down), and the correlation peak

    def changed(self) -> np.ndarray:
        return (self.diff.astype(np.float32) > self.thr) | (self.pix.astype(np.float32) > self.pthr)


LAG_S = 0.5                     # camera motion is measured between frames this far apart


def _shift(a: np.ndarray, b: np.ndarray, win: np.ndarray) -> tuple[float, float, float]:
    """How far the content of `a` moved to become `b` (dx, dy in pixels, sub-pixel), and the
    phase-correlation peak: near 1 for a clean global shift, near 0 when nothing moved as one."""
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    R = B * np.conj(A)
    r = np.fft.irfft2(R / (np.abs(R) + 1e-9), s=a.shape)
    k = int(np.argmax(r))
    py, px = divmod(k, r.shape[1])
    h, w = r.shape

    def sub(c: float, m: float, p: float) -> float:   # parabola through three samples
        d = m - 2 * c + p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m - p) / d

    dy = py + sub(r[py, px], r[(py - 1) % h, px], r[(py + 1) % h, px])
    dx = px + sub(r[py, px], r[py, (px - 1) % w], r[py, (px + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dx), float(dy), float(r[py, px])


def build_grid(frames: Iterable[tuple[float, np.ndarray]]) -> Grid:
    times, blocks, pix, shifts = [], [], [], []
    prev = None
    back: deque = deque()          # (t, luma) of the last ~LAG_S seconds, for camera motion
    win = None
    for t, rgb in frames:
        h, w, _ = rgb.shape
        f = rgb.astype(np.float32) / 255.0
        blocks.append(f.reshape(GH, h // GH, GW, w // GW, 3).mean(axis=(1, 3)).astype(np.float16))
        # Luma smoothed 2 x 2 so single-pixel noise does not count, then the largest change per block.
        y = (f @ np.array([0.299, 0.587, 0.114], np.float32)).reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        if prev is None:
            pix.append(np.zeros((GH, GW), np.float16))
        else:
            d = np.abs(y - prev)
            pix.append(d.reshape(GH, d.shape[0] // GH, GW, d.shape[1] // GW).max(axis=(1, 3)).astype(np.float16))
        if win is None:
            win = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1])).astype(np.float32)
        old = None
        while back and back[0][0] <= t - 0.9 * LAG_S:
            old = back.popleft()
        if old is not None:
            back.appendleft(old)          # keep the newest frame that is old enough for the next one
            dx, dy, pk = _shift(old[1], y, win)
            span = max(t - old[0], 1e-3)
            shifts.append((dx / y.shape[1] / span, dy / y.shape[0] / span, pk))
        else:
            shifts.append((0.0, 0.0, 0.0))
        back.append((t, y))
        prev = y
        times.append(t)
    if not blocks:
        raise ValueError("no video frames decoded")
    bl = np.stack(blocks)
    px = np.stack(pix)
    diff = np.zeros(bl.shape[:3], np.float16)
    diff[1:] = np.abs(bl[1:].astype(np.float32) - bl[:-1].astype(np.float32)).max(axis=-1)
    # Compression noise and camera grain set the floor; real change sits far above it.
    nz, npx = diff[1:].astype(np.float32), px[1:].astype(np.float32)
    floor = float(np.percentile(nz, 75)) if nz.size else 0.0
    pfloor = float(np.percentile(npx, 75)) if npx.size else 0.0
    ident = float((px[1:].reshape(len(px) - 1, -1).max(axis=1) < 0.02).mean()) if len(px) > 1 else 0.0
    return Grid(np.array(times), bl, diff, px, max(0.05, 4 * floor), max(0.08, 4 * pfloor), ident,
                np.array(shifts, np.float32))


def camera_moves(g: Grid, screen: bool = False, min_speed: float = 0.04, min_peak: float = 0.1,
                 min_len: float = 1.5) -> list[dict]:
    """Stretches where the whole picture travelled one way: a camera pan or tilt (or, on a screen
    recording, a scroll). Content moving left means the camera turned right. A subject crossing a
    still background does not count (the background wins the correlation); shake fails min_len.
    min_len 1.5 s: one editing jump (a punch-in the cut test misses) looks like motion in every
    0.5 s window that spans it, ~1.1 s in all; a phone reel got 4 such "pans", real pans last longer."""
    if g.shift is None or len(g.times) < 3:
        return []
    out: list[dict] = []
    cur: dict | None = None
    for t, (dx, dy, pk) in zip(g.times, g.shift):
        speed = float(np.hypot(dx, dy))
        lab = None
        if pk >= min_peak and speed >= min_speed:
            h = "" if abs(dx) < 0.5 * abs(dy) else ("left" if dx > 0 else "right")     # camera turns against the content
            v = "" if abs(dy) < 0.5 * abs(dx) else ("up" if dy > 0 else "down")
            if screen:     # content moving up = the page scrolls down
                h = "" if not h else ("left" if dx > 0 else "right")
                v = "" if not v else ("up" if dy > 0 else "down")
            lab = (h, v)
        if lab and cur and cur["lab"] == lab and t - cur["last"] <= 0.35:
            cur["last"], cur["speeds"] = float(t), cur["speeds"] + [speed]
            continue
        if cur and t - cur["last"] > 0.35:
            out.append(cur)
            cur = None
        if lab and (cur is None or cur["lab"] != lab):
            if cur:
                out.append(cur)
            cur = {"lab": lab, "first": float(t), "last": float(t), "speeds": [speed]}
    if cur:
        out.append(cur)
    moves = []
    for m in out:
        start, end = max(float(g.times[0]), m["first"] - LAG_S), m["last"]
        if end - start < min_len:
            continue
        h, v = m["lab"]
        if screen:
            words = "scrolls " + " and ".join(x for x in (
                {"up": "down", "down": "up"}.get(v, ""), {"left": "right", "right": "left"}.get(h, "")) if x)
        else:
            words = " and ".join(x for x in (f"pans {h}" if h else "", f"tilts {v}" if v else "") if x)
        moves.append({"start": round(start, 2), "end": round(end, 2), "words": words,
                      "speed": round(float(np.mean(m["speeds"])), 3)})
    return moves


@dataclass
class Region:
    x0: float
    y0: float
    x1: float
    y1: float                  # normalised bbox, 0..1
    area: float                # share of the frame's blocks
    colour: str | None = None

    def words(self) -> str:
        cx, cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        v = "top" if cy < 0.34 else "bottom" if cy > 0.66 else ""
        h = "left" if cx < 0.34 else "right" if cx > 0.66 else ""
        where = f"{v}-{h}" if v and h else v or h or "center"
        size = "tiny" if self.area < 0.01 else "small" if self.area < 0.05 else "medium" if self.area < 0.25 else "large"
        return f"{size} {self.colour + ' ' if self.colour else ''}area at {where}"

    def box(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass
class Event:
    id: int
    kind: str                  # change | blink | motion | animating | cut | scene
    start: float
    end: float
    before: float              # time of the last calm frame before
    after: float               # time of the first calm frame after
    peak: float                # time of the frame most different from `before` (the moment to show for a blink)
    region: Region
    path: list[tuple[float, float, float]] = field(default_factory=list)   # (t, x, y) for motion
    shown_until: float | None = None      # a change undone later: visible from start to this time
    undoes: int | None = None             # this event reverts event #undoes
    text_added: list[str] = field(default_factory=list)
    text_removed: list[str] = field(default_factory=list)
    sounds: list[str] = field(default_factory=list)


def _colour(rgb: np.ndarray) -> str:
    c = np.asarray(rgb, np.float32) * 255
    return min(COLOURS, key=lambda k: float(((np.array(COLOURS[k]) - c) ** 2).sum()))


def _region(mask: np.ndarray, colour_src: np.ndarray | None) -> Region:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return Region(0, 0, 1, 1, 0.0)
    r = Region(xs.min() / GW, ys.min() / GH, (xs.max() + 1) / GW, (ys.max() + 1) / GH, float(mask.mean()))
    if colour_src is not None:
        r.colour = _colour(colour_src[mask].mean(axis=0))
    return r


def _direction(dx: float, dy: float) -> str:
    ang = np.degrees(np.arctan2(-dy, dx)) % 360          # screen y grows downward
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


def legs(path: list[tuple[float, float, float]], eps: float = 0.05) -> list[tuple[float, float, float, float, float, float]]:
    """Straight stretches of a motion path (Ramer-Douglas-Peucker): (t0, x0, y0, t1, x1, y1)."""
    if len(path) < 2:
        return []
    pts = np.array(path)

    def rdp(a: int, b: int) -> list[int]:
        if b <= a + 1:
            return [a, b]
        p, q = pts[a, 1:], pts[b, 1:]
        seg = q - p
        n = np.linalg.norm(seg)
        v = pts[a + 1:b, 1:] - p
        d = (np.abs(seg[0] * v[:, 1] - seg[1] * v[:, 0]) / n if n > 1e-9
             else np.linalg.norm(pts[a + 1:b, 1:] - p, axis=1))
        k = int(np.argmax(d))
        if d[k] > eps:
            return rdp(a, a + 1 + k)[:-1] + rdp(a + 1 + k, b)
        return [a, b]

    idx = rdp(0, len(pts) - 1)
    return [(pts[i, 0], pts[i, 1], pts[i, 2], pts[j, 0], pts[j, 1], pts[j, 2]) for i, j in zip(idx, idx[1:])
            if np.hypot(pts[j, 1] - pts[i, 1], pts[j, 2] - pts[i, 2]) > 0.03]


def leg_words(leg: tuple[float, float, float, float, float, float]) -> str:
    t0, x0, y0, t1, x1, y1 = leg
    return f"{_direction(x1 - x0, y1 - y0)} ({_where(x0, y0)} -> {_where(x1, y1)}, {t0:.1f}-{t1:.1f}s)"


def _where(x: float, y: float) -> str:
    if x < 0 or x > 1 or y < 0 or y > 1:
        return "edge"
    v = "top" if y < 0.34 else "bottom" if y > 0.66 else ""
    h = "left" if x < 0.34 else "right" if x > 0.66 else ""
    return f"{v}-{h}" if v and h else v or h or "center"


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a small boolean grid (BFS; the grid is 18 x 32)."""
    seen = np.zeros_like(mask, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(mask, bool)
        stack = [(y0, x0)]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp[y, x] = True
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < GH and 0 <= xx < GW and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append(comp)
    return out


def _dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
            if dy > 0: sh[:dy] = False
            if dy < 0: sh[dy:] = False
            if dx > 0: sh[:, :dx] = False
            if dx < 0: sh[:, dx:] = False
            out |= sh
    return out


@dataclass(eq=False)             # identity, not field comparison: tracks hold numpy arrays
class _Track:
    frames: dict = field(default_factory=dict)    # frame index -> changed mask of this track in that frame
    last: int = -1
    reach: np.ndarray | None = None               # dilated mask of the latest frames, to link the next ones


def cuts(g: Grid, context_s: float = 3.0) -> list[int]:
    """Frames where the whole picture changed at once: a spike in mean block change that is
    large and an outlier against the local baseline (so busy footage does not cut every frame),
    and not a flash (the picture after must differ from the one before)."""
    m = g.diff.astype(np.float32).mean(axis=(1, 2))
    n = len(m)
    if n < 3:
        return []
    dt = float(np.median(np.diff(g.times)))
    half = max(2, int(context_s / dt))
    pad = np.pad(m, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)[:n]
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1)
    out = []
    for i in range(1, n):
        if m[i] < 0.08 or m[i] < med[i] + 5 * (1.4826 * mad[i]) + 0.03:
            continue
        if i + 1 < n and m[i + 1] > m[i]:          # the spike's peak is the cut frame
            continue
        after = min(n - 1, i + max(1, int(0.3 / dt)))
        if np.abs(g.blocks[after].astype(np.float32) - g.blocks[i - 1].astype(np.float32)).max(axis=-1).mean() < 0.04:
            continue                               # changed and came back: a flash, not a cut
        if out and i - out[-1] < max(2, int(0.25 / dt)):
            continue
        out.append(i)
    return out


def _tracks(changed: np.ndarray, gap: int, split: set[int] = frozenset()) -> list[_Track]:
    """Space-time connected regions of change: two things changing at once in different
    places are two events; one thing changing on and off in place is one."""
    live: list[_Track] = []
    done: list[_Track] = []
    for k in range(len(changed)):
        if k in split:                         # nothing continues across a cut
            done += live
            live = []
            continue
        for t in [t for t in live if k - t.last > gap]:
            live.remove(t)
            done.append(t)
        if not changed[k].any():
            continue
        for own in _components(changed[k]):
            comp = _dilate(own, 1)
            size = int(own.sum())
            # Join a track only when this region mostly lies where the track was: something new
            # appearing next to something else (a toast under a spinner) is its own event.
            hits = [t for t in live if (t.reach & own).sum() >= max(1, 0.3 * size) or (size <= 4 and (t.reach & comp).any())]
            if not hits:
                t = _Track()
                live.append(t)
            else:
                t = hits[0]
                for o in hits[1:]:                 # one component bridges two tracks: merge them
                    for f, m in o.frames.items():
                        t.frames[f] = t.frames.get(f, np.zeros_like(m)) | m
                    t.reach |= o.reach
                    live.remove(o)
            t.frames[k] = t.frames.get(k, np.zeros_like(own)) | own
            t.reach = comp if t.last < k - gap // 2 or t.reach is None else (t.reach | comp)
            t.last = k
    return done + live


def detect(g: Grid) -> list[Event]:
    n = len(g.times)
    if n < 2:
        return []
    dt = float(np.median(np.diff(g.times)))
    changed = g.changed()
    strong = (g.diff.astype(np.float32) > 2 * g.thr) | (g.pix.astype(np.float32) > 2 * g.pthr)
    # A single weak block in a single frame is noise.
    changed &= (changed.sum(axis=(1, 2)) >= 2)[:, None, None] | strong
    changed[0] = False
    gap = max(1, int(round(MERGE_GAP_S / dt)))
    cut_frames = cuts(g)
    events: list[Event] = []
    for i in cut_frames:
        b, a = i - 1, min(n - 1, i + 1)
        events.append(Event(0, "cut", float(g.times[i]), float(g.times[i]), float(g.times[b]), float(g.times[a]),
                            float(g.times[a]), Region(0, 0, 1, 1, 1.0)))
    for tr in sorted(_tracks(changed, gap, set(cut_frames)), key=lambda t: min(t.frames)):
        ks = sorted(tr.frames)
        i0, i1 = ks[0], ks[-1]
        b, a = i0 - 1, min(n - 1, i1 + 1)
        union = np.zeros((GH, GW), bool)
        for m in tr.frames.values():
            union |= m
        zone = _dilate(union, 1)
        net = _differs(g, b, a) & zone
        dev = [float((_differs(g, b, k) & zone).sum()) for k in range(i0, a + 1)]
        pk = i0 + int(np.argmax(dev))
        dur = (i1 - i0 + 1) * dt
        u_area, n_area = float(union.mean()), float(net.mean())
        mean_area = float(np.mean([tr.frames[k].mean() for k in ks]))
        path = []
        ys, xs = np.mgrid[0:GH, 0:GW]
        for k in ks:
            w = np.where(tr.frames[k], g.diff[k].astype(np.float32) + g.pix[k].astype(np.float32), 0)
            if w.sum() > 0:
                path.append((float(g.times[k]), float((xs * w).sum() / w.sum() + 0.5) / GW,
                             float((ys * w).sum() / w.sum() + 0.5) / GH))
        travel = 0.0
        if len(path) >= 2:
            pp = np.array(path)[:, 1:]
            travel = float(np.hypot(*(pp.max(axis=0) - pp.min(axis=0))))

        if u_area > FULL and dur <= 3 * dt:
            kind = "cut" if n_area > 0.5 * u_area else "blink"
        elif mean_area > 0.5:
            kind = "scene"
        elif dur >= 0.3 and travel > 0.15 and mean_area < 0.3:
            kind = "motion"
        elif n_area < 0.3 * u_area and dur < 2.0:
            kind = "blink"
        elif dur >= 1.0 and n_area < 0.6 * u_area:
            kind = "animating"
        else:
            kind = "change"
        if kind == "cut" and any(abs(int(i0) - c) <= 2 for c in cut_frames):
            continue                               # already listed by cuts()
        mask = net if kind in ("change", "cut") and net.any() else union
        if kind == "motion":
            # The moving thing, not the background it crossed: blocks far from their own median over the event.
            stack = g.blocks[i0:a + 1].astype(np.float32)
            dev_px = np.abs(stack - np.median(stack, axis=0)).max(axis=-1)
            # Look at a few moments and keep the one where the moving thing stands out most.
            best = max((len(stack) * q // 4 for q in (1, 2, 3)), key=lambda k: int(((dev_px[k] > g.thr) & zone).sum()))
            obj = (dev_px[best] > g.thr) & zone
            reg = _region(obj if obj.any() else union, None)
            if obj.any():   # colour of the block that stands out most: edge blocks mix in the background
                yy, xx = np.unravel_index(np.argmax(np.where(obj, dev_px[best], -1)), obj.shape)
                reg.colour = _colour(stack[best][yy, xx])
            reg.x0, reg.y0, reg.x1, reg.y1 = _region(union, None).box()
        else:
            reg = _region(mask, g.blocks[pk if kind == "blink" else a].astype(np.float32))
        events.append(Event(0, kind, float(g.times[i0]), float(g.times[i1]), float(g.times[b]),
                            float(g.times[a]), float(g.times[pk]), reg, path if kind == "motion" else []))
    events.sort(key=lambda e: e.start)
    for k, e in enumerate(events):
        e.id = k + 1
    _link_reverts(g, events)
    return events


def _differs(g: Grid, i: int, j: int) -> np.ndarray:
    """Blocks whose content differs between frames i and j. Mean colour misses a
    changed character, so blocks are also compared by their brightness spread."""
    bi, bj = g.blocks[i].astype(np.float32), g.blocks[j].astype(np.float32)
    return np.abs(bi - bj).max(axis=-1) > min(g.thr, 0.03)


def _link_reverts(g: Grid, events: list[Event], horizon: float = 5.0) -> None:
    """A change that a later event undoes was only on screen for a while: say so on both."""
    idx = {round(float(t), 4): i for i, t in enumerate(g.times)}

    def frame(t: float) -> np.ndarray:
        return g.blocks[idx[round(t, 4)]].astype(np.float32)

    for i, e in enumerate(events):
        if e.kind != "change" or e.region.area >= FULL:
            continue
        x0, y0, x1, y1 = (int(round(e.region.x0 * GW)), int(round(e.region.y0 * GH)),
                          int(round(e.region.x1 * GW)), int(round(e.region.y1 * GH)))
        pre = frame(e.before)[y0:y1, x0:x1]
        post = frame(e.after)[y0:y1, x0:x1]
        if np.abs(post - pre).max() <= g.thr:
            continue
        for f in events[i + 1:]:
            if f.start - e.after > horizon:
                break
            if f.kind not in ("change", "cut"):
                continue
            now = frame(f.after)[y0:y1, x0:x1]
            if (np.abs(now - pre).max(axis=-1) > g.thr).mean() < 0.15:
                e.shown_until, f.undoes = f.start, e.id
                break


def group(events: list[Event], gap: float = 1.5) -> list[list[Event]]:
    """Runs of small changes in one place (typing, a counter, a lamp) as one line each.
    Neighbours join (typed characters sit side by side); the very same place joins over a longer gap."""
    out: list[list[Event]] = []
    for e in sorted(events, key=lambda x: x.start):
        target = None
        for grp in reversed(out[-6:]):
            last = grp[-1]
            if (e.kind != last.kind or e.kind not in ("change", "blink") or e.region.area >= 0.1
                    or e.undoes is not None or e.shown_until is not None or last.shown_until is not None):
                continue
            same = _overlap(e.region, last.region) > 0.7
            near = _overlap(_grow(e.region), _grow(last.region)) > 0.3
            if (same and e.start - last.end <= 4 * gap) or (near and e.start - last.end <= gap):
                target = grp
                break
        if target is None:
            out.append([e])
        else:
            target.append(e)
    return sorted(out, key=lambda grp: grp[0].start)


def _grow(r: Region, bx: float = 2 / GW, by: float = 1 / GH) -> Region:
    return Region(r.x0 - bx, r.y0 - by, r.x1 + bx, r.y1 + by, r.area)


def _overlap(a: Region, b: Region) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    small = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return inter / small if small > 0 else 0.0


def change_weighted_times(g: Grid, n: int, events: list[Event]) -> list[float]:
    """n moments spread evenly over how much the picture changed, not over the clock,
    plus the first frame and every cut: static stretches get one frame, busy ones many."""
    mass = g.diff.astype(np.float32).mean(axis=(1, 2))
    mass = np.minimum(mass, np.percentile(mass, 99) + 1e-6)  # one cut must not eat the whole budget
    # A floor keeps some clock coverage in long quiet stretches.
    mass = mass + 0.15 * mass.mean() + 1e-6
    cum = np.cumsum(mass)
    targets = (np.arange(n) + 0.5) / n * cum[-1]
    picks = {int(np.searchsorted(cum, t)) for t in targets}
    picks.add(0)
    cut_idx = sorted(int(np.searchsorted(g.times, e.after)) for e in events if e.kind == "cut")
    for a, b in zip([0] + cut_idx, cut_idx + [len(g.times)]):
        picks.add(min(a, len(g.times) - 1))                   # every shot shows up at least once,
        if b - a > 4:
            picks.add((a + b) // 2)                           # twice when it lasts
    for e in events:
        if e.kind == "blink" or e.shown_until is not None:
            picks.add(int(np.searchsorted(g.times, e.peak if e.kind == "blink" else e.after)))
    idx = sorted(min(i, len(g.times) - 1) for i in picks)
    out: list[float] = []
    last = -1
    for i in idx:
        t = float(g.times[i])
        if out and t - out[-1] <= 0.25:  # tiles closer than 0.25 s show the same moment twice
            continue
        # A frame identical to the last kept one (a still screen) adds nothing.
        if last >= 0 and not (np.abs(g.blocks[i].astype(np.float32) - g.blocks[last].astype(np.float32)).max(axis=-1)
                              > g.thr).any() and not (g.pix[last + 1:i + 1].astype(np.float32) > g.pthr).any():
            continue
        out.append(t)
        last = i
    return out


def shot_letters(g: Grid, cut_times: list[float], same: float = 0.06) -> list[tuple[float, str]]:
    """A letter per shot (start time, 'A'...), the same letter when a shot returns to a place seen
    before: questions like 'how many times does it cut back to X' need it in the text, not across
    many sheets. A shot is its colour layout pooled to 9 x 16 and averaged over its frames, which
    small camera moves barely change; shots closer than `same` (mean colour difference) match."""
    if len(g.times) < 2:
        return []
    bounds = [float(g.times[0])] + sorted(cut_times) + [float(g.times[-1]) + 1e-3]
    descs, starts = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (g.times >= a + 0.1) & (g.times < b - 0.05)          # skip the transition frames
        if not m.any():
            m = (g.times >= a) & (g.times < b)
        if not m.any():
            continue
        bl = g.blocks[m].astype(np.float32).mean(axis=0)          # (GH, GW, 3)
        descs.append(bl.reshape(GH // 2, 2, GW // 2, 2, 3).mean(axis=(1, 3)))
        starts.append(a)
    names: list[int] = []
    reps: list[np.ndarray] = []
    for d in descs:
        dist = [float(np.abs(d - r).mean()) for r in reps]
        j = int(np.argmin(dist)) if dist else -1
        if j >= 0 and dist[j] < same:
            names.append(j)
        else:
            reps.append(d)
            names.append(len(reps) - 1)
    return [(round(s, 2), _letters(n)) for s, n in zip(starts, names)]


def _letters(n: int) -> str:
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s
