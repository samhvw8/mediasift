"""Run without making the machine feel slow.

Measured on an M1 Pro 16 GB (bench/lagprobe.py, 2:23 video with speech):
- CPU scheduling was not what made the desktop lag: a full-speed run left a
  normal-priority 5 ms timer as punctual as idle (p99 2.6 ms).
- macOS's background QoS band made the run 5x slower (48 s -> 256 s) for no
  measured gain, so it is not used.
- The lag came from memory: MLX's 5.5 GB buffer cache pushed the Mac into swap
  (fixed in asr.py), and audio and picture passes overlapping their peaks.

So: a plain `nice` (the scheduler prefers the user's apps under contention,
without pinning work to the efficiency cores) and capped worker pools.
`MEDIASIFT_FULL_SPEED=1` skips both.
"""
from __future__ import annotations

import os

_on = False


def enter() -> bool:
    """Call once, before starting work. Returns True when the polite settings are on."""
    global _on
    if os.environ.get("MEDIASIFT_FULL_SPEED") == "1" or _on:
        return _on
    if hasattr(os, "nice"):
        os.nice(10)
    _on = True
    return True


def threads(full: int) -> int:
    """Worker count for a pool or a model: capped when being polite."""
    return min(full, 4) if _on else full


def active() -> bool:
    return _on
