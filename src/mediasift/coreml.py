"""Core ML models through pyobjc — shared via `siftasr.coreml`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from siftasr.coreml import _to_ml, _to_np, available, load, predict
except ModuleNotFoundError:
    _sift_src = Path(__file__).resolve().parents[4] / "libs" / "siftasr" / "src"
    if _sift_src.is_dir():
        sys.path.insert(0, str(_sift_src))
    from siftasr.coreml import _to_ml, _to_np, available, load, predict

__all__ = ["_to_ml", "_to_np", "available", "load", "predict"]
