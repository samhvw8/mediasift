"""On-screen text via Apple's Vision framework (macOS only, multilingual incl. Vietnamese).

Text that appears or disappears is the cheapest possible evidence for an agent:
a line of characters instead of an image to read. The first call on a machine
compiles Vision's model (about a minute, once); after that a process pays about
0.7 s to load and 0.1 s per frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

LANGS = ["en-US", "vi-VT"]


def available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        return True
    except ImportError:
        return False


def read(path: str | Path, box: tuple[float, float, float, float] | None = None) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Lines of text with their normalised (x0, y0, x1, y1) boxes, top-left origin.
    With `box`, Vision reads only that region (faster); boxes are still frame-relative."""
    import Vision
    from Foundation import NSURL
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(LANGS)
    req.setUsesLanguageCorrection_(True)
    if box is not None:
        # Read only the region (Vision's rectangle has a bottom-left origin): much faster than the frame.
        import Quartz
        x0, y0, x1, y1 = box
        req.setRegionOfInterest_(Quartz.CGRectMake(x0, 1 - y1, x1 - x0, y1 - y0))
    h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None)
    ok, err = h.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed on {path}: {err}")
    out = []
    for r in req.results() or []:
        c = r.topCandidates_(1)[0]
        if c.confidence() < 0.4:
            continue
        b = r.boundingBox()          # bottom-left origin, relative to the region when one is set
        x0, y1 = b.origin.x, 1 - b.origin.y
        x1, y0 = x0 + b.size.width, y1 - b.size.height
        if box is not None:
            bw, bh = box[2] - box[0], box[3] - box[1]
            x0, x1 = box[0] + x0 * bw, box[0] + x1 * bw
            y0, y1 = box[1] + y0 * bh, box[1] + y1 * bh
        out.append((str(c.string()).strip(), (x0, y0, x1, y1)))
    return [o for o in out if o[0]]


def diff(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    """Lines added and removed, ignoring order and OCR's spacing noise."""
    norm = lambda s: "".join(s.lower().split())
    b, a = {norm(x): x for x in before}, {norm(x): x for x in after}
    return [a[k] for k in a if k not in b], [b[k] for k in b if k not in a]
