"""On-screen text via Apple Vision on macOS, falling back to Indra OCR (ONNX Runtime CPU) on Linux.

Text that appears or disappears is the cheapest possible evidence for an agent:
a line of characters instead of an image to read. On macOS, Apple Vision runs natively
(multilingual incl. Vietnamese). On Linux (or when `MEDIASIFT_OCR=indra`), Indra OCR
runs a 1M-parameter line detector + CTC recognizer (~18 MB of ONNX models, no PyTorch).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LANGS = ["en-US", "vi-VT"]
INDRA_REPO = "datalab-to/indra-ocr"
INDRA_REV = "v3.1.1"
INDRA_FILES = ("det.onnx", "rec.onnx", "order.onnx", "meta.json")

_INDRA_ENGINE = None


def _log(msg: str) -> None:
    print(f"mediasift: {msg}", file=sys.stderr)


def vision_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        return True
    except ImportError:
        return False


def indra_available() -> bool:
    try:
        import indra_ocr  # noqa: F401
        return True
    except ImportError:
        return False


def backend(preferred: str = "auto") -> str | None:
    pref = (os.environ.get("MEDIASIFT_OCR", "auto") if preferred == "auto" else preferred).lower()
    if pref == "vision":
        return "vision" if vision_available() else None
    if pref == "indra":
        return "indra" if indra_available() else None
    if vision_available():
        return "vision"
    if indra_available():
        return "indra"
    return None


def available(preferred: str = "auto") -> bool:
    return backend(preferred) is not None


def _indra_model_dir() -> Path:
    from . import sound
    dest = sound.cache_root() / "models" / f"indra-ocr-{INDRA_REV}"
    if all((dest / f).exists() for f in INDRA_FILES):
        return dest
    import indra_ocr
    try:
        src = Path(indra_ocr.model_dir(revision=INDRA_REV))
    except Exception:
        from huggingface_hub import snapshot_download
        _log(f"downloading the Indra OCR model (18 MB) (once) from huggingface.co/{INDRA_REPO} ...")
        src = Path(snapshot_download(INDRA_REPO, revision=INDRA_REV, allow_patterns=["*.onnx", "meta.json", "config.json"]))
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not (dest / f.name).exists():
            shutil.copy2(f, dest / f.name)
    return dest


def _indra_engine():
    global _INDRA_ENGINE
    if _INDRA_ENGINE is None:
        from indra_ocr.pipeline import OrtOCR
        from . import polite
        _INDRA_ENGINE = OrtOCR(_indra_model_dir(), threads=polite.threads(4), furniture=True, math=False)
    return _INDRA_ENGINE


def _read_vision(path: str | Path, box: tuple[float, float, float, float] | None = None) -> list[tuple[str, tuple[float, float, float, float]]]:
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


def _read_indra(path: str | Path, box: tuple[float, float, float, float] | None = None) -> list[tuple[str, tuple[float, float, float, float]]]:
    im = Image.open(path).convert("L")
    fw, fh = im.size
    if fw < 8 or fh < 8:
        return []
    if box is not None:
        bx0, by0, bx1, by1 = box
        px0 = int(np.clip(round(bx0 * fw), 0, fw))
        py0 = int(np.clip(round(by0 * fh), 0, fh))
        px1 = int(np.clip(round(bx1 * fw), 0, fw))
        py1 = int(np.clip(round(by1 * fh), 0, fh))
        if px1 - px0 < 8 or py1 - py0 < 8:
            return []
        gray = np.asarray(im.crop((px0, py0, px1, py1)), dtype=np.uint8)
    else:
        px0, py0 = 0, 0
        gray = np.asarray(im, dtype=np.uint8)
    eng = _indra_engine()
    boxes = eng.detect(gray)
    if not len(boxes):
        return []
    rec = eng.recognize(gray, boxes)
    page = eng.assemble((gray.shape[1], gray.shape[0]), boxes, rec, {})
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    for ln in page.lines:
        txt = ln.text.strip()
        if not txt or ln.conf < 0.6:
            continue
        lx0, ly0, lx1, ly1 = ln.box[:4]
        nx0 = float(np.clip((px0 + lx0) / fw, 0.0, 1.0))
        ny0 = float(np.clip((py0 + ly0) / fh, 0.0, 1.0))
        nx1 = float(np.clip((px0 + lx1) / fw, 0.0, 1.0))
        ny1 = float(np.clip((py0 + ly1) / fh, 0.0, 1.0))
        out.append((txt, (nx0, ny0, nx1, ny1)))
    return out


def read(path: str | Path, box: tuple[float, float, float, float] | None = None,
         preferred: str = "auto") -> list[tuple[str, tuple[float, float, float, float]]]:
    """Lines of text with their normalised (x0, y0, x1, y1) boxes, top-left origin.
    Uses Apple Vision on macOS and falls back to Indra OCR on Linux (or when `preferred='indra'`)."""
    b = backend(preferred)
    if b == "vision":
        return _read_vision(path, box)
    if b == "indra":
        return _read_indra(path, box)
    raise RuntimeError("No OCR backend available (install pyobjc-framework-vision on macOS or indra-ocr on Linux)")

def diff(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    """Lines added and removed, ignoring order and OCR's spacing noise."""
    norm = lambda s: "".join(s.lower().split())
    b, a = {norm(x): x for x in before}, {norm(x): x for x in after}
    return [a[k] for k in a if k not in b], [b[k] for k in b if k not in a]
