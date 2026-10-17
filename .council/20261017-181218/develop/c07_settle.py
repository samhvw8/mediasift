"""C07 settling count: strings that sit in cached flow frames but never reach the card.

Reads only cached artifacts (frames/*_768.jpg, card.md, sensed.json); writes JSON to stdout.
Run: uv run python .council/20261017-181218/develop/c07_settle.py
"""
import glob, json, os, re, sys, time
from pathlib import Path

from mediasift import ocr, see

ROOT = Path.home() / ".cache/mediasift/see"
REAL = ["a77493a29bd4f083", "3d219a48b6b4f7e3", "8e7985c0ada69bff", "0d345810e8a16f89",
        "66f3c28341e309e0", "2f128a2426c870cd"]
SAMPLES = ["tiktok_vi", "reel_en", "python_100s", "cs50_lecture", "tears_of_steel", "big_buck_bunny"]

norm = lambda s: re.sub(r"\W+", "", s.lower())
words = lambda s: set(re.findall(r"\w{3,}", s.lower()))


def on_card(text: str, card_n: str, card_w: set) -> bool:
    n = norm(text)
    if len(n) >= 4 and n in card_n:
        return True
    w = words(text)
    return bool(w) and len(w & card_w) / len(w) >= 0.6


def run(d: Path, label: str) -> dict:
    card = (d / "card.md").read_text()
    js = json.loads((d / "sensed.json").read_text())
    speech = " ".join(c["text"] for c in js.get("speech", []))
    card_n, card_w, sp_w = norm(card), words(card), words(speech)
    frames = sorted(glob.glob(str(d / "frames" / "t*_768.jpg")))
    t0 = time.perf_counter()
    seen: dict[str, dict] = {}
    for f in frames:
        t = float(Path(f).name[1:11])
        for text, box in ocr.read(f):
            k = norm(text)
            if len(k) < 4:
                continue
            e = seen.setdefault(k, {"text": text, "times": [], "box": [round(v, 2) for v in box]})
            e["times"].append(t)
    secs = time.perf_counter() - t0
    dur = js.get("duration", 0)
    lasting = []
    for e in seen.values():
        ts_ = e["times"]
        if len(ts_) >= 3 and ts_[-1] - ts_[0] >= 5:
            w = words(e["text"])
            lasting.append({
                "text": e["text"], "from": round(ts_[0], 1), "to": round(ts_[-1], 1), "frames": len(ts_),
                "share": round(len(ts_) / max(len(frames), 1), 2), "box": e["box"],
                "on_card": on_card(e["text"], card_n, card_w),
                "is_speech": bool(w) and len(w & sp_w) / len(w) >= 0.6,
            })
    lasting.sort(key=lambda x: -x["frames"])
    missed = [x for x in lasting if not x["on_card"]]
    return {"video": label, "key": d.name, "content": js.get("content"), "duration": dur,
            "frames": len(frames), "ocr_s": round(secs, 1), "strings": len(seen),
            "lasting": len(lasting), "lasting_missed": len(missed),
            "missed_not_speech": len([x for x in missed if not x["is_speech"]]),
            "half_or_more_missed": len([x for x in missed if x["share"] >= 0.5]),
            "shots": len({l for _, l in js.get("shots", [])}), "cuts": len(js.get("shots", [])),
            "examples": missed[:12]}


def main() -> None:
    if not ocr.available():
        sys.exit("needs macOS Vision")
    dirs = [(ROOT / k, (ROOT / k / "source.txt").read_text().strip().split("/")[-1]) for k in REAL]
    for name in SAMPLES:
        p = Path("samples") / f"{name}.mp4"
        if p.exists():
            d = see.out_dir(p.resolve())
            if (d / "card.md").exists():
                dirs.append((d, name))
            else:
                print(f"no cached card for {name} ({d.name})", file=sys.stderr)
    out = [run(d, label) for d, label in dirs if (d / "frames").is_dir()]
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
