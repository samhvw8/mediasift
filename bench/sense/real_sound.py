"""Sound naming on real recordings: ESC-50 clips spliced at known times, scored without an LLM.

    mise run samples                               # includes 40 ESC-50 clips (20 classes x 2) in samples/esc50/
    uv run python bench/sense/real_sound.py        # build tracks, run `framesift see`, score

Each track is 10 real 5 s recordings with 1.5 s of silence between, over a still picture. A clip
counts as named when a non-silence sound line overlapping its span carries one of the class's
accepted words; a line overlapping no clip is a false line. Counts compare to the clips that
contain separate short sounds (knocks, claps, clicks) only loosely, so they are reported, not scored.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ESC = ROOT / "samples" / "esc50"
OUT = ROOT / "bench" / "sense" / "real_sound"
SR = 44100
ACCEPT = {
    "dog": ["dog", "bark", "bow-wow", "yip", "howl", "growl"],
    "rooster": ["rooster", "crow", "chicken", "fowl", "cock"],
    "door_wood_knock": ["knock", "door", "tap"],
    "clapping": ["clap", "applause"],
    "siren": ["siren"],
    "church_bells": ["bell", "chime"],
    "glass_breaking": ["glass", "shatter", "break"],
    "keyboard_typing": ["typing", "keyboard", "click"],
    "footsteps": ["footstep", "walk"],
    "laughing": ["laugh", "giggle", "chuckle", "snicker"],
    "crying_baby": ["cry", "baby", "infant", "sob", "whimper", "wail"],   # "wail, moan" is how a reader hears it
    "clock_alarm": ["alarm", "beep", "clock", "buzzer", "ring"],
    "car_horn": ["horn", "honk"],
    "thunderstorm": ["thunder", "storm", "rain"],
    "engine": ["engine", "motor", "vehicle", "idling", "car"],
    "mouse_click": ["click", "tick"],
    "coughing": ["cough"],
    "sneezing": ["sneeze"],
    "crackling_fire": ["fire", "crackl"],
    "cat": ["cat", "meow", "purr"],
}


def load(fn: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ESC / fn), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def build() -> list[tuple[Path, list[dict]]]:
    rows = [ln.split() for ln in (ESC / "pick.txt").read_text().splitlines() if ln.strip()]
    random.Random(3).shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = []
    for k in range(0, len(rows), 10):
        chunk = rows[k:k + 10]
        audio, key, t = [], [], 1.0
        audio.append(np.zeros(SR, np.float32))
        for cls, fn in chunk:
            y = load(fn)[: 5 * SR]
            audio.append(y)
            key.append({"class": cls, "file": fn, "start": round(t, 2), "end": round(t + len(y) / SR, 2)})
            t += len(y) / SR + 1.5
            audio.append(np.zeros(int(1.5 * SR), np.float32))
        wav = OUT / f"track{k // 10 + 1}.f32"
        np.concatenate(audio).tofile(wav)
        mp4 = OUT / f"track{k // 10 + 1}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x202030:s=640x360:r=10",
                        "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(wav), "-shortest",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4)], check=True)
        wav.unlink()
        (OUT / f"track{k // 10 + 1}.key.json").write_text(json.dumps(key, indent=1))
        tracks.append((mp4, key))
    return tracks


def sensed(mp4: Path) -> list[dict]:
    r = subprocess.run([str(Path.home() / ".local/bin/framesift"), "see", str(mp4), "--force"], capture_output=True,
                       text=True, check=True)
    d = Path(r.stdout.strip().splitlines()[-1].split(" in ", 1)[1].rstrip(")"))
    sys.path.insert(0, str(ROOT / "src"))
    from framesift.see import _snd_name
    out = []
    for s in json.loads((d / "sensed.json").read_text())["sounds"]:
        if s["kind"] == "silence":
            continue
        a, b = (s["times"][0], s["times"][-1] + 0.2) if s["kind"] == "hits" and s["times"] else (s["start"], s["end"])
        out.append({"label": _snd_name(s["label"]), "start": a, "end": b, "n": len(s["times"]), "kind": s["kind"],
                    "times": s["times"]})
    return out


def main() -> None:
    tracks = build()
    named = total = false = 0
    per_class: dict[str, list[int]] = {}
    lines = []
    for mp4, key in tracks:
        snds = sensed(mp4)
        used = set()
        for clip in key:
            hits = []
            for i, s in enumerate(snds):
                inside = [t for t in s["times"] if clip["start"] - 0.2 <= t <= clip["end"] + 0.2] if s["kind"] == "hits" else []
                if (s["kind"] == "hits" and inside) or (s["kind"] != "hits" and min(s["end"], clip["end"]) - max(s["start"], clip["start"]) > 0.3):
                    hits.append((i, s["label"] if s["kind"] != "hits" else f"{s['label']} x{len(inside)}"))
            ok = any(any(w in lab for w in ACCEPT[clip["class"]]) for _, lab in hits)
            used |= {i for i, _ in hits}
            named += ok
            total += 1
            per_class.setdefault(clip["class"], []).append(int(ok))
            lines.append(f"{'+' if ok else '-'} {mp4.stem} {clip['start']:5.1f}s {clip['class']:16} -> "
                         f"{'; '.join(lab for _, lab in hits) or '(nothing)'}")
        false += sum(1 for i in range(len(snds)) if i not in used)
    print("\n".join(lines))
    print(f"\nnamed {named}/{total} clips ({100 * named / total:.0f}%), false sound lines {false}")
    print("per class:", ", ".join(f"{c} {sum(v)}/{len(v)}" for c, v in sorted(per_class.items())))


if __name__ == "__main__":
    main()
