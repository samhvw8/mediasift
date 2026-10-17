"""Sound naming on real recordings: ESC-50 clips spliced at known times, scored without an LLM.

    mise run samples                               # ESC-50 clips in samples/esc50/
    uv run python bench/sense/real_sound.py        # the 40-clip set (20 classes x 2, folds 1-2)
    uv run python bench/sense/real_sound.py --set full                 # 200 clips, all 50 classes, folds 3-5 (held out)
    uv run python bench/sense/real_sound.py --set full --bed speech    # the same under a voice (LibriSpeech)
    uv run python bench/sense/real_sound.py --set full --bed music --snr 3   # under synthetic music, 3 dB above it
    uv run python bench/sense/real_sound.py --set full --bed song      # under a real song (CC BY)

Each track is 10 real 5 s recordings with 1.5 s of silence between, over a still picture. A clip
counts as named when a non-silence sound line overlapping its span carries one of the class's
accepted words; a line overlapping no clip is a false line. Counts compare to the clips that
contain separate short sounds (knocks, claps, clicks) only loosely, so they are reported, not scored.
"""
from __future__ import annotations

import argparse
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
    # the other 30 ESC-50 classes (--set full)
    "pig": ["pig", "oink"], "cow": ["cow", "moo", "cattle"], "frog": ["frog", "croak"],
    "hen": ["chicken", "hen", "cluck", "fowl"], "insects": ["insect", "buzz", "fly", "bee", "mosquito"],
    "sheep": ["sheep", "bleat", "goat"], "crow": ["crow", "caw", "bird"], "rain": ["rain", "raindrop", "drip"],
    "sea_waves": ["wave", "ocean", "surf", "water", "sea"], "crickets": ["cricket", "insect"],
    "chirping_birds": ["bird", "chirp", "tweet", "song"], "water_drops": ["drip", "drop", "water", "plop"],
    "wind": ["wind", "rustl", "howl", "noise"], "pouring_water": ["pour", "water", "liquid", "trickle", "fill"],
    "toilet_flush": ["toilet", "flush", "water"], "breathing": ["breath", "gasp", "pant", "sigh", "wheeze"],
    "brushing_teeth": ["brush", "teeth", "scrap", "rub"], "snoring": ["snor"],
    "drinking_sipping": ["drink", "sip", "gulp", "swallow"], "door_wood_creaks": ["creak", "squeak", "door"],
    "can_opening": ["can", "open", "fizz", "hiss", "pop", "crack"], "washing_machine": ["washing", "machine", "mechanic", "hum"],
    "vacuum_cleaner": ["vacuum", "hum", "motor"], "clock_tick": ["tick", "clock", "click"],
    "helicopter": ["helicopter", "aircraft", "rotor"], "chainsaw": ["chainsaw", "saw", "motor", "engine"],
    "train": ["train", "rail", "locomotive", "horn", "whistle"], "airplane": ["airplane", "aircraft", "jet", "engine", "plane"],
    "fireworks": ["firework", "explosion", "bang", "pop", "burst", "gunshot", "boom"], "hand_saw": ["saw", "sawing", "scrap", "rasp"],
}
BEDS = ("music", "singing", "speech")      # what a bed itself reads as; not a sound line to score


def load(fn: str) -> np.ndarray:
    # An absolute fn (a LibriSpeech file for the speech bed) ignores ESC.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ESC / fn), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def speech_bed(n: int) -> np.ndarray:
    files = sorted((ROOT / "samples" / "librispeech" / "LibriSpeech" / "dev-clean").rglob("*.flac"))
    random.Random(5).shuffle(files)
    out, have = [], 0
    for f in files:
        y = load(str(f.resolve()))
        out.append(y)
        have += len(y)
        if have >= n:
            break
    return np.concatenate(out)[:n]


def music_bed(n: int) -> np.ndarray:
    """Plain synthetic chords and a soft beat: reads as music, holds no animal or effect sounds."""
    t = np.arange(n) / SR
    roots = [220.0, 174.6, 196.0, 146.8]                  # A F G D, 2 s each
    y = np.zeros(n, np.float32)
    for i, r in enumerate(roots * (n // (2 * SR * 4) + 1)):
        a, b = i * 2 * SR, min(n, (i + 1) * 2 * SR)
        if a >= n:
            break
        tt = t[a:b] - t[a]
        env = np.minimum(1, tt * 20) * np.exp(-tt * 0.8)
        for f in (r, r * 1.26, r * 1.5, r * 2):
            y[a:b] += (np.sin(2 * np.pi * f * tt) * env * 0.2).astype(np.float32)
    for k in range(0, n, SR // 2):                         # a kick every half second
        m = min(n - k, int(0.12 * SR))
        tt = np.arange(m) / SR
        y[k:k + m] += (np.sin(2 * np.pi * 60 * tt) * np.exp(-tt * 30) * 0.5).astype(np.float32)
    return y


def song_bed(n: int) -> np.ndarray:
    """A real CC BY song (Kevin MacLeod, fetched by samples/fetch.sh), looped: real music is where a
    looser rule would invent sounds, and synthetic chords are too easy to tell apart."""
    y = load(str((ROOT / "samples" / "music" / "LrM_Y39Gmhk.wav").resolve()))
    return np.tile(y, n // len(y) + 1)[:n]


def rms(y: np.ndarray) -> float:
    loud = y[np.abs(y) > 0.1 * np.abs(y).max()] if len(y) and np.abs(y).max() > 0 else y
    return float(np.sqrt((loud ** 2).mean())) if len(loud) else 0.0


def build(pick: str, bed: str, snr: float) -> list[tuple[Path, list[dict]]]:
    rows = [ln.split() for ln in (ESC / pick).read_text().splitlines() if ln.strip()]
    random.Random(3).shuffle(rows)
    out = OUT / (Path(pick).stem + ("" if bed == "none" else f"_{bed}{snr:g}"))
    out.mkdir(parents=True, exist_ok=True)
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
        mix = np.concatenate(audio)
        if bed != "none":
            b = {"speech": speech_bed, "music": music_bed, "song": song_bed}[bed](len(mix))
            clip_level = np.median([rms(load(fn)[: 5 * SR]) for _, fn in chunk])
            mix = mix + b * (clip_level / (rms(b) + 1e-9) / 10 ** (snr / 20))
            mix = mix / max(1.0, float(np.abs(mix).max()) / 0.98)
        wav = out / f"track{k // 10 + 1}.f32"
        mix.astype(np.float32).tofile(wav)
        mp4 = out / f"track{k // 10 + 1}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x202030:s=640x360:r=10",
                        "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(wav), "-shortest",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4)], check=True)
        wav.unlink()
        (out / f"track{k // 10 + 1}.key.json").write_text(json.dumps(key, indent=1))
        tracks.append((mp4, key))
    return tracks


def sensed(mp4: Path) -> list[dict]:
    r = subprocess.run([str(Path.home() / ".local/bin/mediasift"), "see", str(mp4), "--force"], capture_output=True,
                       text=True, check=True)
    d = Path(r.stdout.strip().splitlines()[-1].split(" in ", 1)[1].rstrip(")"))
    sys.path.insert(0, str(ROOT / "src"))
    from mediasift.see import _snd_name
    out = []
    for s in json.loads((d / "sensed.json").read_text())["sounds"]:
        if s["kind"] == "silence":
            continue
        a, b = (s["times"][0], s["times"][-1] + 0.2) if s["kind"] == "hits" and s["times"] else (s["start"], s["end"])
        out.append({"label": _snd_name(s["label"]), "start": a, "end": b, "n": len(s["times"]), "kind": s["kind"],
                    "times": s["times"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["pick", "full"], default="pick")
    ap.add_argument("--bed", choices=["none", "speech", "music", "song"], default="none")
    ap.add_argument("--snr", type=float, default=6.0, help="dB the sounds sit above the bed")
    a = ap.parse_args()
    tracks = build(f"{a.set}.txt", a.bed, a.snr)
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
        false += sum(1 for i in range(len(snds)) if i not in used and snds[i]["label"] not in BEDS)
    print("\n".join(lines))
    print(f"\nnamed {named}/{total} clips ({100 * named / total:.0f}%), false sound lines {false}")
    print("per class:", ", ".join(f"{c} {sum(v)}/{len(v)}" for c, v in sorted(per_class.items())))
    res = OUT / f"result_{a.set}_{a.bed}{'' if a.bed == 'none' else f'{a.snr:g}'}.json"
    res.write_text(json.dumps({"named": named, "total": total, "false": false, "lines": lines,
                               "per_class": {c: v for c, v in sorted(per_class.items())}}, indent=1))
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
