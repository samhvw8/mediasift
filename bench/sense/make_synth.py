"""Synthetic videos with exact ground truth for the senses keyframes + transcript miss.

Each clip targets one gap named in the brief: transient events, order of steps,
motion direction, cause and effect, counting, non-speech sound, speaker changes,
and pointing at something while saying "this one". The script writes
`<out>/<clip>.mp4` and `<out>/<clip>.key.json` (questions, answers, evidence spans).

    uv run python bench/sense/make_synth.py bench/sense/clips

Speech comes from macOS `say`; everything else is drawn with PIL and numpy.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 44100
FPS = 30
FONT = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"


def font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO if mono else FONT, size)


# ---------------------------------------------------------------- audio helpers

def silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), np.float32)


def place(track: np.ndarray, clip: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    n = min(len(clip), len(track) - i)
    track[i:i + n] += gain * clip[:n]


def tone(freq: float | list[float], sec: float, decay: float = 0.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    freqs = freq if isinstance(freq, list) else [freq]
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    env = np.exp(-decay * t) if decay else np.ones_like(t)
    fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.005)  # 5 ms ramps, no clicks
    return (amp * y * env * fade).astype(np.float32)


def key_click(rng: np.random.Generator) -> np.ndarray:
    n = int(0.018 * SR)
    y = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.003 * SR))
    return (0.25 * np.diff(y, prepend=0)).astype(np.float32)  # high-passed burst


def siren(sec: float) -> np.ndarray:
    """A wailing siren: 650-1500 Hz sweep, 2.5 s period, with harmonics."""
    t = np.arange(int(sec * SR)) / SR
    f = 1075 + 425 * np.sin(2 * np.pi * t / 2.5)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    env = np.minimum(1, t / 0.3) * np.minimum(1, (sec - t) / 0.5)
    return (0.18 * y * env).astype(np.float32)


def music(sec: float) -> np.ndarray:
    """C - G - Am - F, plucked triads with a kick on each beat."""
    chords = [[261.6, 329.6, 392.0], [196.0, 246.9, 293.7], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6]]
    y = silence(sec)
    beat = 0.4375
    for i in range(int(sec / beat)):
        place(y, tone(chords[(i // 4) % 4], beat, decay=4, amp=0.22), i * beat)
        place(y, tone(60, 0.15, decay=25, amp=0.5), i * beat)
    return y


def phone_ring(sec: float) -> np.ndarray:
    """North American ring: 440 + 480 Hz, warbled at 20 Hz."""
    t = np.arange(int(sec * SR)) / SR
    y = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 20 * t)))
    return (0.15 * y).astype(np.float32)


def thud() -> np.ndarray:
    t = np.arange(int(0.35 * SR)) / SR
    return (0.8 * np.sin(2 * np.pi * (90 - 40 * t) * t) * np.exp(-t * 14)).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def speech(text: str, voice: str, tmp: Path) -> np.ndarray:
    f = tmp / f"say_{abs(hash((text, voice)))}.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(f), text], check=True)
    return load_audio(f) * 0.9


# ---------------------------------------------------------------- video writer

def write(out: Path, frames, n_frames: int, size: tuple[int, int], audio: np.ndarray, tmp: Path) -> None:
    wav = tmp / "a.f32"
    np.clip(audio, -1, 1).astype(np.float32).tofile(wav)
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(wav),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "60", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n_frames):
        p.stdin.write(np.asarray(frames(i / FPS), np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}")


def lerp(a: float, b: float, t0: float, t1: float, t: float) -> float:
    if t <= t0:
        return a
    if t >= t1:
        return b
    u = (t - t0) / (t1 - t0)
    u = u * u * (3 - 2 * u)  # ease in/out
    return a + (b - a) * u


def cursor(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    d.polygon([(x, y), (x, y + 22), (x + 6, y + 17), (x + 11, y + 27), (x + 15, y + 25), (x + 10, y + 15), (x + 17, y + 15)],
              fill="black", outline="white")


# ---------------------------------------------------------------- clip 1: UI flow (screen)

def clip_ui(tmp: Path, rng: np.random.Generator):
    W, H, DUR = 960, 540, 32.0
    email = "sam@acme.io"
    ev: dict = {"keys": []}

    def typing(text_len: int, t0: float, t1: float, t: float) -> int:
        if t < t0:
            return 0
        return min(text_len, int((t - t0) / (t1 - t0) * text_len) + 1) if t < t1 else text_len

    # timeline
    E1, P1, CLICK1, ERR0, ERR1 = (1.0, 5.0), (6.0, 7.5), 9.6, 11.8, 12.5
    E2, P2, CLICK2, OK = (14.0, 17.5), (18.5, 20.0), 21.5, 23.0
    SIDE = [(25.5, "Invoices"), (26.5, "Customers"), (27.5, "Reports")]
    PICK = 29.5
    for (a, b), n in [(E1, len(email)), (P1, 8), (E2, len(email)), (P2, 8)]:
        ev["keys"] += [a + (b - a) * k / n for k in range(n)]

    f_title, f_lbl, f_in, f_btn = font(22), font(18), font(20, mono=True), font(20)

    def frame(t: float):
        im = Image.new("RGB", (W, H), (236, 239, 244))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 44], fill=(40, 44, 52))
        d.text((20, 10), "Acme Billing - Sign in", font=f_title, fill="white")
        ox = 200 if t >= SIDE[0][0] else 0
        # sidebar builds up one item at a time
        if t >= SIDE[0][0]:
            d.rectangle([0, 44, 180, H], fill=(52, 58, 70))
            for k, (ts, name) in enumerate(SIDE):
                if t >= ts:
                    y = 80 + k * 50
                    if t >= PICK and name == "Reports":
                        d.rectangle([8, y - 8, 172, y + 30], fill=(80, 140, 230))
                    d.text((24, y), name, font=f_lbl, fill="white")
        x0 = 300 + ox // 2
        cleared = ERR1 <= t < E2[0]
        first = t < ERR1
        e_txt = email[:typing(len(email), *E1, t)] if first else ("" if cleared else email[:typing(len(email), *E2, t)])
        p_txt = "•" * (typing(8, *P1, t) if first else (0 if cleared else typing(8, *P2, t)))
        for k, (lbl, val) in enumerate([("Email", e_txt), ("Password", p_txt)]):
            y = 120 + k * 90
            d.text((x0, y), lbl, font=f_lbl, fill=(60, 60, 60))
            d.rectangle([x0, y + 26, x0 + 360, y + 62], fill="white", outline=(150, 150, 150))
            d.text((x0 + 10, y + 32), val, font=f_in, fill="black")
        label = "Submit" if t < ERR1 else "Retry"
        pressed = CLICK1 <= t < CLICK1 + 0.2 or CLICK2 <= t < CLICK2 + 0.2
        d.rectangle([x0, 310, x0 + 140, 352], fill=(30, 90, 200) if pressed else (60, 130, 240))
        d.text((x0 + 30, 320), label, font=f_btn, fill="white")
        if CLICK1 + 0.2 <= t < ERR0 or CLICK2 + 0.2 <= t < OK:
            a = t * 6
            cx, cy = x0 + 170, 331
            for j in range(8):
                g = 230 - j * 25
                d.ellipse([cx + 12 * math.cos(a + j * 0.785) - 3, cy + 12 * math.sin(a + j * 0.785) - 3,
                           cx + 12 * math.cos(a + j * 0.785) + 3, cy + 12 * math.sin(a + j * 0.785) + 3], fill=(g, g, g))
            d.text((x0 + 195, 320), "Signing in...", font=f_lbl, fill=(80, 80, 80))
        if ERR0 <= t < ERR1:
            d.rectangle([x0 - 20, 380, x0 + 420, 430], fill=(200, 40, 40))
            d.text((x0, 392), "Error 503: upstream timeout", font=f_btn, fill="white")
        if t >= OK:
            d.rectangle([x0 - 20, 380, x0 + 420, 430], fill=(40, 150, 70))
            d.text((x0, 392), "Welcome back, Sam", font=f_btn, fill="white")
        # mouse path
        path = [(0, 800, 450), (5.2, 800, 450), (5.8, x0 + 300, 260), (8.0, x0 + 300, 260), (9.4, x0 + 70, 330),
                (12.8, x0 + 70, 330), (13.6, x0 + 300, 170), (18.0, x0 + 300, 170), (18.3, x0 + 300, 260),
                (20.4, x0 + 300, 260), (21.3, x0 + 70, 330), (28.0, x0 + 70, 330), (29.3, 60, 236), (DUR, 60, 236)]
        for (ta, xa, ya), (tb, xb, yb) in zip(path, path[1:]):
            if ta <= t <= tb:
                cursor(d, lerp(xa, xb, ta, tb, t), lerp(ya, yb, ta, tb, t))
                break
        return im

    audio = silence(DUR)
    for k in ev["keys"]:
        place(audio, key_click(rng), k)
    for c in (CLICK1, CLICK2, PICK):
        place(audio, tone(2500, 0.03, decay=80, amp=0.3), c)
    place(audio, tone([110, 116], 0.6, decay=5, amp=0.6), ERR0)          # low dull "bonk"
    place(audio, tone([1318.5, 1760], 0.8, decay=4, amp=0.3), OK)        # bright two-note chime
    place(audio, tone(1760, 0.6, decay=5, amp=0.3), OK + 0.15)

    q = [
        ("transient", "An error message appeared briefly on screen. What exactly did it say?",
         "Error 503: upstream timeout", ["Error 503: upstream timeout", "503 upstream timeout", "error 503 upstream timeout"], [[ERR0, ERR1]]),
        ("sequence", "What did the app show between the first click on the button and the error?",
         "a spinner with 'Signing in...'", ["spinner", "Signing in", "loading"], [[CLICK1, ERR0]]),
        ("count", "How many times did the user click the blue button under the form?", "2", ["2", "two", "twice"], [[CLICK1, CLICK1 + 0.3], [CLICK2, CLICK2 + 0.3]]),
        ("state", "Right after the error, what happened to the text in the Email and Password fields?",
         "both fields were cleared (empty)", ["cleared", "empty", "erased", "reset"], [[ERR1, E2[0]]]),
        ("state", "What did the blue button's label say when it was clicked the second time?", "Retry", ["Retry"], [[ERR1, CLICK2 + 0.2]]),
        ("audio", "Describe the sound that played when the error appeared, compared with the sound when sign-in succeeded.",
         "a low, dull thud/bonk for the error; a high bright chime for success", ["low", "dull", "deep"], [[ERR0, ERR0 + 0.6], [OK, OK + 0.9]]),
        ("sequence", "In what order did the items appear in the left sidebar?", "Invoices, Customers, Reports",
         ["Invoices, Customers, Reports"], [[SIDE[0][0], SIDE[2][0] + 0.5]]),
        ("sequence", "Which sidebar item did the user click at the end?", "Reports", ["Reports"], [[PICK, DUR]]),
    ]
    return "ui_flow", frame, DUR, (W, H), audio, q


# ---------------------------------------------------------------- clip 2: motion and collision (camera-like)

def clip_motion(tmp: Path, rng: np.random.Generator):
    W, H, DUR = 640, 360, 20.0
    bg = (rng.random((H, W, 3)) * 40 + 90).astype(np.float32)  # textured floor
    bg = np.asarray(Image.fromarray(bg.astype(np.uint8)).resize((W // 8, H // 8)).resize((W, H), Image.BILINEAR), np.float32)
    HIT = 12.5

    def red(t: float) -> tuple[float, float]:
        x = lerp(80, 320, 1.0, 5.0, t)
        y = lerp(260, 110, 6.0, 8.0, t)
        if t > HIT:
            x = 320 - (t - HIT) * 150  # knocked left, leaves the frame at ~15.2 s
        return x, y

    def blue(t: float) -> tuple[float, float] | None:
        if t < 10.0:
            return None
        return max(320 + 56, W + 30 - (t - 10.0) * 120), 110

    def frame(t: float):
        noise = rng.standard_normal((H, W, 1)) * 4  # sensor grain: no two frames identical
        im = Image.fromarray(np.clip(bg + noise, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        if t >= 17.0:
            a = int(lerp(0, 255, 17.0, 18.0, t))
            d.rectangle([520, 250, 600, 330], fill=(a, a, 0))
        x, y = red(t)
        d.ellipse([x - 28, y - 28, x + 28, y + 28], fill=(220, 40, 40))
        b = blue(t)
        if b:
            d.ellipse([b[0] - 28, b[1] - 28, b[0] + 28, b[1] + 28], fill=(40, 80, 220))
        return im

    audio = silence(DUR)
    place(audio, (rng.standard_normal(len(audio)) * 0.01).astype(np.float32), 0)  # room tone
    place(audio, thud(), HIT)
    q = [
        ("motion", "In which direction did the red ball move first?", "right (left to right)", ["right"], [[1.0, 5.0]]),
        ("motion", "After its first move and a pause, which direction did the red ball move next?", "up", ["up", "upward"], [[6.0, 8.0]]),
        ("cause", "Why did the red ball start moving left?", "the blue ball rolled in from the right and hit it",
         ["blue ball hit it", "collision with the blue ball", "struck by the blue ball"], [[10.0, HIT + 0.5]]),
        ("motion", "Where did the red ball end up?", "it left the frame on the left side", ["left the frame", "off screen to the left", "exited left"], [[HIT, 15.5]]),
        ("audio", "What sound was heard when the two balls met?", "a low thud", ["thud", "bump", "knock", "low impact"], [[HIT, HIT + 0.4]]),
        ("state", "What objects are visible in the last second of the video?", "the blue ball and a yellow square",
         ["blue ball and yellow square"], [[18.0, DUR]]),
    ]
    return "motion", frame, DUR, (W, H), audio, q


# ---------------------------------------------------------------- clip 3: counting and audio-visual sync

def clip_count(tmp: Path, rng: np.random.Generator):
    W, H, DUR = 640, 360, 12.0
    FLASH = [1.0, 2.1, 2.9, 4.6, 5.2, 7.0, 8.8, 10.3]
    BEEP = [1.5, 2.9, 4.0, 7.0, 9.5]
    f = font(20)

    def frame(t: float):
        im = Image.new("RGB", (W, H), (25, 25, 30))
        d = ImageDraw.Draw(im)
        on = any(s <= t < s + 0.15 for s in FLASH)
        d.ellipse([270, 130, 370, 230], fill=(255, 255, 220) if on else (70, 70, 60))
        d.text((250, 300), "signal lamp", font=f, fill=(150, 150, 150))
        return im

    audio = silence(DUR)
    for b in BEEP:
        place(audio, tone(1000, 0.12, amp=0.35), b)
    q = [
        ("count", "How many times did the lamp flash?", "8", ["8", "eight"], [[0, DUR]]),
        ("count", "How many beeps were heard?", "5", ["5", "five"], [[0, DUR]]),
        ("av-sync", "Which lamp flashes (1st, 2nd, ...) happened at the same moment as a beep?", "the 3rd and the 6th",
         ["3rd and 6th", "third and sixth"], [[2.9, 3.05], [7.0, 7.15]]),
    ]
    return "count", frame, DUR, (W, H), audio, q


# ---------------------------------------------------------------- clip 4: audio scene over a still picture

def clip_scene(tmp: Path, rng: np.random.Generator):
    W, H, DUR = 640, 360, 36.0
    f1, f2 = font(34), font(22)

    def frame(t: float):
        im = Image.new("RGB", (W, H), (20, 24, 48))
        d = ImageDraw.Draw(im)
        d.text((60, 120), "The Night Shift", font=f1, fill="white")
        d.text((60, 175), "Episode 3" if t < 29.0 else "Q & A", font=f2, fill=(180, 190, 255))
        return im

    audio = silence(DUR)
    place(audio, speech("Welcome back to the night shift. Tonight we have a special guest.", "Samantha", tmp), 0.5)
    place(audio, music(7.0), 4.0)
    place(audio, siren(4.0), 14.0)
    place(audio, speech("Thanks for having me. It is great to be here.", "Daniel", tmp), 18.5)
    place(audio, phone_ring(2.0), 23.0)
    place(audio, phone_ring(1.2), 26.0)
    place(audio, speech("Sorry about that. Let's start with your first question.", "Samantha", tmp), 29.0)
    q = [
        ("audio", "What kind of sound came right after the host's welcome?", "music", ["music", "a song", "chords"], [[4.0, 11.0]]),
        ("audio", "What happened between the music and the guest's first words?", "a few seconds of silence, then a siren",
         ["silence then a siren", "siren"], [[11.0, 18.5]]),
        ("audio", "How many different voices speak in the video?", "2", ["2", "two"], [[0.5, 33.0]]),
        ("audio", "What interrupted things after the guest spoke?", "a telephone ringing", ["phone ringing", "telephone", "ringing phone"], [[23.0, 27.2]]),
        ("count", "How many separate rings were there?", "2", ["2", "two"], [[23.0, 27.2]]),
        ("cause", "What is the host apologizing for at the end?", "the phone ringing / the interruption", ["phone", "ringing", "interruption"], [[23.0, 32.0]]),
    ]
    return "audio_scene", frame, DUR, (W, H), audio, q


# ---------------------------------------------------------------- clip 5: pointing while speaking

def clip_deixis(tmp: Path, rng: np.random.Generator):
    W, H, DUR = 800, 450, 18.0
    bars = {"Q1": 180, "Q2": 300, "Q3": 70, "Q4": 220}
    xs = {k: 140 + i * 160 for i, k in enumerate(bars)}
    base = 380
    f = font(22)
    dot_path = [(0, "Q1"), (1.0, "Q1"), (3.0, "Q4"), (6.0, "Q4"), (6.6, "Q3"), (10.0, "Q3"), (10.6, "Q2"), (14.0, "Q2"), (14.6, "Q1"), (DUR, "Q1")]

    def dot(t: float) -> tuple[float, float]:
        for (ta, a), (tb, b) in zip(dot_path, dot_path[1:]):
            if ta <= t <= tb:
                xa, ya = xs[a] + 30, base - bars[a] + 10
                xb, yb = xs[b] + 30, base - bars[b] + 10
                return lerp(xa, xb, ta, tb, t), lerp(ya, yb, ta, tb, t)
        return xs["Q1"] + 30, base - bars["Q1"] + 10

    def frame(t: float):
        im = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(im)
        d.text((40, 20), "Revenue by quarter", font=font(28), fill="black")
        for k, h in bars.items():
            d.rectangle([xs[k], base - h, xs[k] + 60, base], fill=(90, 120, 170))  # same colour: no label from colour
            d.text((xs[k] + 12, base + 10), k, font=f, fill="black")
        x, y = dot(t)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 0, 0))
        return im

    audio = silence(DUR)
    place(audio, speech("Here is the year at a glance.", "Daniel", tmp), 0.5)
    place(audio, speech("And this one is the problem.", "Daniel", tmp), 8.0)
    place(audio, speech("This one was our best quarter.", "Daniel", tmp), 11.5)
    q = [
        ("deixis", "When the speaker says 'this one is the problem', which quarter's bar is being pointed at?", "Q3", ["Q3"], [[8.0, 9.5]]),
        ("deixis", "Which quarter does the speaker call the best quarter?", "Q2", ["Q2"], [[11.5, 13.0]]),
        ("sequence", "List the bars the red pointer visited, in order.", "Q1, Q4, Q3, Q2, Q1", ["Q1, Q4, Q3, Q2, Q1", "Q1 Q4 Q3 Q2"], [[0, DUR]]),
    ]
    return "deixis", frame, DUR, (W, H), audio, q


CLIPS = [clip_ui, clip_motion, clip_count, clip_scene, clip_deixis]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for make in CLIPS:
            name, frame, dur, size, audio, qs = make(tmp, rng)
            write(out / f"{name}.mp4", frame, int(dur * FPS), size, audio, tmp)
            key = {"video": name, "duration": dur, "questions": [
                {"id": f"q{i + 1:02d}", "type": ty, "question": qq, "answer": a, "accept": acc, "evidence": ev}
                for i, (ty, qq, a, acc, ev) in enumerate(qs)]}
            (out / f"{name}.key.json").write_text(json.dumps(key, indent=1))
            print(f"{name}: {dur:.0f}s, {len(qs)} questions")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/sense/clips")
