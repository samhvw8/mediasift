"""Harmony and rhythm benchmark for mediasift.

Tests three capabilities against ground-truth audio and video cuts:
1. Harmony mode (`major`, `minor`, `drone`) across keys, registers, voicings, and chord progressions.
2. Harmonic / key shift detection (`harmony_arc`) within +/- 1.5 s on beds >= 6 s.
3. Rhythm tempo (BPM within +/- 4% or 2x/0.5x metrical level), pulse clarity (0 false BPMs on
   ambient/drone/speech), rhythm shifts, and visual cut-to-beat sync (`beat_sync`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mediasift import audio, sound

SR = 16000


def _note_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _rich_chord(midis: list[int], sec: float, decay: float = 1.8, amp: float = 0.22,
                detune: float = 0.0) -> np.ndarray:
    """Synthesize a chord with 4 harmonics per note so chroma is realistic, not pure sines."""
    t = np.arange(int(sec * SR), dtype=np.float32) / SR
    y = np.zeros_like(t)
    for m in midis:
        f0 = _note_hz(m) * (1.0 + detune)
        for h, w in ((1, 1.0), (2, 0.55), (3, 0.30), (4, 0.15)):
            if f0 * h < SR * 0.45:
                y += w * np.sin(2 * np.pi * (f0 * h) * t)
    y /= max(len(midis), 1) * 1.6
    env = np.exp(-decay * t) if decay > 0 else np.ones_like(t)
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
    return (amp * y * env * fade).astype(np.float32)


def _kick(sec: float = 0.14, amp: float = 0.55) -> np.ndarray:
    t = np.arange(int(sec * SR), dtype=np.float32) / SR
    f = 55.0 + 95.0 * np.exp(-t * 35.0)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (amp * np.sin(ph) * np.exp(-t * 18.0)).astype(np.float32)


def _hat(rng: np.random.Generator, sec: float = 0.04, amp: float = 0.15) -> np.ndarray:
    n = int(sec * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    hp = np.diff(noise, prepend=0.0)
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * hp * np.exp(-t * 65.0)).astype(np.float32)


def _place(track: np.ndarray, clip: np.ndarray, at_s: float) -> None:
    i = int(round(at_s * SR))
    if i < 0 or i >= len(track):
        return
    n = min(len(clip), len(track) - i)
    track[i:i + n] += clip[:n]


def make_progression(root_midi: int, mode: str, bpm: float, sec: float,
                     rng: np.random.Generator, with_drums: bool = True) -> np.ndarray:
    """Build a realistic diatonic chord progression in major, minor, or static single-note drone."""
    y = np.zeros(int(sec * SR), np.float32)
    beat_s = 60.0 / bpm
    if mode == "drone":
        # Sustained fifth or octave pedal with slow breathing modulation, no beat pulses
        t = np.arange(len(y), dtype=np.float32) / SR
        f0 = _note_hz(root_midi)
        mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.17 * t)
        y = 0.22 * mod * (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * (f0 * 2.0) * t))
        return y.astype(np.float32)

    if mode == "major":
        # I - V - vi - IV in semitone offsets from root
        prog = [
            [0, 4, 7, 12],
            [7, 11, 14, 19],
            [9, 12, 16, 21],
            [5, 9, 12, 17],
        ]
    elif mode == "minor":
        # i - VI - III - VII or i - iv - v - i in minor
        prog = [
            [0, 3, 7, 12],
            [8, 12, 15, 20],
            [5, 8, 12, 17],
            [7, 10, 14, 19],
        ]
    else:
        raise ValueError(mode)

    n_beats = int(sec / beat_s)
    for b in range(n_beats):
        t0 = b * beat_s
        ch = [root_midi + o for o in prog[(b // 2) % len(prog)]]
        _place(y, _rich_chord(ch, beat_s * 0.95, decay=1.4, amp=0.22), t0)
        if with_drums:
            _place(y, _kick(), t0)
            _place(y, _hat(rng), t0 + 0.5 * beat_s)
    return np.clip(y, -0.98, 0.98)


def run_bench() -> dict:
    rng = np.random.default_rng(42)

    # 1. Harmony mode bench: 18 clips (6 major keys, 6 minor keys, 6 drones/ambiguous)
    mode_cases: list[tuple[str, np.ndarray, str]] = []
    for k, root in enumerate([48, 50, 53, 55, 57, 60]):  # C3, D3, F3, G3, A3, C4
        mode_cases.append((f"major_root_{root}", make_progression(root, "major", 110 + 4 * k, 10.0, rng), "major"))
        mode_cases.append((f"minor_root_{root}", make_progression(root, "minor", 96 + 4 * k, 10.0, rng), "minor"))
        mode_cases.append((f"drone_root_{root}", make_progression(root, "drone", 90, 10.0, rng, with_drums=False), "drone"))

    mode_hits = 0
    for name, y, expected in mode_cases:
        hr = audio.analyze_harmony_rhythm(y, SR)
        pred = hr["mode"]
        ok = pred == expected
        mode_hits += int(ok)
    mode_acc = mode_hits / len(mode_cases)

    # 2. Harmonic shift bench: 10 clips (16 s each, shift at t_shift in 6.5..9.5 s)
    shift_cases = []
    for i in range(10):
        t_shift = 6.5 + 0.3 * i
        m1, m2 = ("major", "minor") if i % 2 == 0 else ("minor", "major")
        r1 = 48 + (i % 5) * 2
        r2 = r1 + (3 if m1 == "major" else 4)
        y1 = make_progression(r1, m1, 112.0, t_shift, rng)
        y2 = make_progression(r2, m2, 112.0, 16.0 - t_shift, rng)
        y = np.concatenate([y1, y2])
        shift_cases.append((t_shift, m1, m2, y))

    shift_hits = 0
    for t_shift, m1, m2, y in shift_cases:
        hr = audio.analyze_harmony_rhythm(y, SR)
        arcs = hr.get("harmony_arc", [])
        if len(arcs) >= 2 and arcs[0][2] == m1 and arcs[-1][2] == m2 and abs(arcs[1][0] - t_shift) <= 1.5:
            shift_hits += 1
    shift_recall = shift_hits / len(shift_cases)

    # 3. Rhythm BPM & false-BPM bench
    bpms = [72.0, 84.0, 96.0, 108.0, 120.0, 128.0, 140.0, 152.0, 165.0, 137.0]
    bpm_hits = 0
    for b in bpms:
        y = make_progression(52, "minor" if int(b) % 2 else "major", b, 12.0, rng, with_drums=True)
        hr = audio.analyze_harmony_rhythm(y, SR)
        est = hr["bpm"]
        if est is not None and any(abs(est - b * mult) / (b * mult) <= 0.04 for mult in (0.5, 1.0, 2.0)):
            bpm_hits += 1
    bpm_acc = bpm_hits / len(bpms)

    # Ambient / drone / speech-like controls: must report bpm=None (0 false BPMs)
    false_bpm = 0
    n_controls = 8
    for i in range(n_controls):
        if i < 4:
            y = make_progression(48 + i * 3, "drone", 100.0, 10.0, rng, with_drums=False)
        else:
            # Irregular speech-like bursts
            y = np.zeros(10 * SR, np.float32)
            t = 0.4
            while t < 9.2:
                dur = float(rng.uniform(0.18, 0.65))
                _place(y, _rich_chord([50 + int(rng.integers(0, 7))], dur, decay=3.0, amp=0.2), t)
                t += dur + float(rng.uniform(0.12, 0.75))
        hr = audio.analyze_harmony_rhythm(y, SR)
        if hr["bpm"] is not None:
            false_bpm += 1

    # 4. Cut-to-beat sync bench: 6 beat-synced cut lists vs 6 random cut lists
    sync_sep = 0
    for b in [96.0, 110.0, 120.0, 128.0, 140.0, 150.0]:
        y = make_progression(55, "major", b, 14.0, rng, with_drums=True)
        hr = audio.analyze_harmony_rhythm(y, SR)
        beat_s = 60.0 / b
        synced_cuts = [round(k * 2 * beat_s + float(rng.uniform(-0.025, 0.025)), 3)
                       for k in range(2, int(13.0 / (2 * beat_s)))]
        unsynced_cuts = [round(k * 2 * beat_s + 0.37 * beat_s, 3)
                         for k in range(2, int(13.0 / (2 * beat_s)))]
        s_on = audio.cut_beat_sync(hr["beats"], synced_cuts)
        s_off = audio.cut_beat_sync(hr["beats"], unsynced_cuts)
        if s_on["synced"] and not s_off["synced"] and s_on["ratio"] >= 0.75 and s_off["ratio"] < 0.4:
            sync_sep += 1
    sync_acc = sync_sep / 6.0

    res = {
        "mode_acc": round(mode_acc, 3),
        "mode_hits": f"{mode_hits}/{len(mode_cases)}",
        "shift_recall": round(shift_recall, 3),
        "shift_hits": f"{shift_hits}/{len(shift_cases)}",
        "bpm_acc": round(bpm_acc, 3),
        "bpm_hits": f"{bpm_hits}/{len(bpms)}",
        "false_bpm": f"{false_bpm}/{n_controls}",
        "sync_acc": round(sync_acc, 3),
        "sync_hits": f"{sync_sep}/6",
    }
    print(
        f"harmony_rhythm bench: mode={res['mode_hits']} ({mode_acc:.1%}) | "
        f"shift={res['shift_hits']} ({shift_recall:.1%}) | "
        f"bpm={res['bpm_hits']} ({bpm_acc:.1%}) | "
        f"false_bpm={res['false_bpm']} | "
        f"cut_sync={res['sync_hits']} ({sync_acc:.1%})"
    )
    ok = (mode_acc >= 0.85 and shift_recall >= 0.80 and bpm_acc >= 0.90
          and false_bpm == 0 and sync_acc == 1.0)
    if not ok:
        sys.exit(1)
    return res


if __name__ == "__main__":
    run_bench()
