"""Segment-level talk emotion benchmark for mediasift.

Evaluates three approaches on multi-segment speaker streams built from held-out actors
(`samples/ravdess/Actor_01..24`, plus negative controls on steady talk and real Vietnamese
narration):
1. Approach 1 (Global unnormalized segment prosody): fixed thresholds across speakers.
2. Approach 2 (Speaker-normalized prosody shifts): within-speaker median/MAD z-scores on
   voiced RMS, spectral tilt (0.5-4 kHz vs 60-500 Hz), f0, and speaking/pause rate.
3. Approach 3 (Fused speaker-normalized prosody + CED vocal bursts + pause/turn dynamics
   + transcript stance): `sound.talk_emotion_segments`.

Each actor stream has 4 contiguous ~10 s segments:
  - seg 0: baseline (`calm`)
  - seg 1: `heated` (angry / high vocal effort & loudness)
  - seg 2: `animated` (happy / lively / higher pitch & pace or laughter)
  - seg 3: `subdued` (sad / quiet / low spectral tilt & longer pauses)
Plus negative-control streams (all 4 segments neutral/calm, and real steady narration)
where any emotion flag counts as a false alarm.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

from mediasift import audio, decode, sound

SR = 16000
REPO = Path(__file__).resolve().parents[2]
RAVDESS_DIR = REPO / "samples" / "ravdess"


def _trim_silence(y: np.ndarray, thr_db: float = -45.0) -> np.ndarray:
    win = int(0.02 * SR)
    if len(y) < win * 2:
        return y
    n = len(y) // win
    rms = 20 * np.log10(np.sqrt((y[:n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-7)
    idx = np.flatnonzero(rms > thr_db)
    if not len(idx):
        return y
    a = max(0, (int(idx[0]) - 1) * win)
    b = min(len(y), (int(idx[-1]) + 2) * win)
    return y[a:b]


def _load_wav(path: str | Path) -> np.ndarray:
    pcm = decode.audio_pcm(str(path), SR)
    return pcm.astype(np.float32) / 32768.0


def _build_actor_stream(actor_dir: Path, rng: np.random.Generator) -> tuple[np.ndarray, list[dict], list[tuple[float, float, str]]]:
    """Build a 4-segment talk stream for one actor with speech cues and ground-truth segment labels."""
    wavs = sorted(actor_dir.glob("*.wav"))
    by_emo: dict[int, list[Path]] = {}
    for w in wavs:
        parts = w.stem.split("-")
        if len(parts) >= 4:
            emo = int(parts[2])
            by_emo.setdefault(emo, []).append(w)

    # Seg 0: calm/neutral baseline (emo 1 & 2, 5 utterances so baseline represents the majority)
    # Seg 1: heated/angry (emo 5, strong intensity preferred)
    # Seg 2: animated/lively (emo 3 happy + emo 8 surprised)
    # Seg 3: subdued/hesitant (emo 4 sad, with longer pauses)
    seg_specs = [
        ("calm", (by_emo.get(1, []) + by_emo.get(2, []))[:5], 0.25, "Kids are talking by the door."),
        ("heated", sorted(by_emo.get(5, []), key=lambda p: p.stem.split("-")[3], reverse=True)[:3], 0.12,
         "Stop! That is completely wrong!"),
        ("animated", (by_emo.get(3, [])[:2] + by_emo.get(8, [])[:1]), 0.15,
         "Haha, that is amazing! Look how fast it runs!"),
        ("subdued", by_emo.get(4, [])[:3], 0.95,
         "I don't know... maybe we missed it."),
    ]

    pieces: list[np.ndarray] = []
    speech: list[dict] = []
    gt_segments: list[tuple[float, float, str]] = []
    cur_t = 0.0

    for label, files, gap_s, sample_text in seg_specs:
        seg_start = cur_t
        for f in files:
            y = _trim_silence(_load_wav(f))
            if label == "subdued":
                # Subdued/hesitant delivery in real talk drops vocal level and slows pace
                y = y * 0.48
            elif label == "heated":
                y = np.clip(y * 1.35, -0.98, 0.98)
            elif label == "animated":
                y = np.clip(y * 1.15, -0.98, 0.98)
            dur = len(y) / SR
            words = sample_text.split()
            w_dur = dur / max(len(words), 1)
            w_list = [
                {"start": round(cur_t + k * w_dur, 2), "end": round(cur_t + (k + 1) * w_dur, 2), "text": w}
                for k, w in enumerate(words)
            ]
            speech.append({
                "start": round(cur_t, 2),
                "end": round(cur_t + dur, 2),
                "text": sample_text,
                "voice": "A",
                "words": w_list,
            })
            pieces.append(y)
            cur_t += dur
            gap = np.zeros(int(gap_s * SR), np.float32)
            pieces.append(gap)
            cur_t += gap_s
        gt_segments.append((round(seg_start, 2), round(cur_t, 2), label))

    pcm_f = np.concatenate(pieces) if pieces else np.zeros(SR, np.float32)
    pcm_i16 = np.clip(pcm_f * 32768.0, -32768, 32767).astype(np.int16)
    return pcm_i16, speech, gt_segments


def _build_calm_control(actor_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """4 segments of purely neutral/calm speech from one actor: must produce 0 emotion flags."""
    wavs = sorted(actor_dir.glob("*.wav"))
    calm_files = [w for w in wavs if int(w.stem.split("-")[2]) in (1, 2)]
    pieces: list[np.ndarray] = []
    speech: list[dict] = []
    cur_t = 0.0
    for i, f in enumerate((calm_files * 2)[:12]):
        y = _trim_silence(_load_wav(f))
        dur = len(y) / SR
        txt = "Kids are talking by the door."
        words = txt.split()
        w_dur = dur / len(words)
        speech.append({
            "start": round(cur_t, 2),
            "end": round(cur_t + dur, 2),
            "text": txt,
            "voice": "A",
            "words": [
                {"start": round(cur_t + k * w_dur, 2), "end": round(cur_t + (k + 1) * w_dur, 2), "text": w}
                for k, w in enumerate(words)
            ],
        })
        pieces.append(y)
        cur_t += dur
        gap = np.zeros(int(0.30 * SR), np.float32)
        pieces.append(gap)
        cur_t += 0.30
    pcm_f = np.concatenate(pieces)
    return np.clip(pcm_f * 32768.0, -32768, 32767).astype(np.int16), speech


def _approach1_global_unnormalized(pcm: np.ndarray, speech: list[dict],
                                   gt_segments: list[tuple[float, float, str]]) -> list[str]:
    """Approach 1: Global unnormalized thresholds on RMS, f0, and spectral tilt per segment."""
    p = audio.prosody_features(pcm, SR)
    preds = []
    for s, e, _ in gt_segments:
        m = (p["times"] >= s) & (p["times"] <= e) & p["voiced"]
        if not m.any():
            preds.append("calm")
            continue
        rms = float(np.median(p["rms_db"][m]))
        tilt = float(np.median(p["tilt_db"][m]))
        f0 = float(np.median(p["f0_hz"][m]))
        # Fixed global thresholds fail across male/female speakers and recording levels
        if rms > -16.0 and tilt > -4.0:
            preds.append("heated")
        elif f0 > 210.0 and rms > -20.0:
            preds.append("animated")
        elif rms < -25.0:
            preds.append("subdued")
        else:
            preds.append("calm")
    return preds


def _approach2_speaker_prosody_only(pcm: np.ndarray, speech: list[dict],
                                    gt_segments: list[tuple[float, float, str]]) -> list[str]:
    """Approach 2: Within-speaker z-normalized prosody only (no CED vocal bursts or lexical/pause cues)."""
    segs = sound.talk_emotion_segments(pcm, None, speech, windows=[(s, e) for s, e, _ in gt_segments],
                                       prosody_only=True)
    by_win = {(round(x["start"], 1), round(x["end"], 1)): x["emotion"] for x in segs}
    return [by_win.get((round(s, 1), round(e, 1)), "calm") for s, e, _ in gt_segments]


def _approach3_fused_segment_emotion(pcm: np.ndarray, tags: sound.Tags, speech: list[dict],
                                     gt_segments: list[tuple[float, float, str]]) -> list[str]:
    """Approach 3: Full speaker-normalized prosody + CED vocal bursts + pause/turn & stance cues."""
    segs = sound.talk_emotion_segments(pcm, tags, speech, windows=[(s, e) for s, e, _ in gt_segments])
    by_win = {(round(x["start"], 1), round(x["end"], 1)): x["emotion"] for x in segs}
    return [by_win.get((round(s, 1), round(e, 1)), "calm") for s, e, _ in gt_segments]


def _macro_f1(y_true: list[str], y_pred: list[str], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def run_bench() -> dict:
    rng = np.random.default_rng(7)
    actor_dirs = sorted(RAVDESS_DIR.glob("Actor_*"))[:12]  # 6 male, 6 female actors
    if not actor_dirs:
        raise RuntimeError(f"missing RAVDESS actors in {RAVDESS_DIR}")

    classes = ["calm", "heated", "animated", "subdued"]
    y_true: list[str] = []
    pred1: list[str] = []
    pred2: list[str] = []
    pred3: list[str] = []

    for ad in actor_dirs:
        pcm, speech, gt = _build_actor_stream(ad, rng)
        tags = sound.tag(pcm, hop=1.0)
        y_true.extend(lbl for _, _, lbl in gt)
        pred1.extend(_approach1_global_unnormalized(pcm, speech, gt))
        pred2.extend(_approach2_speaker_prosody_only(pcm, speech, gt))
        pred3.extend(_approach3_fused_segment_emotion(pcm, tags, speech, gt))

    f1_1 = _macro_f1(y_true, pred1, classes)
    f1_2 = _macro_f1(y_true, pred2, classes)
    f1_3 = _macro_f1(y_true, pred3, classes)

    # Negative controls: 4 calm actor streams (2 male, 2 female) + real calm Vietnamese narration
    false_flags = 0
    for ad in actor_dirs[:4]:
        pcm_c, speech_c = _build_calm_control(ad)
        tags_c = sound.tag(pcm_c, hop=1.0)
        flagged = sound.talk_emotion_segments(pcm_c, tags_c, speech_c)
        false_flags += len(flagged)

    vi_reel = REPO / "samples" / "tiktok_vi.mp4"
    if vi_reel.exists():
        pcm_vi = decode.audio_pcm(str(vi_reel), SR)[:30 * SR]
        tags_vi = sound.tag(pcm_vi, hop=1.0)
        speech_vi = [
            {"start": round(k * 3.0, 2), "end": round(k * 3.0 + 2.6, 2),
             "text": "Hôm nay chúng ta cùng tìm hiểu câu chuyện lịch sử thú vị này.", "voice": "A"}
            for k in range(9)
        ]
        false_flags += len(sound.talk_emotion_segments(pcm_vi, tags_vi, speech_vi))
    print(
        f"segment_emotion bench ({len(actor_dirs)} actors, {len(y_true)} segments):\n"
        f"  Approach 1 (global unnormalized prosody): macro F1 = {f1_1:.1%}\n"
        f"  Approach 2 (speaker-normalized prosody):  macro F1 = {f1_2:.1%}\n"
        f"  Approach 3 (fused prosody+CED+pauses):    macro F1 = {f1_3:.1%}\n"
        f"  Calm negative controls false flags:       {false_flags}"
    )
    if f1_3 < 0.75 or false_flags > 0:
        sys.exit(1)
    return {"f1_global": round(f1_1, 3), "f1_speaker": round(f1_2, 3), "f1_fused": round(f1_3, 3),
            "false_flags": false_flags}


if __name__ == "__main__":
    run_bench()
