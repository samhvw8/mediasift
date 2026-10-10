"""Audio track: loudness, log-mel timbre, pauses, and an acoustic-change curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
HOP_S = 0.05
WIN_S = 0.1
N_MELS = 40
ABS_SILENCE_DB = -55.0


@dataclass
class AudioTrack:
    times: np.ndarray    # (M,) frame centres, seconds
    rms_db: np.ndarray   # (M,) dBFS
    logmel: np.ndarray   # (M, N_MELS)
    silent: np.ndarray   # (M,) bool
    pauses: list[tuple[float, float]]  # silent runs long enough to cut inside


def _mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(60.0), hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c > lo:
            fb[m - 1, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        if hi > c:
            fb[m - 1, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return fb


def build_track(y: np.ndarray, sr: int = SR, min_pause_s: float = 0.3) -> AudioTrack:
    hop, win = int(HOP_S * sr), int(WIN_S * sr)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    n_fft = 1 << (win - 1).bit_length()
    fb = _mel_filterbank(n_fft, sr, N_MELS)
    window = np.hanning(win).astype(np.float32)
    rms_db = np.empty(n, np.float32)
    logmel = np.empty((n, N_MELS), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop][:n]
    scale = 1.0 / 32768.0 if y.dtype == np.int16 else 1.0
    for a in range(0, n, 8192):  # blocks keep memory flat on long files
        blk = frames[a:a + 8192].astype(np.float32) * scale
        rms_db[a:a + len(blk)] = 20 * np.log10(np.sqrt((blk ** 2).mean(axis=1)) + 1e-6)
        spec = np.abs(np.fft.rfft(blk * window, n=n_fft)) ** 2
        logmel[a:a + len(blk)] = np.log(spec @ fb.T + 1e-8)
    times = (np.arange(n) * hop + win / 2) / sr

    # Pauses relative to this recording's own dynamic range, so a quiet lecture
    # and a loud vlog both get sensible thresholds. Near-constant sound (music,
    # a drone) has no pauses rather than random ones.
    # p2, not p5: short pauses can be a tiny share of a dense recording.
    floor, loud = np.percentile(rms_db, 2), np.percentile(rms_db, 90)
    silent = rms_db < ABS_SILENCE_DB
    if loud - floor >= 6.0:
        silent |= rms_db < floor + 0.35 * (loud - floor)
    pauses = _runs(silent, times, min_pause_s)
    return AudioTrack(times, rms_db, logmel, silent, pauses)


def _runs(mask: np.ndarray, times: np.ndarray, min_len: float) -> list[tuple[float, float]]:
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            s, e = times[start] - HOP_S / 2, times[i - 1] + HOP_S / 2
            if e - s >= min_len:
                out.append((float(s), float(e)))
            start = None
    return out


def novelty(track: AudioTrack, window_s: float) -> np.ndarray:
    """Distance between the mean timbre of the previous and next `window_s`.
    Fires on speech->music, speaker/room changes, a new sound bed, silence->sound."""
    x = track.logmel.astype(np.float64)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    n = len(x)
    w = max(1, int(round(window_s / HOP_S)))
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    out = np.zeros(n, np.float32)
    for i in range(1, n):
        a, b = max(0, i - w), min(n, i + w)
        left = (c[i] - c[a]) / (i - a)
        right = (c[b] - c[i]) / (b - i)
        out[i] = np.linalg.norm(left - right) / np.sqrt(x.shape[1])
    return out
