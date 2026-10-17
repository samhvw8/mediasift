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


# ------------------------------------------------------------------ harmony, rhythm & prosody

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Kessler / Temperley diatonic pitch-class profiles (unit-normalized on use)
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], np.float32)
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], np.float32)


def _chroma_basis(n_fft: int, sr: int, fmin: float = 65.41, fmax: float = 1046.50) -> np.ndarray:
    """(12, n_fft // 2 + 1) constant-Q row-normalized projection from FFT bins to 12 pitch classes."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    W = np.zeros((12, len(freqs)), np.float32)
    valid = (freqs >= fmin) & (freqs < fmax)
    midi = np.zeros_like(freqs)
    midi[valid] = 69.0 + 12.0 * np.log2(freqs[valid] / 440.0)
    inv_f = np.zeros_like(freqs, dtype=np.float32)
    inv_f[valid] = 100.0 / freqs[valid]  # equalize bin density per octave
    for pc in range(12):
        d = np.abs(((midi - pc + 6.0) % 12.0) - 6.0)
        m = valid & (d < 0.55)
        W[pc, m] = 0.5 * (1.0 + np.cos(np.pi * d[m] / 0.55)) * inv_f[m]
    row_sums = W.sum(axis=1, keepdims=True)
    W /= np.maximum(row_sums, 1e-8)
    return W


def chromagram(y: np.ndarray, sr: int = SR, hop_s: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, full_chroma (T, 12), bass_chroma (T, 12)) with spectral whitening."""
    scale = 1.0 / 32768.0 if y.dtype == np.int16 else 1.0
    yf = y.astype(np.float32) * scale
    n_fft = 4096
    hop = max(1, int(round(hop_s * sr)))
    if len(yf) < n_fft:
        yf = np.pad(yf, (0, n_fft - len(yf)))
    n = 1 + (len(yf) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(yf, n_fft)[::hop][:n]
    win = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * win, n=n_fft))
    # Subtract local moving-average spectral floor along frequency axis to isolate harmonic partials
    k_bins = 15
    kernel = np.ones(k_bins, np.float32) / k_bins
    floor = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, mag)
    harm = np.maximum(0.0, mag - 1.15 * floor)
    mag_comp = np.log1p(40.0 * harm)
    # Exact 4-octave range C2..C6 (65.41..1046.50 Hz) and bass range C2..F4 (65.41..349.23 Hz)
    w_full = _chroma_basis(n_fft, sr, 65.41, 1046.50)
    w_bass = _chroma_basis(n_fft, sr, 65.41, 349.23)
    c_full = mag_comp @ w_full.T
    c_bass = mag_comp @ w_bass.T
    times = (np.arange(n) * hop + n_fft / 2) / sr
    return times, c_full, c_bass


def _classify_key_mode(c_full: np.ndarray, c_bass: np.ndarray) -> tuple[str, str, float]:
    """Classify (key_name, mode in {'major', 'minor', 'drone'}, tension) for a segment."""
    if len(c_full) == 0:
        return "C", "drone", 0.0
    energy = c_full.sum(axis=1)
    active = energy > np.percentile(energy, 15) * 0.25
    cf = c_full[active] if active.any() else c_full
    cb = c_bass[active] if active.any() else c_bass

    mean_c = cf.mean(axis=0)
    tot = float(mean_c.sum()) + 1e-8
    p_c = mean_c / tot

    # Frame-to-frame normalized chroma change (zero on static drones, high on chord progressions)
    cf_norm = cf / (cf.sum(axis=1, keepdims=True) + 1e-8)
    if len(cf_norm) >= 6:
        step = min(8, len(cf_norm) - 1)
        chord_flux = float(np.mean(np.linalg.norm(cf_norm[step:] - cf_norm[:-step], axis=1)))
    else:
        chord_flux = 0.0

    top2_share = float(np.sort(p_c)[-2:].sum())
    if top2_share > 0.72 and chord_flux < 0.14:
        root = int(np.argmax(cb.mean(axis=0) + 0.5 * mean_c))
        return PITCH_NAMES[root], "drone", 0.1
    # Score all 12 major and 12 minor keys using:
    # 1. Krumhansl-Kessler profile correlation on full chroma
    # 2. Instantaneous triad quality (major +4 vs minor +3 relative to each frame's bass root)
    # 3. Opening / tonic bass anchor (first 25% of frames and overall bass distribution)
    c_z = (p_c - p_c.mean()) / (p_c.std() + 1e-8)
    maj_p = (_KK_MAJOR - _KK_MAJOR.mean()) / _KK_MAJOR.std()
    min_p = (_KK_MINOR - _KK_MINOR.mean()) / _KK_MINOR.std()

    bass_mean = cb.mean(axis=0)
    bass_mean /= float(bass_mean.sum()) + 1e-8
    n_open = max(1, len(cb) // 4)
    bass_open = cb[:n_open].mean(axis=0)
    bass_open /= float(bass_open.sum()) + 1e-8

    best_score = -1e9
    best_key = "C"
    best_mode = "major"

    for r in range(12):
        kk_maj = float(np.dot(c_z, np.roll(maj_p, r))) / 12.0
        kk_min = float(np.dot(c_z, np.roll(min_p, r))) / 12.0

        # Tonic triad third (+4 vs +3) and fifth (+7) in full chroma, anchored by tonic bass
        tonic_weight = 0.55 * bass_mean[r] + 0.45 * bass_open[r]
        maj_triad = p_c[r] + 1.25 * p_c[(r + 4) % 12] - 0.65 * p_c[(r + 3) % 12] + 0.6 * p_c[(r + 7) % 12]
        min_triad = p_c[r] + 1.25 * p_c[(r + 3) % 12] - 0.65 * p_c[(r + 4) % 12] + 0.6 * p_c[(r + 7) % 12]

        s_maj = 0.50 * kk_maj + 1.40 * tonic_weight + 0.45 * maj_triad
        s_min = 0.50 * kk_min + 1.40 * tonic_weight + 0.45 * min_triad

        if s_maj > best_score:
            best_score, best_key, best_mode = s_maj, PITCH_NAMES[r], "major"
        if s_min > best_score:
            best_score, best_key, best_mode = s_min, PITCH_NAMES[r], "minor"

    # Semitone dissonance / harmonic tension (adjacent semitone products + minor/tritone weight)
    semitone_clash = float(np.sum(p_c * np.roll(p_c, 1)) * 12.0)
    tension = float(np.clip(semitone_clash + (0.15 if best_mode == "minor" else 0.0), 0.0, 1.0))
    return best_key, best_mode, round(tension, 3)


def _onset_tempo(y: np.ndarray, sr: int = SR) -> tuple[float | None, float, list[float]]:
    """Return (bpm or None, pulse_clarity, beat_times_s) from mel/low-weighted spectral-flux autocorrelation."""
    scale = 1.0 / 32768.0 if y.dtype == np.int16 else 1.0
    yf = y.astype(np.float32) * scale
    hop_s = 0.02  # 50 Hz onset envelope
    hop = int(round(hop_s * sr))
    win = int(round(0.04 * sr))
    if len(yf) < win * 4:
        return None, 0.0, []
    n = 1 + (len(yf) - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(yf, win)[::hop][:n]
    fb = _mel_filterbank(1024, sr, 24)
    mag2 = np.abs(np.fft.rfft(frames * np.hanning(win).astype(np.float32), n=1024)) ** 2
    mel_spec = np.log1p(50.0 * np.sqrt(mag2 @ fb.T))
    diff = np.maximum(0.0, np.diff(mel_spec, axis=0, prepend=mel_spec[:1]))
    # Weight bass/low-mid bands (bands 0..10: 60-1200 Hz) higher so kicks/chords beat hi-hats
    band_w = np.linspace(1.6, 0.5, 24, dtype=np.float32)
    flux = (diff * band_w).mean(axis=1)
    bass_flux = diff[:, :8].mean(axis=1)

    # Causal baseline subtraction so onset peaks stay at the attack frame
    k = 12
    pad = np.pad(flux, (k, 0), mode="edge")
    local_mean = np.convolve(pad, np.ones(k, np.float32) / k, mode="valid")[:len(flux)]
    env = np.maximum(0.0, flux - 0.85 * local_mean)
    if float(env.max()) < 1e-4 or float(flux.std()) < 0.010:
        return None, 0.0, []

    env_z = env - env.mean()
    var = float(np.dot(env_z, env_z))
    if var <= 1e-9:
        return None, 0.0, []

    # Autocorrelation on both full onset envelope and bass/downbeat envelope
    n_ac = 1 << (2 * len(env_z) - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(env_z, n=n_ac)) ** 2)[:len(env_z)] / var

    b_z = bass_flux - bass_flux.mean()
    b_var = float(np.dot(b_z, b_z)) + 1e-9
    ac_bass = np.fft.irfft(np.abs(np.fft.rfft(b_z, n=n_ac)) ** 2)[:len(b_z)] / b_var

    # Search lags in 64..180 BPM (lag 17..47 frames at 50 Hz)
    min_lag = int(round((60.0 / 180.0) / hop_s))
    max_lag = min(len(ac) // 2 - 1, int(round((60.0 / 64.0) / hop_s)))
    if max_lag <= min_lag + 2:
        return None, 0.0, []

    def _hmax(arr: np.ndarray, idx_arr: np.ndarray, rad: int = 1) -> np.ndarray:
        out = np.zeros(len(idx_arr), np.float32)
        for d in range(-rad, rad + 1):
            j = np.clip(idx_arr + d, 0, len(arr) - 1)
            out = np.maximum(out, arr[j])
        return out

    lags = np.arange(min_lag, max_lag + 1)
    bpms = 60.0 / (lags * hop_s)
    prior = np.exp(-0.5 * ((np.log2(bpms / 120.0)) / 0.70) ** 2)
    comb = (0.55 * _hmax(ac, lags, 1) + 0.45 * _hmax(ac_bass, lags, 1)
            + 0.35 * _hmax(ac, lags * 2, 1) + 0.25 * _hmax(ac_bass, lags * 2, 1))
    scored = comb * prior

    idx = int(np.argmax(scored))
    best_lag = int(lags[idx])
    # Guard against 3/2 dotted subharmonic when the true beat lag (2/3 * best_lag) has strong bass/onset peaks
    lag_23_cands = [int(np.floor(best_lag * 2.0 / 3.0)), int(np.ceil(best_lag * 2.0 / 3.0))]
    lag_23 = max((c for c in lag_23_cands if c >= min_lag), key=lambda c: float(ac_bass[c] + ac[c]), default=0)
    if lag_23 >= min_lag and ac_bass[lag_23] >= 0.65 * ac_bass[best_lag] and ac[lag_23] >= 0.60 * ac[best_lag]:
        best_lag = lag_23
    # Refine integer best_lag within +/- 1 frame on ac_bass + ac
    lo_l, hi_l = max(min_lag, best_lag - 1), min(max_lag, best_lag + 1)
    best_lag = lo_l + int(np.argmax(ac_bass[lo_l:hi_l + 1] + ac[lo_l:hi_l + 1]))

    raw_peak = float(ac[best_lag])
    double_peak = float(_hmax(ac, np.array([best_lag * 2]), 1)[0])
    triple_peak = float(_hmax(ac, np.array([best_lag * 3]), 2)[0])
    clarity = float(max(raw_peak, 0.65 * raw_peak + 0.35 * double_peak))

    # True musical pulse repeats at L, 2L, and 3L; random speech bursts do not
    med_ac = float(np.median(ac[min_lag:max_lag + 1]))
    if clarity < 0.28 or raw_peak < 0.22 or double_peak < 0.16 or triple_peak < 0.12 or (raw_peak - med_ac) < 0.18:
        return None, round(max(0.0, clarity), 3), []

    # Refine fractional lag using the 2x and 3x harmonic peaks (3x precision against frame quantization)
    l2_idx = best_lag * 2 - 1 + int(np.argmax(ac_bass[best_lag * 2 - 1:best_lag * 2 + 2] + ac[best_lag * 2 - 1:best_lag * 2 + 2]))
    lag_f = 0.5 * float(l2_idx)
    if 1 <= l2_idx < len(ac) - 1:
        comb_sig = ac_bass + ac
        y0, y1, y2 = float(comb_sig[l2_idx - 1]), float(comb_sig[l2_idx]), float(comb_sig[l2_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-6:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.45, 0.45))
            lag_f = 0.5 * (l2_idx + delta)

    bpm = 60.0 / (lag_f * hop_s)
    beat_s = 60.0 / bpm
    dur = len(yf) / sr

    # Lock beat phase using bass-weighted onset envelope and snap each beat to local onset peak (+/- 2 frames)
    phase_env = 0.75 * bass_flux + 0.25 * env
    n_phases = max(20, int(round(lag_f * 2)))
    best_phase = 0.0
    best_phase_score = -1.0
    for p in np.linspace(0.0, beat_s, n_phases, endpoint=False):
        bt = np.arange(p, dur, beat_s)
        fi = np.clip(np.round(bt / hop_s).astype(int), 0, len(phase_env) - 1)
        sc = float(phase_env[fi].mean()) if len(fi) else 0.0
        if sc > best_phase_score:
            best_phase_score = sc
            best_phase = float(p)

    # Verify regular beat occupancy along the grid (music beats stay active; random speech bursts have gaps)
    nom_times = np.arange(best_phase, dur - 0.1, beat_s)
    if len(nom_times) >= 4:
        hit_thr = float(np.percentile(phase_env, 55))
        active_hits = 0
        for t_nom in nom_times:
            c_idx = int(round(t_nom / hop_s))
            lo_i, hi_i = max(0, c_idx - 2), min(len(phase_env) - 1, c_idx + 2)
            if float(phase_env[lo_i:hi_i + 1].max()) > max(hit_thr, 0.02):
                active_hits += 1
        if active_hits / len(nom_times) < 0.72:
            return None, round(max(0.0, clarity * 0.5), 3), []
    # Lock beat phase using bass-weighted onset envelope and snap each beat to local onset peak (+/- 2 frames)
    phase_env = 0.75 * bass_flux + 0.25 * env
    n_phases = max(20, int(round(lag_f * 2)))
    best_phase = 0.0
    best_phase_score = -1.0
    for p in np.linspace(0.0, beat_s, n_phases, endpoint=False):
        bt = np.arange(p, dur, beat_s)
        fi = np.clip(np.round(bt / hop_s).astype(int), 0, len(phase_env) - 1)
        sc = float(phase_env[fi].mean()) if len(fi) else 0.0
        if sc > best_phase_score:
            best_phase_score = sc
            best_phase = float(p)

    beats: list[float] = []
    for t_nom in np.arange(best_phase, dur, beat_s):
        c_idx = int(round(t_nom / hop_s))
        lo_i = max(0, c_idx - 2)
        hi_i = min(len(phase_env) - 1, c_idx + 2)
        pk_i = lo_i + int(np.argmax(phase_env[lo_i:hi_i + 1]))
        beats.append(round(float(pk_i * hop_s), 3))
    return round(bpm, 1), round(clarity, 3), beats


def analyze_harmony_rhythm(y: np.ndarray, sr: int = SR) -> dict:
    """Extract segment-level key, mode (`major`, `minor`, `drone`), harmonic shifts (`harmony_arc`),
    tempo (`bpm`), `pulse_clarity`, and `beats` from a 16 kHz mono signal."""
    dur = round(len(y) / sr, 2)
    times, c_full, c_bass = chromagram(y, sr, hop_s=0.1)
    key, mode, tension = _classify_key_mode(c_full, c_bass)

    # Harmonic change detection (CUSUM two-segment chroma distance) for beds >= 6 s
    harmony_arc: list[tuple[float, float, str, str]] = [(0.0, dur, mode, key)]
    if dur >= 6.0 and len(times) >= 60:
        margin = max(25, int(len(times) * 0.2))
        best_i = -1
        best_dist = 0.0
        cf_n = c_full / (c_full.sum(axis=1, keepdims=True) + 1e-8)
        cb_n = c_bass / (c_bass.sum(axis=1, keepdims=True) + 1e-8)
        feat = np.concatenate([cf_n, 0.8 * cb_n], axis=1)
        csum = np.concatenate([np.zeros((1, feat.shape[1])), np.cumsum(feat, axis=0)])
        n_tot = len(times)
        for i in range(margin, n_tot - margin):
            left = csum[i] / i
            right = (csum[n_tot] - csum[i]) / (n_tot - i)
            w_split = np.sqrt((i * (n_tot - i)) / (n_tot * n_tot)) * 2.0
            d = float(np.linalg.norm(left - right) * w_split)
            if d > best_dist:
                best_dist = d
                best_i = i
        if best_i > 0 and best_dist >= 0.085:
            k1, m1, _ = _classify_key_mode(c_full[:best_i], c_bass[:best_i])
            k2, m2, _ = _classify_key_mode(c_full[best_i:], c_bass[best_i:])
            t_s = round(float(times[best_i]), 2)
            if (m1 != m2 or (k1 != k2 and best_dist >= 0.16)) and 2.5 <= t_s <= dur - 2.5:
                harmony_arc = [(0.0, t_s, m1, k1), (t_s, dur, m2, k2)]
    bpm, clarity, beats = _onset_tempo(y, sr)
    return {
        "key": key,
        "mode": mode,
        "tension": tension,
        "harmony_arc": harmony_arc,
        "bpm": bpm,
        "pulse_clarity": clarity,
        "beats": beats,
    }


def cut_beat_sync(beats: list[float], cut_times: list[float], tol_s: float = 0.065) -> dict:
    """Measure how many visual cuts land within +/- `tol_s` of the beat grid."""
    if len(beats) < 4 or len(cut_times) < 3:
        return {"synced": False, "hits": 0, "total": len(cut_times), "ratio": 0.0}
    grid = np.asarray(beats, dtype=np.float64)
    beat_s = float(np.median(np.diff(grid)))
    hits = 0
    for c in cut_times:
        dist = float(np.min(np.abs(grid - c)))
        if dist <= tol_s:
            hits += 1
    ratio = hits / len(cut_times)
    chance = min(0.5, (2.0 * tol_s) / max(beat_s, 1e-3))
    synced = hits >= 3 and ratio >= max(0.60, 1.8 * chance)
    return {"synced": bool(synced), "hits": hits, "total": len(cut_times), "ratio": round(ratio, 3)}


def prosody_features(y: np.ndarray, sr: int = SR, hop_s: float = HOP_S) -> dict[str, np.ndarray]:
    """Vectorized frame-level prosody: `times`, `voiced`, `rms_db`, `tilt_db` (0.5-4 kHz vs
    60-500 Hz spectral balance), and `f0_hz` via harmonic autocorrelation."""
    scale = 1.0 / 32768.0 if y.dtype == np.int16 else 1.0
    yf = y.astype(np.float32) * scale
    hop = max(1, int(round(hop_s * sr)))
    win = int(round(0.064 * sr))  # 64 ms window (1024 samples at 16 kHz)
    if len(yf) < win:
        yf = np.pad(yf, (0, win - len(yf)))
    n = 1 + (len(yf) - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(yf, win)[::hop][:n]
    w = np.hanning(win).astype(np.float32)
    xw = frames * w

    rms_db = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-7)
    spec = np.abs(np.fft.rfft(xw, n=1024)) ** 2
    freqs = np.fft.rfftfreq(1024, 1.0 / sr)
    lo_mask = (freqs >= 60.0) & (freqs < 500.0)
    hi_mask = (freqs >= 500.0) & (freqs <= 4000.0)
    e_lo = spec[:, lo_mask].sum(axis=1) + 1e-10
    e_hi = spec[:, hi_mask].sum(axis=1) + 1e-10
    tilt_db = 10.0 * np.log10(e_hi / e_lo)

    # Vectorized autocorrelation pitch in [75 Hz, 420 Hz]
    ac = np.fft.irfft(spec, n=1024)[:, :win // 2]
    ac0 = ac[:, 0] + 1e-9
    min_lag = int(round(sr / 420.0))
    max_lag = min(ac.shape[1] - 1, int(round(sr / 75.0)))
    sub = ac[:, min_lag:max_lag + 1] / ac0[:, None]
    best_rel = np.argmax(sub, axis=1)
    peak_corr = sub[np.arange(n), best_rel]
    lags = best_rel + min_lag
    f0_hz = sr / np.maximum(lags.astype(np.float32), 1.0)

    loud_p90 = float(np.percentile(rms_db, 90)) if n else -40.0
    voiced = (rms_db > max(-48.0, loud_p90 - 26.0)) & (peak_corr > 0.28)
    times = (np.arange(n) * hop + win / 2) / sr
    return {
        "times": times.astype(np.float32),
        "voiced": voiced,
        "rms_db": rms_db.astype(np.float32),
        "tilt_db": tilt_db.astype(np.float32),
        "f0_hz": f0_hz.astype(np.float32),
    }
