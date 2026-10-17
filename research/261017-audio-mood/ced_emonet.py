# framesift env: CED mean probs + cheap prosody per RAVDESS file (features for a cross-actor probe).
import sys, glob, os, time, numpy as np, librosa
from framesift import sound, decode
files = sorted(glob.glob(f"{sys.argv[1]}/*/*.wav")); P, F = [], []; t0 = time.perf_counter(); tot = 0
for f in files:
    pcm = decode.audio_pcm(f, 16000); y = pcm.astype(np.float32) / 32768; tot += len(y) / 16000
    P.append(sound.tag(pcm, hop=1.0, top_k=527).probs.astype(np.float32).mean(0))
    rms = librosa.feature.rms(y=y, frame_length=800, hop_length=160)[0]; v = rms > rms.max() * 0.1
    f0 = librosa.yin(y, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=160)[: len(v)]
    f0v = np.log(f0[v[: len(f0)]]) if v.any() else np.zeros(1)
    db = 20 * np.log10(rms[v] + 1e-6)
    F.append([f0v.mean(), f0v.std(), np.percentile(f0v, 90) - np.percentile(f0v, 10), db.mean(), db.std(), db.max(),
              v.mean(), librosa.feature.spectral_centroid(y=y, sr=16000).mean() / 1000, librosa.feature.zero_crossing_rate(y).mean()])
print(f"CED+prosody {len(files)} files {tot/60:.1f} min in {time.perf_counter()-t0:.1f}s")
np.savez(sys.argv[2], files=np.array([os.path.relpath(f, sys.argv[1]) for f in files]), ced=np.stack(P), pros=np.array(F, np.float32))
