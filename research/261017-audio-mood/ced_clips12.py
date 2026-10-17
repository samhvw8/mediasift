# CED-small (framesift's own tagger) on every AudioSet eval clip: mean class probs over 1 s-hop windows.
import sys, glob, os, time, numpy as np
from framesift import sound, decode
root = sys.argv[1]; files = sorted(glob.glob(f"{root}/*/*.wav"))
out = {}; t0 = time.perf_counter(); tot = 0
for f in files:
    pcm = decode.audio_pcm(f, 16000); tot += len(pcm) / 16000
    t = sound.tag(pcm, hop=0.5, top_k=12)
    out[os.path.relpath(f, root)] = t.probs.astype(np.float32).mean(0)
print(f"CED {len(files)} clips, {tot/60:.1f} min audio in {time.perf_counter()-t0:.1f}s")
np.savez(sys.argv[2], names=np.array(list(out)), probs=np.stack(list(out.values())), labels=np.array(sound._labels()))
