import sys, numpy as np
from framesift import sound, decode
names = sound._labels()
music = sorted(sound.MUSIC)
mood = [i for i in music if names[i].endswith(" music") and names[i].split()[0] in ("Happy","Sad","Tender","Exciting","Angry","Scary","Funny")]
print("MOOD classes:", [(i,names[i]) for i in mood])
for f in sys.argv[1:]:
    pcm = decode.audio_pcm(f, 16000)
    if pcm is None:
        import subprocess
        raw = subprocess.run(["ffmpeg","-v","quiet","-i",f,"-ac","1","-ar","16000","-f","s16le","-"],capture_output=True).stdout
        pcm = np.frombuffer(raw,np.int16)
    t = sound.tag(pcm, hop=1.0, top_k=527)
    p = t.probs.astype(np.float32)
    m = p[:, music].max(1) > 0.35
    print(f"\n== {f}: {len(t.times)} windows, music in {m.sum()}")
    if m.sum()==0: continue
    mp = p[m].mean(0)
    top = [i for i in np.argsort(-mp) if i in sound.MUSIC][:12]
    print("  top music-branch:", ", ".join(f"{names[i]} {mp[i]:.2f}" for i in top))
    print("  moods:", ", ".join(f"{names[i]} {mp[i]:.3f}" for i in mood))
    # mood arc per 10 s
    for a in range(0, len(t.times), 15):
        seg = p[a:a+15]; mm = seg[:, music].max(1) > 0.35
        if mm.sum()<3: continue
        sp = seg[mm].mean(0)
        best = max(mood, key=lambda i: sp[i])
        g = [i for i in np.argsort(-sp) if i in sound.MUSIC and names[i] not in sound.MUSIC_GENERIC][:3]
        print(f"   {t.times[a]:6.0f}s mood={names[best]} {sp[best]:.2f} | {', '.join(f'{names[i]} {sp[i]:.2f}' for i in g)}")
