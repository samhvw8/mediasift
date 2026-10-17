import sys, time, numpy as np, sherpa_onnx, subprocess
M = sys.argv[1]
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{M}/model.int8.onnx", tokens=f"{M}/tokens.txt", num_threads=2, use_itn=False, language="auto")
def run(y):
    s = rec.create_stream(); s.accept_waveform(16000, y); rec.decode_stream(s); return s.result
for f in sys.argv[2:]:
    raw = subprocess.run(["ffmpeg","-v","quiet","-i",f,"-t","60","-ac","1","-ar","16000","-f","s16le","-"],capture_output=True).stdout
    y = np.frombuffer(raw,np.int16).astype(np.float32)/32768
    t=time.perf_counter(); out=[]
    for a in range(0, len(y), 16000*5):
        r = run(y[a:a+16000*5]); out.append((a//16000, r.lang, r.emotion, r.event, r.text[:40]))
    dt=time.perf_counter()-t
    print(f"== {f}: {len(y)/16000:.0f}s audio in {dt:.1f}s")
    for o in out: print("  ", o)
