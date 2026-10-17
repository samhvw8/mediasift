import sys, glob, os, time, json, numpy as np, onnxruntime as ort, soundfile as sf
files = sorted(glob.glob(f"{sys.argv[1]}/Actor_*/*.wav")); M = sys.argv[2]; out = {}
h = json.load(open(f"{M}/emotion2vec_head.json")); W, B = np.array(h["weight"], np.float32), np.array(h["bias"], np.float32)
for tag, fn in (("fp32", "emotion2vec_plus_base.onnx"), ("int8", "emotion2vec_plus_base.int8.onnx")):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; s = ort.InferenceSession(f"{M}/{fn}", so)
    t0 = time.perf_counter(); R = []; E = []; tot = 0
    for f in files:
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        if sr != 16000:
            import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        tot += len(y) / 16000
        e = s.run(None, {s.get_inputs()[0].name: y.reshape(1, -1)})[0][0].mean(0); E.append(e)
        z = W @ e + B; p = np.exp(z - z.max()); R.append(p / p.sum())
    dt = time.perf_counter() - t0
    print(f"emotion2vec {tag}: {len(files)} files {tot/60:.1f} min in {dt:.0f}s = {dt/(tot/60):.1f} s per audio min")
    out[tag] = np.stack(R); out[tag + "_emb"] = np.stack(E)
np.savez(sys.argv[3], files=np.array([os.path.basename(f) for f in files]), labels=np.array(h["labels"]), **out)
