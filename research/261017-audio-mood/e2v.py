import sys, json, time, numpy as np, onnxruntime as ort, subprocess
M = sys.argv[1]
h = json.load(open(f"{M}/emotion2vec_head.json")); W, B, L = np.array(h["weight"], np.float32), np.array(h["bias"], np.float32), h["labels"]
so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession(f"{M}/emotion2vec_plus_base.int8.onnx", so)
def emo(y):
    f = sess.run(None, {sess.get_inputs()[0].name: y.reshape(1, -1).astype(np.float32)})[0][0].mean(0)
    z = W @ f + B; p = np.exp(z - z.max()); return p / p.sum()
if __name__ == "__main__":
    for f in sys.argv[2:]:
        raw = subprocess.run(["ffmpeg","-v","quiet","-i",f,"-t","90","-ac","1","-ar","16000","-f","s16le","-"],capture_output=True).stdout
        y = np.frombuffer(raw,np.int16).astype(np.float32)/32768
        t=time.perf_counter(); rows=[]
        for a in range(0, len(y)-16000, 16000*4):
            p = emo(y[a:a+16000*4]); top=np.argsort(-p)[:2]
            rows.append(f"{a//16000:3d}s {L[top[0]]} {p[top[0]]:.2f} / {L[top[1]]} {p[top[1]]:.2f}")
        print(f"== {f.split('/')[-1]}: {len(y)/16000:.0f}s in {time.perf_counter()-t:.1f}s"); print("\n".join(rows))
