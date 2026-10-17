# EffNet-Discogs + MTG-Jamendo mood/theme head, and CLAP zero-shot, on the same clips.
import sys, glob, os, time, json, numpy as np, onnxruntime as ort, soundfile as sf, essentia.standard as es
root, M = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{root}/*/*.wav"))
so = ort.SessionOptions(); so.intra_op_num_threads = 4
enc = ort.InferenceSession(f"{M}/discogs-effnet-bsdynamic-1.onnx", so)
head = ort.InferenceSession(f"{M}/mtg_jamendo_moodtheme-discogs-effnet-1.onnx", so)
mel = es.TensorflowInputMusiCNN()
def patches(y):
    fr = [mel(es.Windowing(type="hann", normalized=False)(f) if False else f) for f in []]
    frames = np.array([mel(f) for f in es.FrameGenerator(y, frameSize=512, hopSize=256, startFromZero=True)])
    n = max(1, 1 + (len(frames) - 128) // 62)
    if len(frames) < 128: frames = np.pad(frames, ((0, 128 - len(frames)), (0, 0)))
    return np.stack([frames[i*62:i*62+128] for i in range(n)]).astype(np.float32)
E = {}; t0 = time.perf_counter(); tot = 0
for f in files:
    y, sr = sf.read(f, dtype="float32"); tot += len(y) / sr
    act, emb = enc.run(None, {"melspectrogram": patches(y)})
    mood = head.run(None, {"embeddings": emb})[0]
    E[os.path.relpath(f, root)] = (mood.mean(0), act.mean(0))
dt = time.perf_counter() - t0
print(f"EffNet {len(files)} clips, {tot/60:.1f} min in {dt:.1f}s ({dt/(tot/60):.2f} s per audio min)")
names = list(E)
res = dict(names=np.array(names), moodtheme=np.stack([E[n][0] for n in names]), genre400=np.stack([E[n][1] for n in names]))
if "--clap" in sys.argv:
    import torch, librosa
    from transformers import ClapModel, ClapProcessor
    torch.set_num_threads(4)
    mid = "laion/clap-htsat-unfused"
    model, proc = ClapModel.from_pretrained(mid).eval(), ClapProcessor.from_pretrained(mid)
    prompts = json.load(open(sys.argv[sys.argv.index("--clap") + 1]))
    with torch.no_grad():
        te = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True)); te = getattr(te, 'pooler_output', te)
        te = te / te.norm(dim=-1, keepdim=True)
        t0 = time.perf_counter(); A = []
        for n in names:
            y, _ = librosa.load(f"{root}/{n}", sr=48000)
            ae = model.get_audio_features(**proc(audio=[y], sampling_rate=48000, return_tensors="pt")); ae = getattr(ae, 'pooler_output', ae)
            A.append((ae / ae.norm(dim=-1, keepdim=True) @ te.T)[0].numpy())
    dt = time.perf_counter() - t0
    print(f"CLAP {dt:.1f}s ({dt/(tot/60):.2f} s per audio min)")
    res["clap"] = np.stack(A); res["prompts"] = np.array(prompts)
np.savez(sys.argv[3], **res)
