# Keep clips where every rater gave intensity 2 for one mapped emotion; write 16 kHz wavs.
import sys, glob, ast, urllib.parse, subprocess, os, collections, pyarrow.parquet as pq
CAT = {"Anger":"angry","Fear":"fearful","Elation":"happy","Amusement":"happy","Sadness":"sad","Astonishment":"surprised","Disgust":"disgusted",
       "Distress":"hi","Triumph":"hi","Fatigue":"lo","Contentment":"lo","Emotional Numbness":"lo","Relief":"lo"}
src, out = sys.argv[1], sys.argv[2]; n = collections.Counter()
for f in sorted(glob.glob(f"{src}/*.parquet")):
    t = pq.read_table(f)
    for i, (a, lab) in enumerate(zip(t.column("audioId").to_pylist(), t.column("label").to_pylist())):
        votes = [list(d.values())[0] for d in ast.literal_eval(lab)]
        cats = {urllib.parse.unquote(list(v)[0]) for v in votes}; scores = [list(v.values())[0] for v in votes]
        if len(cats) != 1 or min(scores) < 2 or len(scores) < 2: continue
        c = CAT.get(cats.pop())
        if not c: continue
        d = f"{out}/{c}"; os.makedirs(d, exist_ok=True); dst = f"{d}/{os.path.basename(f)[:13]}_{i}.wav"
        subprocess.run(["ffmpeg","-v","error","-nostdin","-y","-i","pipe:0","-ac","1","-ar","16000",dst], input=a["bytes"], check=True); n[c] += 1
print(dict(n))
