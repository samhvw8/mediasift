# Stream the AudioSet eval shards, write the picked clips as 16 kHz mono wav, keep nothing else.
import json, io, os, sys, collections, subprocess, pyarrow.parquet as pq, fsspec
from concurrent.futures import ThreadPoolExecutor
out = sys.argv[1]; hits = json.load(open(sys.argv[2]))
by = collections.defaultdict(list)
for i, g, r, vid, kind, c in hits: by[i].append((g, r, vid, c))
fs = fsspec.filesystem("https")
def shard(i):
    f = pq.ParquetFile(fs.open(f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval/{i:02d}.parquet", block_size=8 << 20))
    for g in sorted({h[0] for h in by[i]}):
        rows = f.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
        for gg, r, vid, c in by[i]:
            if gg != g: continue
            os.makedirs(f"{out}/{c}", exist_ok=True)
            dst = f"{out}/{c}/{vid}.wav"
            if not os.path.exists(dst):
                subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", dst],
                               input=rows[r]["bytes"], check=True)
    print("shard", i, "done", flush=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(shard, sorted(by)))
print("EXTRACT DONE")
