"""Voice letters on real speech: LibriSpeech dev-clean speakers spliced into dialogues.

    uv run python bench/sense/voices.py [CAMPP_ONNX]

Each dialogue alternates 2 or 3 real speakers over 12 turns (3-10 s each, 0.4 s gaps). Methods
get the true turn spans and label them; scored by pairwise agreement (two turns are "same voice"
in both the truth and the labels, or "different" in both) and by voices found vs true.
"""
from __future__ import annotations

import itertools
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LS = ROOT / "samples" / "librispeech" / "LibriSpeech" / "dev-clean"
SR = 16000
DIALOGUES = [("251", "1272"), ("1462", "2035"), ("2078", "3752"), ("251", "1462"), ("1272", "2035"),
             ("251", "1272", "3752"), ("1462", "2035", "2078"), ("251", "1462", "3752")]


def load(f: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16)


def utterances(spk: str, rng: random.Random) -> list[np.ndarray]:
    files = sorted((LS / spk).rglob("*.flac"))
    rng.shuffle(files)
    out = []
    for f in files:
        y = load(f)
        if 3 * SR <= len(y) <= 10 * SR:
            out.append(y)
        if len(out) >= 12:
            break
    return out


def dialogue(spks: tuple[str, ...], rng: random.Random):
    pools = {s: utterances(s, rng) for s in spks}
    order = [spks[i % len(spks)] for i in range(12)]
    for i in range(2, 12, 4):                     # some speakers keep the floor for two turns
        order[i] = order[i - 1]
    pcm, spans, truth, t = [np.zeros(SR // 2, np.int16)], [], [], 0.5
    for s in order:
        y = pools[s].pop()
        pcm.append(y)
        spans.append((t, t + len(y) / SR))
        truth.append(s)
        t += len(y) / SR + 0.4
        pcm.append(np.zeros(int(0.4 * SR), np.int16))
    return np.concatenate(pcm), spans, truth


def pairwise(truth: list[str], got: list[str | None]) -> float:
    got = [g or "?" for g in got]
    pairs = list(itertools.combinations(range(len(truth)), 2))
    return sum((truth[i] == truth[j]) == (got[i] == got[j]) for i, j in pairs) / len(pairs)


def campp(model: str):
    import sherpa_onnx
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=4))

    def label(pcm: np.ndarray, spans, thr: float = 0.5) -> list[str]:
        embs = []
        for s, e in spans:
            st = ex.create_stream()
            st.accept_waveform(SR, pcm[int(s * SR):int(e * SR)].astype(np.float32) / 32768)
            st.input_finished()
            v = np.array(ex.compute(st))
            embs.append(v / np.linalg.norm(v))
        sums: list[np.ndarray] = []
        out = []
        for v in embs:
            d = [1 - float(v @ (c / np.linalg.norm(c))) for c in sums]
            j = int(np.argmin(d)) if d else -1
            if j >= 0 and d[j] < thr:
                sums[j] = sums[j] + v
            else:
                sums.append(v.copy())
                j = len(sums) - 1
            out.append(chr(65 + j))
        return out
    return label


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from framesift import audio, sound
    rng = random.Random(11)
    methods = {"titanet (shipped)": lambda pcm, spans: sound.voices(pcm, spans)}
    if len(sys.argv) > 1:
        for thr in (0.5, 0.6, 0.7):
            f = campp(sys.argv[1])
            methods[f"{Path(sys.argv[1]).stem[:12]} thr {thr}"] = (lambda f, thr: lambda pcm, spans: f(pcm, spans, thr))(f, thr)
    scores = {m: [] for m in methods}
    found = {m: [] for m in methods}
    for spks in DIALOGUES:
        pcm, spans, truth = dialogue(spks, rng)
        for m, fn in methods.items():
            got = fn(pcm, spans)
            scores[m].append(pairwise(truth, got))
            found[m].append(f"{len(set(g for g in got if g))}/{len(spks)}")
    for m in methods:
        print(f"{m:20} pairwise agreement {np.mean(scores[m]):.2f}   voices found/true: {' '.join(found[m])}")


if __name__ == "__main__":
    main()
