# Update Log

## 2026-10-10
* **Updated**: [framesift see](decisions/see-sensor.md) — first real-use run: portrait tiles, event crops and tile ids only for listed events, one voice said once; card version see-3.
* **Updated**: [framesift see](decisions/see-sensor.md) — tagger check: Zipformer-small 35/40, CED + Zipformer average 34/40, CED-small alone 36/40 stays.
* **Updated**: [framesift see](decisions/see-sensor.md) — the 4 remaining sound misses traced to CED itself; a runner-up-class naming rule was tried and reverted (missed the cough, added noise to 12 lines).
* **Updated**: [framesift see](decisions/see-sensor.md) — 3 s tagging windows plus span trimming: real-sound naming 25/40 -> 36/40.
* **Updated**: [framesift see](decisions/see-sensor.md) — real-recording checks: ESC-50 sound naming 25/40, LibriSpeech voice grouping 1.00 with TitaNet (replaces the spectrum heuristic, 0.73); `--on gpu-box` measured over ssh (58x realtime, Mac CPU halved).
* **Updated**: [framesift see](decisions/see-sensor.md) — review fixes: VFR and rotated frames, ASR cue merging, audio-only `look`, cache keyed by `--asr`, atomic card, remote ASR timeouts, whole-file copies in sound.
* **Added**: [framesift see](decisions/see-sensor.md) — v0.3 sensor card (picture events, CED-named and counted sounds, OCR, ~1 fps flow sheets, `look` with a sound strip); synthetic bench 26/26 vs 22/26 for uniform sheets; real footage at parity (31/32 vs 32/32); Mac resource fixes (MLX cache cap: swap churn 3.9 GB -> 125 MB; background QoS rejected, 5x slower).
* **Updated**: [pipeline](architecture/pipeline.md) — `see` stages and modules (`events.py`, `sound.py`, `ocr.py`, `see.py`, `polite.py`); `analyze` kept as is.

## 2026-10-08
* **Updated**: [Embedding search](decisions/embedding-search.md), [pipeline](architecture/pipeline.md) — the embedder moved from PyTorch in a warm server to llama.cpp started per command (`find` ~2 s, `index --on HOST` for a GPU box); same retrieval (hit@3 0.87); torch, the `[embed]` extra and `framesift server` removed. llama.cpp is pinned for framesift only and found through mise from any folder.
* **Updated**: [Embedding search](decisions/embedding-search.md) — llama.cpp engine test: same retrieval as PyTorch, Metal ↔ CUDA vectors agree to 0.9998, 2.7-3.2 s start, 1.25 s text-only query start.
* **Updated**: [Embedding search](decisions/embedding-search.md) — long-video baseline: agents grep the 207 KB storyboard (78.7k tokens per question) and `find` costs more (88.0k); Vietnamese ASR measured, whisper turbo best at 2.37% WER.
* **Added**: [Embedding search](decisions/embedding-search.md) — `index` / `find` / `look` on EmbeddingGemma 2, with the retrieval check (hit@3 0.87 vs 0.19 random) and four rounds of the agent QA bench (accuracy tie with contact sheets; 3x fewer image tokens per one-off question but equal total session tokens on short videos).
* **Updated**: [pipeline](architecture/pipeline.md) — new optional search stage (`embed.py`, `index.py`, `query.py`).

## 2026-09-30
* **Creation**: Bundle for framesift v0.1 — [pipeline](architecture/pipeline.md) and five [decisions](decisions/index.md), each with the measurements from the sample corpus and real screen recordings.
