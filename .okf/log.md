# Update Log

## 2026-10-10
* **Updated**: [framesift see](decisions/see-sensor.md) — real bench r13: q02 (pan vs tilt) now right; 31/32.
* **Updated**: [framesift see](decisions/see-sensor.md) — camera moves (pan / tilt / scroll) from phase correlation; card see-7.
* **Updated**: [framesift see](decisions/see-sensor.md) — CED-base retested on the 200-clip bench: no gain; CED-small stays.
* **Updated**: [framesift see](decisions/see-sensor.md) — keeping hits inside speech stretches tried and reverted (+5 named, +7 stray).
* **Updated**: [framesift see](decisions/see-sensor.md) — plain name for change ringing: silent set 161 -> 164.
* **Updated**: [framesift see](decisions/see-sensor.md) — sounds under music named from 0.18 over 1.5 s; names below 0.3 hedged as 'maybe'; card see-6.
* **Updated**: [framesift see](decisions/see-sensor.md) — real-song bed; weak lone hits in music dropped: stray lines 48 -> 32.
* **Updated**: [framesift see](decisions/see-sensor.md) — 200-clip held-out sound bench with speech/music beds; per-onset speech check cuts stray lines under speech 55 -> 20.
* **Updated**: [framesift see](decisions/see-sensor.md) — a failing box (GPU busy) falls back to local speech instead of failing the card.
* **Updated**: [framesift see](decisions/see-sensor.md) — speaker embeddings in parallel: 1h45m podcast 318 s -> 213 s.
* **Updated**: [framesift see](decisions/see-sensor.md) — OCR speed-ups measured and rejected (text pre-check loses text; threads ~10%).
* **Updated**: [framesift see](decisions/see-sensor.md) — box speech overlaps the picture and OCR passes: 29 s -> 18-20 s on a 2:44 reel, no added lag.
* **Updated**: [framesift see](decisions/see-sensor.md) — `fix` / `feedback`: agent corrections teach sound names (3 agreeing fixes).
* **Updated**: [framesift see](decisions/see-sensor.md) — real bench r12 after the cue and watermark changes: parity 31/32.
* **Updated**: [framesift see](decisions/see-sensor.md) — platform watermarks named once in the header, not per jump; card see-5.
* **Updated**: [framesift see](decisions/see-sensor.md) — whisper cue splitting: comma, pause, capital, then cap; no mid-phrase cuts or fragments; card see-4.
* **Updated**: [framesift see](decisions/see-sensor.md) — synthetic bench r11 with the final code: 26/26.
* **Updated**: [framesift see](decisions/see-sensor.md) — real bench r10 with the final code: parity at 31/32, 51 vs 58 images.
* **Updated**: [framesift see](decisions/see-sensor.md) — real bench r7 caught a regression (crop rule on screen recordings); the rule now applies to camera footage only, confirmed in r8/r9.
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
