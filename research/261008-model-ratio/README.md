# Model ratio research: accuracy per resource for framesift (2026-10-08)

Question: which Hugging Face models give the best accuracy per unit of resource (params, RAM/VRAM, speed on an M1 Pro 16 GB and an RTX 3060 12 GB) for framesift's three model slots, and what is there besides a VLM for frame content?

Method: 8 gatherer agents (EN, VI, ZH, ZH-TW) over HF cards and leaderboards, arXiv, GitHub issues and forums, plus two local tests (Apple Vision OCR and PP-OCRv6 on this Mac). Per-track reports in this folder. Baselines are our own measurements from today (`research/261008-vi-asr-eval/results.md`, `bench/qa*/`).

## Recommendation

| slot | keep | try next (one local test each) | why |
|---|---|---|---|
| ASR English | parakeet-tdt-0.6b-v3 (parakeet-mlx, 24x RT on M1 Pro) | **granite-speech-5.0-470m-turboctc** (Apache-2.0, 470M) | same leaderboard WER (5.84 vs 5.66), ~3.5x throughput, ~165-250x RT on M1 Max via MLX/Core ML; output is lowercase without punctuation ([card](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc), [Granite-MLX](https://github.com/kylehowells/Granite-MLX)) |
| ASR Vietnamese | whisper-large-v3-turbo (2.37% on our TTS slice; Mac = GPU) | **Gipformer 1.5 (68M, MIT)**, Qwen3-ASR-1.7B | Gipformer: best in its own table on real tech-podcast speech (27.5% vs Qwen3-ASR 28.0, Zipformer-30M 29.8, ChunkFormer 39.8, PhoWhisper-L 42.9, Parakeet-vi 48.7), ~50x RT on CPU, commercial-OK — but turbo is missing from that table. Qwen3-ASR: FLEURS-vi 4.91 vs turbo 8.10 in one test, no Vietnamese timestamps |
| Frame embedding | EmbeddingGemma 2 (440M text+vision) | Qwen3-VL-Embedding-2B (Apache-2.0) only if recall is short | EG2 is ~3.6x more accurate per parameter (MMEB Image 57.3 at 0.44B vs 75.0 at 2.1B); nothing sub-1B and commercial-OK beats it. No public Vietnamese image-retrieval number for either |
| Frame text (instead of a VLM) | — (today: Qwen3.5-4B over HTTP) | **Apple Vision OCR on the Mac** (`vi-VT`, 75-120 ms/frame warm, tested here); **PP-OCRv6** on the 3060 for non-Vietnamese | small OCR beats a VLM on screens: PP-OCRv6 small scores 79.7 on screen text vs 73.8 for Qwen3-VL-235B (vendor set); PP-OCRv6 drops Vietnamese tone marks (reproduced: "Tiếng Việt" → "Ting Vit") |
| Frame captions/tags | — | SigLIP2/EG2 zero-shot over ~20 labels (slide, terminal, whiteboard, …); a 0.8-1.3B VLM only where OCR text is thin | tagging models (RAM++) are stale since 2023; no specialised small captioner exists |

Runs: nothing new running. Swapping a model changes a dependency, not the architecture. Apple Vision OCR would need a long-lived helper (first call 66-72 s to compile, then ~0.3 s per new process, 75-120 ms warm).

## The VLM alternative, as a tiered stack
1. **Every keyframe: OCR.** Apple Vision on Mac (multilingual incl. Vietnamese, ~0.1 s); PP-OCRv6 tiny/small via ONNX on Linux (0.35-1.3 s on M4; zero the denormal weights first, [RapidOCR #743](https://github.com/RapidAI/RapidOCR/issues/743)). OCR text goes into the speech-style text index, so `find` and grep both reach on-screen text.
2. **Dense slides or documents: a 0.9-1B OCR model**, PaddleOCR-VL-1.6 (Apache-2.0, 109 languages, MLX/GGUF) or GLM-OCR (MIT, no Vietnamese). 16-22 s per page on a Mac, so only for the frames OCR flags as dense.
3. **A caption only where frames carry little text**, from a 0.8-1.3B VLM (Qwen3.5-0.8B, MiniCPM-V 4.6), about 1/5 of today's 4B.

## Where sources disagree
- **Qwen3-ASR on Vietnamese**: better than turbo on FLEURS-vi read speech (one 150-clip test, [Whisper Notes](https://whispernotes.app/blog/qwen3-asr-vs-whisper)), but behind Zipformer-30M and PhoWhisper-medium on natural speech ([VietAudio leaderboard](https://huggingface.co/spaces/VietAudio-team/Vietnamese-ASR-Leaderboard): 9.84 vs 6.77 / 7.61).
- **Parakeet-CTC-vi**: 9.3 on its card, 12.81 on VietAudio, 5.27 on our TTS slice, 48.7 on Gipformer's tech set. TTS numbers do not predict real speech; our turbo 2.37% has the same weakness.
- **EmbeddingGemma 2 scores** are vendor-reported; MTEB maintainers could not reproduce them yet ([mteb PR 5588](https://github.com/embeddings-benchmark/mteb/pull/5588)).
- **WeMM-Embedding-2B** (MMEB 77.9) beat jina-clip-v2 strongly on 120 Chinese queries in one community test, but uses custom code and its weights license is unclear.

## What is new since 2026-10-01
Almost nothing in seven days: Granite TurboCTC leaderboard entry (10-02), FUTO asr4all (10-07, English only), Omni-Embed-Mini (10-01, weak on images), EmbeddingGemma 2 (10-06), TeleOCR/Xiaomi-OCR-0 re-uploads, PepperOCR-VL (AGPL). The relevant shifts (PP-OCRv6, PaddleOCR-VL-1.6, Gipformer 1.5, Qwen3-ASR ports, Granite 5.0) landed May-September 2026.

## Unresolved, in order of value
1. **Vietnamese ASR on real speech**: whisper turbo vs Gipformer 1.5 vs Qwen3-ASR-1.7B on FLEURS-vi or real podcast audio with a reference (our only reference today is TTS).
2. **Frame OCR on real keyframes**: Apple Vision vs PP-OCRv6 small vs PaddleOCR-VL-1.6 on compressed screen recordings with Vietnamese slides and English code.
3. **Vietnamese text→frame retrieval**: EG2 (70 and 280 tokens) vs Qwen3-VL-Embedding-2B vs SigLIP2 So400m on our frames.
4. Granite TurboCTC on our English audio (WER, timestamps, punctuation step).

## Sources (per-track reports hold the full lists)
`en-asr.md`, `vi-asr.md`, `zh-asr.md`, `en-embedding.md`, `zh-embedding.md`, `en-frame-extract.md`, `zh-frame-extract.md`, `zhtw-ondevice.md`.
