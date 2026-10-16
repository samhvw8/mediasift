# Q3: Apple Silicon ASR alternatives (EN gatherer) - 2026-10-16

Baseline (from brief): QUAN-Transcribe 156x RT M1 Pro (vi); parakeet-mlx v3 ~25x M1 Pro; mlx-whisper large-v3-turbo.

## Bottom line (brutal)
1. Non-Vietnamese path: FluidAudio (Swift, Core ML on ANE) runs the SAME parakeet-tdt-0.6b-v3 at 129-210x RT vs 25x for parakeet-mlx. ~5-8x faster, same languages, same WER. Cost: Swift CLI via subprocess (no Python wheel). New `ultra` variant: same speed, FLEURS mean WER 14.81 -> 11.67. [GEM]
2. Nothing found that beats QUAN on Vietnamese speed. Only Qwen3-ASR-1.7B beats it on accuracy (FLEURS vi 5.55 vs Whisper-turbo 8.1), but runs ~10-13x RT on Mac, i.e. ~12x slower than QUAN. Candidate as an optional "accuracy mode" / fallback for fillers + English names (QUAN drops those). [GEM]
3. Apple SpeechAnalyzer: good English (2.12/4.56 LS), ~30 locales, no evidence Vietnamese is supported. Not a replacement. [MEH]
4. Open ASR Leaderboard top open-weight models are English-centric and measured on H100/H200 RTFx, not Mac. Only Qwen3-ASR-1.7B and Cohere Transcribe on it cover Vietnamese.

## Open ASR Leaderboard (fetched live 2026-10-16, "Last updated 15 Oct 2026")
Source: https://hf-audio-open-asr-leaderboard.hf.space/ (default English datasets avg; RTFx = GPU, not Apple). Multilingual tab NOT retrievable (JS tab) - gap.
Top is closed APIs (OtterAI 4.55, zoom 4.70, azure 5.02, ElevenLabs scribe_v2 5.24, AssemblyAI 5.39...). Open weights, avg WER / RTFx:
| Model | Avg WER | RTFx (GPU) | Notes |
|---|---|---|---|
| Qwen/Qwen3-ASR-1.7B-hf | 5.40 | 819 | 30 langs incl. vi, Apache-2.0 |
| CohereLabs/cohere-transcribe-03-2026 | 5.77 | 902 | 14 langs incl. vi, 2B, Apache-2.0 |
| TheStageAI/thewhisper-large-v3-turbo | 5.91 | 1177 | |
| nvidia/canary-qwen-2.5b | 5.94 | 862 | English |
| ibm-granite/granite-speech-4.1-2b | 5.96 | 545 | |
| nvidia/parakeet-tdt-0.6b-v2 | 6.02 | 5988 | English only |
| ibm-granite/granite-speech-5.0-470m-turboctc | 6.02 | 20831 | English only, Apache-2.0, mlx-audio >=0.5.1 supports it |
Arxiv paper of the leaderboard: https://arxiv.org/html/2510.06961v4 (TDT/CTC 10-100x faster than LLM decoders at slightly higher WER).

## Sub-area (a): ANE / Core ML ports - GEM 2, MEH 3
### FluidAudio parakeet-tdt-0.6b-v3 / ultra / redux [GEM]
- Swift SDK + `fluidaudiocli`, Core ML on ANE. 3102 stars, pushed 2026-10-15, latest v0.17.7 (2026-10-08). License: not shown by gh (spdx empty) - check LICENSE; HF models CC-BY-4.0 upstream. https://github.com/FluidInference/FluidAudio
- M4 Pro (2024 MBP, 48GB, macOS 26.0), FLEURS 24 langs: avg WER 14.7, RTFx 209.8 (range 183-237). LibriSpeech test-clean v3 2.5% WER, RTFx 155.6 overall. https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md
- M5 Pro, ANE: v3 2.27/4.12 LS clean/other, 149-152x clean, 335x on 60-min file; ultra 2.13/3.81, 457x on 60-min, FLEURS 24-lang mean 11.67 (v3 14.81); redux (2-bit, 220MB) 13.06 FLEURS mean but 83.9x. https://github.com/FluidInference/FluidAudio/releases
- HF card claims ~110x RTF on M4 Pro. No M1 Pro number published (UNMEASURED; M1 ANE is ~2x slower than M4 per Apple specs, still expect >>25x - needs a local run).
- Languages: 25 European + Japanese (+ SenseVoice, Paraformer for Mandarin in FluidAudio). NO Vietnamese. Output has no punctuation for v3 per Benchmarks.md.
- Python 3.14: no wheel; shell out to Swift binary (`swift run -c release fluidaudiocli transcribe file.wav --model-version ultra`) or build once.
- Issue: iOS int8 encoder fails on A16 (#828), M-series fine. Redux first ANE compile "several minutes" (cached).
### Phonon-2 (FermionResearch / FluidInference/phonon-2-coreml) [MEH for us - English only]
- Parakeet-v3-derived, 2-bit QAT encoder, 164MB. LS clean 1.72/other 3.92, OA-leaderboard 7 English sets avg 5.21. 174x RT M5 MacBook Air (MLX); Core ML ANE 159x LS clean on M5 Pro, 480-644x on 1-h file (tweet, UNVERIFIED for 644x). CC-BY-4.0. https://huggingface.co/FermionResearch/Phonon-2 ; https://x.com/Alex_tra_memory/status/2109868778615976113
- English only -> irrelevant unless English-specific route wanted.
### Nemotron 3.5 ASR Streaming Core ML (FluidInference) [MEH]
- 40 locales incl. vi-VN (transcription-ready tier). FLEURS vi WER 11.2-13.6 depending on chunk size (80ms-1.12s) -> worse than Whisper-turbo (8.1) and QUAN. License openmdw-1.1. https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b . Streaming-oriented; no reason to use for file transcription.
### WhisperKit / Argmax [MEH]
- argmax-oss-swift (WhisperKit) 6405 stars, pushed 2026-10-11. large-v3-turbo: 42x RT default ANE-only, 72x GPU+ANE on M2 Ultra (maintainer, 2024). https://github.com/argmaxinc/argmax-oss-swift/discussions/243 . Slower than parakeet ANE; Whisper-quality, 99 langs.
- Argmax Pro SDK 3 (commercial) ships Qwen3-ASR-1.7B on iOS/Mac and parakeet-v2 (359x speed factor, M4 mini, English). https://www.argmaxinc.com/blog/apple-and-argmax ; tweet https://x.com/argmax/status/2082875040807203061. Paid -> skip.
- OpenBench (argmaxinc/OpenBench, 89 stars) = benchmark harness, not a model.
### Cohere Transcribe Core ML 6-bit (BarathwajAnandan) [MEH-GEM]
- 1.4GB, M4 Pro GPU compute: 35.9x RT, LS test-clean 2.58%. Community port, 4.9k downloads. https://huggingface.co/BarathwajAnandan/cohere-transcribe-03-2026-CoreML-6bit . Slower than parakeet ANE; value is only vi coverage (see c).

## Sub-area (b): MLX ports - GEM 2, MEH 2
### Qwen3-ASR 0.6B / 1.7B (Alibaba) [GEM for vi accuracy]
- Apache-2.0. 30 langs + 22 Chinese dialects, includes Vietnamese. Official FLEURS vi WER: 1.7B 5.55, 0.6B 8.52, (Flash API 3.64); MLC-SLM vi 14.92 (1.7B). Whisper-large-v3 on same table is far worse in several langs. https://arxiv.org/html/2601.21337v1 (Tables A.2). LS clean 1.63 (1.7B).
- Independent Mac run (M5 MacBook Air, ~150 FLEURS sentences): vi 4.91 vs Whisper-turbo 8.10; speed 9.6x RT (1.7B) vs 2.6x (turbo, whisper.cpp-style); peak 2.43GiB. https://whispernotes.app/blog/qwen3-asr-vs-whisper (vendor-ish blog but paired run with CIs).
- MLX ports: mlx-qwen3-asr (moona3k, 229 stars, pushed 2026-09-20; PyPI, Python >=3.10, CLI `mlx-qwen3-asr`, streaming + word timestamps): 0.6B fp16 RTF 0.029 on 10s clip (~34x), 4-bit 0.018 (~55x); 1.7B fp16 RTF 0.077 (~13x). Chip NOT stated (UNVERIFIED which M-series). LS clean 0.6B 2.33 / 1.7B 1.94. https://pypi.org/project/mlx-qwen3-asr/ . Also gabrimatic/qwen3-asr-mlx (PyPI qwen3-asr-mlx), mlx-community/Qwen3-ASR-0.6B-8bit (218k dl, via mlx-audio). Other runtime smoke test: 0.6B 9-15x RT, 1.7B 4.6-7x (drguptavivek, chip unstated).
- Net: 1.7B accuracy for vi good (5-5.5%), speed ~5-13x RT on Mac => 12-30x slower than QUAN. Handles English words/fillers inside Vietnamese better than a CTC reader (not tested; hypothesis). Mixed-language long dubs: test it on the 47-min clean dub as the head-to-head vs QUAN 5.45 / Whisper 2.75.
### mlx-audio (Blaizzy) [GEM as the single Python harness]
- 8051 stars, MIT?, pushed 2026-10-14, release v0.5.9 (2026-10-13). One `python -m mlx_audio.stt.generate --model ...` CLI for parakeet (v2/v3), Qwen3-ASR, Cohere ASR, Granite Speech 5.0, MOSS-Transcribe-Diarize, Whisper etc. https://github.com/Blaizzy/mlx-audio . Easiest route to trial every candidate on Py3.14 (check wheel/deps on 3.14 - untested).
### Cohere Transcribe via MLX [MEH until vi WER found]
- 2B, Apache-2.0, 14 langs incl. Vietnamese (en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko). https://docs.cohere.com/docs/transcribe . #1 Open ASR (5.42 at launch; now 5.77 on refreshed set). MLX: native Swift (neo773, 0 stars, 4.7GB resident), soniqo speech-swift INT5 default: 0.0150 mean RTF (~67x, chip UNSTATED) 1.62GiB, `speech transcribe --engine cohere` https://soniqo.audio/guides/cohere-transcribe ; mlx-audio module. Per-language WER chart is an image (vi value not extracted) - GAP. TypeWhisper note: "strong for ... Vietnamese" (vendor-ish).
### Granite Speech 5.0 470M TurboCTC, Nemotron-speech-en MLX [MEH]
- English only. Granite 5.0 mlx-q8 ports exist (iky1e). Skip for multilingual.

## Sub-area (c): new model families - GEM 2, MEH 4, NOISE 2
- Moonshine tiny-vi (27M) [MEH/GEM-small]: FLEURS vi WER 13.2-13.3 (vs Whisper-medium 13.45, Whisper-tiny 91.9); CV17 vi 22.7. Tiny/fast, edge CPU. License CONFLICT: GitHub README says all MIT except legacy non-streaming non-English under non-commercial Moonshine Community License; DeepWiki lists vi under Community license; an HF GGUF card says MIT. -> UNVERIFIED; check LICENSE file. 11.2k stars, pushed 2026-10-02. Worse than QUAN; only useful as ultra-cheap CPU LID/draft. https://huggingface.co/myflow-ai/moonshine-tiny-vi-gguf
- transcribe.cpp (handy-computer) [GEM as runtime]: MIT, 2048 stars, pushed today, ggml, 16+ families (Qwen3-ASR 0.6B/1.7B, Cohere, Voxtral, Parakeet v2/v3/ultra, Moonshine, SenseVoice, Nemotron, canary-1b-v2), bindings Python/TS/Rust/Swift, Metal. A single non-Python-dependency path (C lib + Python binding) to GGUF of everything. https://github.com/handy-computer/transcribe.cpp . Speed on Apple not published in what I read - GAP.
- Voxtral Mini 4B Realtime 2602 [MEH]: Apache-2.0, 4B, FLEURS avg 8.72% at 480ms (13 langs: ar de en es fr hi it nl pt zh ja ko ru). NO Vietnamese. ~8GB bf16; HN: "very slow on M1 Max" with C impl (https://news.ycombinator.com/item?id=46954049). Skip.
- Canary-1b-v2 [MEH]: 25 European langs, 978M, FLEURS 8.40, OA-leaderboard avg 7.15 RTFx 749 (GPU). No vi. MLX/GGUF ports exist (small downloads). Skip.
- Canary-Qwen-2.5B [NOISE for us]: English, OA avg 5.94.
- SenseVoice-small [NOISE for vi]: zh/yue/en/ja/ko; no vi. FluidAudio has Core ML variant for Mandarin.
- VibeVoice-ASR (Microsoft) [MEH]: 50+ langs, 60-min single pass with speakers+timestamps; vLLM; no Apple Silicon speed/size verified; vi inclusion not verified. https://github.com/microsoft/VibeVoice
- MOSS-Transcribe-Diarize [MEH]: 0.9B, 50+ langs, mlx-audio; LS 2.07/4.68 on M2 Pro; ~1.7GB; empty-output bug when speech starts at t=0 (pad 1s silence). https://lyonesse.app/blog/parakeet-moss-apple-speech-benchmark.html (vendor blog, raw transcripts public).
- Qwen-Audio-3.0-ASR (arXiv 2609.07549) [UNKNOWN]: CV15 vi WER row exists; weights/licence availability not verified. Closed? skip until checked.
- Kyutai STT, Gemma 3n audio, lightning-whisper-mlx, whisper.cpp Core ML encoder, Phi-4-multimodal: NOT RESEARCHED (budget; none expected to beat the above for vi: Kyutai = en/fr; lightning-whisper-mlx is Whisper on MLX, same family as current mlx-whisper).

## Sub-area (d): Apple SpeechAnalyzer / SpeechTranscriber - GEM 0, MEH 2
- English: LS test-clean 2.12%, test-other 4.56% (M2 Pro 32GB, macOS 26.5.1), beats Whisper Small (3.74/7.95); Parakeet v2 CoreML int8 2.01/3.40; v3 2.51/4.28. ~12-40x RT across engines, SpeechAnalyzer ~3x Whisper Small, no precise table. https://lyonesse.app/blog/apple-speech-api-benchmark.html and round 2 (vendor blog: app seller; transcripts published, Whisper column reproduces OpenAI numbers).
- Argmax: M4 Mac mini macOS 26 beta, long-form conversational: SpeechTranscriber error 14.0, speed factor 70x; whisper-base.en 15.2/111x; parakeet-v2 (Pro) 11.7/359x. https://www.argmaxinc.com/blog/apple-and-argmax (competitor-authored).
- Locales: ~30 (blog claims). Vietnamese NOT confirmed. Could not retrieve Apple's supportedLocales list; a Stack Overflow list showing vi-VN is for the legacy SFSpeechRecognizer. UNVERIFIED: test `SpeechTranscriber.supportedLocales` locally on this Mac (macOS 27 here). Requires Swift/pyobjc; no Python wheel; 570-point HN thread notes English-only and "Whisper is wrong baseline": https://news.ycombinator.com/item?id=48894752
- Verdict: could replace parakeet only for English if wanting zero-download; no vi.

## Vietnamese head-to-head table (FLEURS vi test unless noted)
| Model | WER | Speed on Mac | Source |
|---|---|---|---|
| QUAN-Transcribe (brief) | 6.87 author avg; 5.45 on dub | 156x M1 Pro | brief |
| Qwen3-ASR-1.7B | 5.55 (paper) / 4.91 (M5 Air run) | 9.6x M5 Air; 13x (mlx, chip?) | arXiv, whispernotes |
| Qwen3-ASR-0.6B | 8.52 | 34-55x (mlx 4-bit, chip?) | arXiv, PyPI |
| Whisper large-v3-turbo | 8.10 (M5 Air run) | 2.6x there | whispernotes |
| Nemotron 3.5 streaming | 11.2-13.6 | n/a vi | HF card |
| Moonshine tiny-vi | 13.2 | CPU, tiny | HF GGUF card |
| Cohere Transcribe | ? | 36x CoreML M4 Pro, ~67x MLX INT5 (chip?) | HF, soniqo |
Note: FLEURS read-speech vs the user's 47-min dub; not comparable to the 5.45/2.75 numbers. Needs own test on the dub.

## Per-sub-area counts
- (a) ANE/Core ML: GEM 2 (FluidAudio v3/ultra speedup; FluidAudio ultra accuracy), MEH 3 (Phonon-2, Nemotron CoreML, WhisperKit), NOISE 1 (Argmax Pro)
- (b) MLX: GEM 2 (Qwen3-ASR MLX; mlx-audio harness), MEH 2 (Cohere MLX, Granite)
- (c) New families: GEM 2 (Cohere Transcribe coverage incl vi; transcribe.cpp runtime), MEH 5, NOISE 2
- (d) Apple native: GEM 0, MEH 2 (ran 2 extra searches; vi support still unconfirmed)

## Recommended actions (tests, not decisions)
1. Local M1 Pro bench: `fluidaudiocli` ultra vs v3 vs parakeet-mlx on a 47-min non-vi file (RTFx + WER vs mlx-whisper). Expect 5x+ speedup.
2. Local M1 Pro bench: mlx-qwen3-asr 1.7B (8-bit encoder) on the 47-min Vietnamese dub vs QUAN 5.45 / Whisper 2.75; check fillers/English names.
3. Check Apple `SpeechTranscriber.supportedLocales` for vi locally (1 min).
4. Cohere vi WER: run mlx-audio Cohere on same dub if (2) disappoints.

## Unresolved
- Open ASR multilingual tab values (JS, not fetched); Cohere per-language vi WER (image); RTFx on M1 Pro for every Core ML/MLX candidate (all published on M2 Pro-M5); chip for mlx-qwen3-asr and soniqo numbers; Moonshine vi licence; FluidAudio LICENSE spdx; Apple vi support; Qwen-Audio-3.0 weights; Python 3.14 wheel support for mlx-audio/mlx (not tested).
- Social-only claims (UNVERIFIED): Phonon-2 644x 1-h RTFx (Alex_tra_memory tweet); Argmax Pro SDK 3 Qwen3-ASR claims; Reddit snippet "Qwen3-ASR vi 5.1 vs Whisper 9.8" (search snippet only, not fetched).

## Sources
- https://hf-audio-open-asr-leaderboard.hf.space/
- https://arxiv.org/html/2510.06961v4
- https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md
- https://github.com/FluidInference/FluidAudio/releases
- https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml
- https://huggingface.co/FermionResearch/Phonon-2
- https://x.com/Alex_tra_memory/status/2109868778615976113
- https://x.com/argmax/status/2082875040807203061
- https://github.com/argmaxinc/argmax-oss-swift/discussions/243
- https://www.argmaxinc.com/blog/apple-and-argmax
- https://lyonesse.app/blog/apple-speech-api-benchmark.html
- https://lyonesse.app/blog/parakeet-moss-apple-speech-benchmark.html
- https://news.ycombinator.com/item?id=48894752
- https://news.ycombinator.com/item?id=46954049
- https://arxiv.org/html/2601.21337v1
- https://pypi.org/project/mlx-qwen3-asr/
- https://github.com/drguptavivek/qwen3-asr-mlx-runtime/blob/main/docs/performance.md
- https://whispernotes.app/blog/qwen3-asr-vs-whisper
- https://github.com/Blaizzy/mlx-audio
- https://docs.cohere.com/docs/transcribe
- https://huggingface.co/BarathwajAnandan/cohere-transcribe-03-2026-CoreML-6bit
- https://soniqo.audio/guides/cohere-transcribe
- https://github.com/neo773/cohere-transcribe-mlx
- https://www.typewhisper.com/en/addons/cohere-local/macos/
- https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- https://huggingface.co/nvidia/canary-1b-v2
- https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc
- https://huggingface.co/myflow-ai/moonshine-tiny-vi-gguf
- https://github.com/moonshine-ai/moonshine
- https://deepwiki.com/moonshine-ai/moonshine/8.3-language-support
- https://github.com/handy-computer/transcribe.cpp
- https://github.com/microsoft/VibeVoice
- https://arxiv.org/html/2609.07549v2
- gh repo view: FluidInference/FluidAudio, argmaxinc/argmax-oss-swift, Blaizzy/mlx-audio, moonshine-ai/moonshine, moona3k/mlx-qwen3-asr, handy-computer/transcribe.cpp, argmaxinc/OpenBench (stars/pushedAt as of 2026-10-16)
