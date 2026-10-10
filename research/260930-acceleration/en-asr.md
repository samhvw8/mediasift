# ASR backends: Apple Silicon + CUDA (EN) — gathered 2026-09-30
Baseline: faster-whisper small, CPU, M1 Pro, 54 s / 6 min (RTF 0.15, ~6.7x). Input = 16k mono float32 numpy. Need segment (ideally word) timestamps, pip/uv.
Tags: GEM = specific numbers/hardware/failure; MEH = generic; NOISE = ignore.

## A. Repo health (gh, 2026-09-30)
| Pkg | PyPI latest (date) | Repo pushed | Note |
|---|---|---|---|
| mlx-whisper 0.4.3 | 2025-08-29 | ml-explore/mlx-examples pushed 2026-04-06, no release tags | STALE-ish; open issues #1440 (parity tests fail w/ MLX 0.32.0), #1441, #1421, #1418 (KeyError 'words' w/ translate) |
| parakeet-mlx 0.5.2 | 2026-06-05 | 2026-08-29, 984 stars | active; open #49 mx.concat incompatible w/ MLX 0.30.6 (Feb 2026) |
| mlx-audio 0.5.7 | 2026-09-28 | 2026-09-28, 7964 stars | very active; bundles whisper/parakeet/Qwen3-ASR etc |
| lightning-whisper-mlx 0.0.10 | 2024-04-02 | 2024-05-08 | DEAD; #24/#26 quantized models broken w/ newer MLX |
| pywhispercpp 1.5.1 | 2026-08-22 | 2026-08-22 | PyPI wheels CPU-only; Metal/CUDA need source build |
| whisper.cpp v1.9.4 | - | 2026-09-28, 54k stars | active; has native Parakeet TDT support now |
| faster-whisper 1.2.1 | 2025-10-31 | 2025-11-19 | quiet ~11 months; ctranslate2 4.8.2 (2026-08-31) still moving; CUDA12+cuDNN9 |
| insanely-fast-whisper 0.0.15 | 2024-05-27 | 2025-10-25 | DEAD-ish |
| distil-whisper | - | 2025-01-08 | DEAD repo (weights still usable via faster-whisper/HF) |
| nemo_toolkit 3.0.0 | 2026-08-07 | 2026-09-30 | active, heavy (torch) |
| onnx-asr 0.12.0 | 2026-07-15 | 2026-08-16, 381 stars | active, tiny deps |
| moonshine-voice 0.1.5 | 2026-08-24 | 2026-08-31, 11k stars | active |
| sherpa-onnx 1.13.8 | 2026-09-10 | 2026-09-22 | active |
| WhisperKit v1.1.0 / FluidAudio v0.17.4 | not on PyPI (Swift SDKs) | active | no pip path |
Sources: gh repo view (all); PyPI json API https://pypi.org/pypi/<pkg>/json; issues: https://github.com/ml-explore/mlx-examples/issues/1440 , https://github.com/senstella/parakeet-mlx/issues/49 , https://github.com/mustafaaljadery/lightning-whisper-mlx/issues/24 [GEM: maintenance]

## B. Findings

### F1 [GEM] Parakeet TDT has word+segment timestamps natively, runs on BOTH stacks via different packages
- NeMo: `asr_model.transcribe([...], timestamps=True)` -> char/word/segment timestamps. `pip install nemo_toolkit["asr"]`. HF card: Open ASR Leaderboard mean WER 6.05, RTFx 3386 (v2, batch, big GPU). https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- parakeet-mlx: `AlignedResult -> sentences -> tokens` with start/end (token-level; `--highlight-words`), default model parakeet-tdt-0.6b-v3, auto chunking 120 s w/ 15 s overlap, beam option. https://github.com/senstella/parakeet-mlx
- onnx-asr: pip, takes numpy arrays directly, token-level timestamps (`with_timestamps`), providers CUDA/TensorRT/CoreML/CPU. https://github.com/istupakov/onnx-asr

### F2 [GEM] onnx-asr RTFx table (Parakeet TDT 0.6B v2/v3 English; RTX=5070 Ti; T4 Colab; CPU=Ryzen 9800X3D)
- CPU x64: 35-37x (int8 ~31x). Arm Cortex-A53 CPU: 1.0x.
- CUDAExecutionProvider: RTX 5070 Ti 88.7 (v2) / 91.4 (v3); T4 57.6/57.5.
- TensorRT: RTX 5070 Ti 298 (fp32) / 329 (fp16) v2; T4 237 fp16.
- Whisper large-v3-turbo via onnx-asr: CPU 3.9x; CUDA RTX 54.8 (fp32)/76 (fp16); T4 9.2/25.9. Whisper base CUDA RTX 108.
- No Apple/CoreML numbers published there. https://istupakov.github.io/onnx-asr/benchmarks/
- Implication: x64 CPU parakeet (36x) already ~5x faster than our M1 Pro faster-whisper-small (6.7x) IF M1 CPU ORT ~ comparable (UNVERIFIED on M1).

### F3 [GEM] onnx-asr Parakeet hard length cliff
- ~300 s OK, ~600 s -> ORT "broadcast error 2624 by 7624" (fixed-size rel-pos buffer, >~5000 encoded frames). README says "20-30 s for most models" (understates). Workaround: chunk ~280 s windows at quiet frames + offset timestamps, or use VAD. Our 6 min = 360 s -> MUST chunk/VAD. Same wall in HF transformers#43383. https://github.com/istupakov/onnx-asr/issues/131 (open, 2026-07-11)
- onnx-asr has no timestamps for Whisper/Canary yet (#115, #116) -> only Parakeet/CTC/RNNT models have them.

### F4 [GEM] Apple M5 Pro benchmark (LibriSpeech test-clean first 200 utts), same harness, isolated processes
- Parakeet TDT v3 INT8: WER 2.37, RTF 0.009 (117x). Qwen3-ASR 0.6B MLX 4/5/8-bit: 2.20/1.74/1.82 WER, 70-86x (RTF 0.012-0.015). Qwen3-ASR 1.7B MLX 5-bit: WER 1.32, 36x. WhisperKit large-v3-turbo FP16: WER 1.71, 11.9x. Nemotron streaming INT8 17x. Sustained 42 min NE load: no throttle.
- Caveat: vendor (soniqo/aufklarer, Swift library) numbers, M5 Pro, NOT M1. https://soniqo.audio/benchmarks
- Also Reddit r/LocalLLaMA claim Qwen3-ASR at 40x RT on MLX: https://www.reddit.com/r/LocalLLaMA/comments/1rz920q/we_beat_whisper_large_v3_on_librispeech_with_a/

### F5 [GEM] M1-specific numbers (closest to our hardware)
- HN commenter: Parakeet v3 25x RT vs Whisper Turbo 14x RT on M1; CER 3.8% vs 4.0% on BBC news audio. https://news.ycombinator.com/item?id=47259210 (anecdotal; impl unnamed)
- FluidAudio (Swift/CoreML ANE): ~155x on M1; MLX-GPU path 300x (chip unstated) ; ~190x on M4 Pro (1 h in ~19 s); 110x on M4 Pro v3 per HF card. Blog = vendor marketing; contradicts itself (M1 155x vs HF 110x M4 Pro for v3 vs v2). https://macparakeet.com/blog/whisper-to-parakeet-neural-engine/ , https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml
- FluidAudio PR text: "M1 Mac ~28x (clean) ~25x (other)" (conflicts w/ 155x above; that line may be for a different model/pipeline — unresolved). https://github.com/FluidInference/FluidAudio/pull/907
- Oliver Wehrens: Whisper large (openai pytorch-mlx port) 10 min audio on M1 Pro 16-core GPU 216 s (2.8x); RTX 4090 same file 186 s stock whisper but 8 s w/ insanely-fast-whisper (flash attn, batched) -> 75x. Old (Dec 2023). https://owehrens.com/whisper-nvidia-rtx-4090-vs-m1pro-with-mlx/ ; HN https://news.ycombinator.com/item?id=38628184 (same 4090 8 s claim in comments)

### F6 [GEM] mlx-whisper vs whisper.cpp (same machine, Jan 2026 re-test, large-v3-turbo, MLK long wav; chip not named in snippet, earlier post says M1 Max)
- whisper.cpp 26.7 s +/-0.6 vs mlx-whisper 13.1 s +/-0.3 => mlx 2.03x faster (earlier run 1.78x). https://notes.billmill.org/dev_blog/2026/01/updated_my_mlx_whisper_vs._whisper.cpp_benchmark.html
- Counter: Josh Marshall (14" M1 Pro) same model quantized 8-bit large-v2: whisper.cpp 1m vs mlx 1m36 -> whisper.cpp faster. So "mlx 3x faster" claim was apples-to-oranges (distilled vs full). https://simonwillison.net/2024/Aug/13/mlx-whisper/
- Reddit: mlx-whisper 12 min audio in <18 s w/ large-v3-turbo on M3 Max (~40x). https://www.reddit.com/r/LocalLLaMA/comments/1ftuq9i/whisper_turbo_vs_whisper_mlx/

### F7 [MEH] JustVoice M1->M4 table (whisper.cpp Metal; blog, methodology vague): M1 8-core: small q5 ~6x, medium ~2.5x, large-v3 ~1.0x; M2 Pro: small 11x, large-v3 2.5x; M4: small 12x, large-v3 2.6x. Metal gives 1.4-1.9x over CPU. => whisper.cpp small on M1 ~6x ~= our current faster-whisper CPU 6.7x: NO GAIN. https://justvoice.ai/blog/whisper-benchmark-apple-silicon-m3-m4
Whisper.cpp ANE not used unless CoreML encoder built manually (WHISPER_COREML=1). https://github.com/absadiki/pywhispercpp (pip wheels CPU only; `WHISPER_COREML=1 pip install git+...` needed; README offers no CUDA line; BBC-Esq noted 2024 no CUDA python binding for .cpp https://github.com/m-bain/whisperX/discussions/1352)

### F8 [GEM] CUDA faster-whisper numbers
- Official README (RTX 3070 Ti 8GB, 13 min audio, large-v2 fp16 beam5): openai/whisper 2m23s; whisper.cpp FA 1m05s; faster-whisper 1m03s (12x); batch_size=8 17s (46x); int8 batch8 16s. distil-large-v3 batch16: 25m50s on long YT set. https://github.com/SYSTRAN/faster-whisper
- Third party (gigagpu.com, RTX 4090, single stream): large-v3-turbo INT8 80x, FP16 62x; small INT8 140x; WhisperX batched 16 turbo = 175x; WhisperX adds wav2vec2 alignment ~0.4-0.6x audio duration -> 80x => ~55x w/ word timestamps. Vendor blog, unverified. https://gigagpu.com/rtx-4090-24gb-whisper-benchmark/
- insanely-fast-whisper 4090 8 s/10 min (75x) large-v3 (HN); repo dead since 2024-05 (0.0.15). https://news.ycombinator.com/item?id=38629951
- faster-whisper CUDA setup pain: only CUDA12+cuDNN9 on latest ctranslate2; older stacks need ctranslate2 pins (4.4.0 / 3.24.0). https://github.com/SYSTRAN/faster-whisper
- faster-whisper has NO Metal/MPS backend (CPU on Mac only; ctranslate2 Accelerate). Word timestamps: `word_timestamps=True` (cross-attn DTW).

### F9 [GEM] whisper.cpp now ships Parakeet TDT (C API/parakeet-cli) + independent parakeet.cpp (mudler) on ggml CUDA/Metal/Vulkan
- DeepWiki: include/parakeet.h, src/parakeet.cpp. Model ggml-parakeet-tdt-0.6b-v3-f16.bin. https://deepwiki.com/ggml-org/whisper.cpp/2.5-parakeet-model-support ; https://github.com/mudler/parakeet.cpp (GGUF f16/q8/K-quants, CUDA/Metal/Vulkan/HIP)
- Issue #4088 "Hallucination with parakeet" (closed): log shows CUDA driver insufficient -> fell back to CPU, output "tamb tamb..." on one noisy file; encode 938 ms; user-error/edge case, but shows Parakeet can hallucinate repetition. https://github.com/ggml-org/whisper.cpp/issues/4088
- pywhispercpp exposure of parakeet: UNVERIFIED (likely not).

### F10 [GEM] MLX has a CUDA backend: `pip install "mlx[cuda]"` (Linux)
- First version on PyPI; "works for many common cases with notable exceptions". Means mlx-whisper / parakeet-mlx / mlx-audio are nominally portable to NVIDIA Linux, but UNTESTED for ASR; no published speeds. https://github.com/ml-explore/mlx/discussions/2422 ; https://pypi.org/project/mlx/ . MLX itself v0.32.3 (2026-09-29). mlx-whisper/parakeet-mlx pin issues vs MLX versions recur (#49 concat, #1440 parity) -> version-pin risk.

### F11 [GEM] Reliability bugs on mlx-whisper
- Long files: stuck repeating same sentence for minutes (workaround repo 1c7/mlx-whisper-long; chunking / condition_on_previous_text=False / hallucination_silence_threshold). https://github.com/1c7/mlx-whisper-long
- Hangs: clip_timestamps infinite hang (#1256 workaround: duration-0.1), MP3 stall (#958). https://github.com/ml-explore/mlx-examples/issues/1256 , /958
- Numpy array input: mlx_whisper.transcribe accepts ndarray/str path (known API; UNVERIFIED this session).
- lightning-whisper-mlx: hallucinations "Thanks for watching!" (#15), quantized broken (#24/#26), no word timestamps documented; claims "10x whisper.cpp, 4x mlx" never reproduced (Bill Mill/Marshall comparisons show ~2x at best). https://github.com/mustafaaljadery/lightning-whisper-mlx

### F12 [GEM] Accuracy tradeoffs (English)
- Parakeet TDT 0.6B v2 EN: leaderboard mean WER 6.05 (v3 6.32 per whispernotes) vs Whisper large-v3-turbo 7.83 (same source). Canary-Qwen 2.5B #1 (5.63, GPU only, ~8x slower than Parakeet); IBM Granite speech 3.3 8B 5.85 at RTFx ~31. https://whispernotes.app/blog/parakeet-v3-default-mac-model ; https://www.reddit.com/r/LocalLLaMA/comments/1rm36ew/whats_the_best_local_asr_model_for_realtime/
- Contradiction: HN user saw Whisper better than Parakeet on Euro-English meetings / noisy/accented; Parakeet better on clean read speech. Lecture audio is usually clean => Parakeet likely fine, needs spot check. https://news.ycombinator.com/item?id=44380330
- Parakeet: no hallucinated loops usually, outputs nothing on silence, word timestamps from frames not decoder estimates (rss-to-whisper issue #101). https://github.com/baz8080/rss-to-whisper/issues/101
- Parakeet punctuation+caps: v2 yes; segment splitting is by sentence punctuation (good for segments). Timestamps granularity: 80 ms frames (subsampling 8).
- Moonshine (moonshine-voice 0.1.5, 11k stars): built for low-latency streaming/edge, CPU/ONNX; not a throughput play for GPU; no measured Mac/RTX numbers collected. https://github.com/moonshine-ai/moonshine [MEH]
- Kyutai stt-2.6b-en: pip `moshi`, word timestamps script, CUDA server (`cargo install --features cuda`), MLX variant exists (stt-1b-en_fr-mlx) but 2.6b is CUDA/PyTorch; repo last push 2026-01-26; streaming/realtime (~1x by design) [MEH for batch]. https://github.com/kyutai-labs/delayed-streams-modeling
- Distil-whisper: EN-only, ~6x faster than large-v3 at ~1% WER cost; repo dead 2025-01; usable as `distil-large-v3` in faster-whisper and lightning/mlx. [MEH]
- Qwen3-ASR (0.6B/1.7B): best WER in MLX (1.32-1.82 on LS clean), 36-70x on M5 Pro via mlx-audio/soniqo; CUDA via PyTorch/transformers (UNVERIFIED speeds); timestamps need separate forced aligner (UNVERIFIED). [MEH until verified]

### F13 [NOISE/MEH] skipped
- RunAnywhere MetalRT "714x" https://www.runanywhere.ai/blog/metalrt-speech-fastest-stt-tts-apple-silicon (proprietary, not pip).
- promptquorum "large-v3 10x RT on M5 Pro" (contradicts JustVoice M4 2.6x; SEO). naveedn/whisper-bench README example output (fabricated sample numbers). gigagpu/powergpu/vexascribe/inferencebench/H100 597x (GPU-vendor SEO, other than as flagged).
- Reddit search_reddit/HN/Lobsters keyword search mostly empty/irrelevant; Lobsters hits were old (Whisper 3080 medium ~2x RT, 2022). Elite-forum coverage thin: HN items used: 38628184, 38629951, 47259210, 44380330; Reddit: 1ftuq9i, 1rm36ew, 1rz920q.

## C. Estimate for our workload (360 s audio), extrapolated — NOT measured
- Current: 54 s (M1 Pro CPU, faster-whisper small).
- Parakeet-mlx on M1 Pro, if ~25x (HN M1): ~14 s; if 50-100x (MLX GPU): 4-7 s; model-load 1-3 s extra (HF download ~1.2 GB first time).
- mlx-whisper large-v3-turbo on M1 Pro: ~14x?? (JustVoice M1 large ~1x whisper.cpp; HN M1 turbo 14x) -> ~26 s; mlx-whisper small likely ~15-25x -> 15-25 s.
- CUDA: faster-whisper small/turbo int8 on any RTX 30/40: <5 s; Parakeet via onnx-asr CUDA on RTX 5070 Ti ~91x => ~4 s; TensorRT ~300x but setup heavy.

## D. Ranked recommendation
### (a) Apple Silicon
1. parakeet-mlx (pip, numpy-capable via mlx arrays/load audio path; word/token timestamps; 10x+ over Whisper-turbo per anecdotes; active; English WER better than Whisper turbo on clean audio). Pin mlx. Keep default 120 s chunk/15 s overlap (or chunk yourself; issue #43 abnormal timestamp jump seen Jan 2026, closed). Risk: MLX version breakage (#49), model-specific quality on noisy audio.
2. mlx-whisper (large-v3-turbo or small) — pragmatic fallback: same Whisper output/word_timestamps (DTW), ~2x whisper.cpp; beware long-file loop bug -> condition_on_previous_text=False, chunk to <=30 s w/ VAD; package stale (0.4.3, Aug 2025), mlx-examples open parity issue vs MLX 0.32.
3. onnx-asr + Parakeet (CoreML/CPU provider) — only if you want ONE code path for both platforms; CoreML provider speed on M1 UNVERIFIED; needs own chunking (300 s cliff).
Avoid: lightning-whisper-mlx (dead), whisper.cpp via pywhispercpp (no gain vs current on M1: ~6x small; wheels CPU-only, Metal/CoreML need source build), WhisperKit/FluidAudio (Swift, not pip, though fastest on ANE: 155x M1 claimed).
### (b) CUDA
1. NeMo Parakeet (nemo_toolkit[asr], timestamps=True) — best speed/WER; heavy torch dep. Lighter alternative: onnx-asr[gpu] + Parakeet (CUDA 91x RTX 5070 Ti, TensorRT ~300x) + own chunking at <=280 s.
2. faster-whisper (already in use; just device="cuda", compute_type="float16"/"int8_float16", BatchedInferencePipeline batch 8-16 => 46x on 3070 Ti; turbo int8 ~80x on 4090) with word_timestamps=True. Zero new deps; needs CUDA12+cuDNN9; repo quiet since Oct 2025.
3. WhisperX (v3.8.6 2026-05-25) if you want batched + forced-alignment word timestamps (accurate word times, ~55x on 4090).
Avoid: insanely-fast-whisper (dead), pywhispercpp for CUDA.
### Cross-platform single abstraction suggestion
Backend interface with 2-3 impls: `parakeet-mlx` (mac), `faster-whisper` (CUDA + CPU fallback), optional `onnx-asr` parakeet. Keep faster-whisper CPU as universal fallback (already works).

## E. Unresolved
1. Real parakeet-mlx RTF on M1 Pro (not just M5 Pro / unnamed M1 HN anecdote). Needs 6-min lecture run: parakeet-mlx v3, mlx-whisper small + turbo, onnx-asr CoreML.
2. Does parakeet-mlx accept a numpy/mx array directly (API `model.transcribe(path)` only?) — else write temp WAV.
3. Word-level vs token-level: parakeet-mlx tokens are subword; word merge needed? (`--highlight-words` implies merge exists).
4. onnx-asr CoreML provider actually usable/fast for Parakeet on M1? (dynamic shapes are a known ORT-CoreML weak spot — not verified.)
5. mlx[cuda] real ASR performance/compatibility with parakeet-mlx; nobody published.
6. pywhispercpp Parakeet exposure; whisper.cpp parakeet speeds on Metal/CUDA unpublished.
7. FluidAudio "M1 ~28x" vs "155x" discrepancy.
8. Qwen3-ASR timestamps and CUDA speeds; Kyutai/Moonshine batch speeds; any other 2026 models (Cohere Transcribe, Voxtral etc.) not investigated.
9. Accuracy on actual lecture audio (accents, jargon) parakeet vs whisper — need WER spot check.
10. faster-whisper maintenance: ~11 months since release; check CUDA 13/Blackwell compat (search returned nothing).
