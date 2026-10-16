# ZH gather: fast multilingual ASR / tagging / diarization on Apple Silicon (Q3+Q4)

Date 2026-10-16. Language: Simplified Chinese queries. Compare against brief.md (QUAN-Transcribe 156x M1 Pro, parakeet ~25x, CED-small sherpa-onnx CPU, TitaNet-small).

## Bottom line
- No Chinese-community model beats QUAN-Transcribe for Vietnamese speed+accuracy on M1 Pro. Nothing tested on M1/M2 with Vietnamese in the ZH sources.
- Best new Vietnamese-capable candidate: Qwen3-ASR-1.7B (FLEURS vi WER 5.55, official report) via mlx-qwen3-asr. But speed ~13x RTF-equivalent on M4 Pro (RTF 0.077 at 10 s) vs QUAN 156x. Accuracy ~ QUAN author's own 7.01% avg. No gain.
- SenseVoice-Small: very fast on Mac (118x on 27 min Chinese, M4 Pro, MLX) but NO Vietnamese. Useful only as zh/yue/ja/ko/en fast lane, not for framesift's Vietnamese gap.
- Fun-ASR-MLT-Nano (800M, vi supported, Apache): vi FLEURS WER 7.75 per tech report, WORSE than Whisper-large-v3 (6.51). MLX port exists but young. Zhihu claim "nano still beats Whisper" is wrong for vi.
- Dolphin: vi supported, Apache, but vi FLEURS WER 13.0-13.8 (small/large) = bad. Reject.
- Dasheng/CED: Chinese community adds nothing new; CED is already what framesift uses. ced.cpp (ggml, MIT) is a CPU alternative to sherpa-onnx, no Mac numbers.
- Diarization: sherpa-onnx docs RTF table says NeMo TitaNet (INT8) 0.110 vs 3D-Speaker INT8 0.241. Current TitaNet-small choice is the faster one. No reason to switch to CAM++/3D-Speaker for speed.
- New unexpected find: macOS 26 SpeechTranscriber (Apple on-device ASR on ANE). Chinese-dev measured it. Vietnamese support unknown, UNVERIFIED for framesift use.

## Coverage counts (GEM / MEH / NOISE)
| sub-area | GEM | notes |
|---|---|---|
| (a) Qwen3-ASR / SenseVoice / FunASR on Mac | 6 | Whisper Notes bench, mlx-qwen3-asr bench, funasr-mlx, fun-asr-nano-mlx RTF, madeye CoreML ANE, voice-notes macOS26 ANE PR |
| (b) other ZH ASR with Vietnamese | 4 | Fun-ASR tech report vi table, Qwen3-ASR report vi, Dolphin arXiv vi table, FireRed (no vi) |
| (c) Dasheng/CED tagging | 2 (below 3; 2 extra searches run) | ced.cpp/ced-gguf, CED HF card; Chinese forums essentially silent. Documented gap. |
| (d) diarization 3D-Speaker/CAM++ | 3 | sherpa RTF table via issue 3239, bili-scribe issue 47, sherpa PR 4021 batching |

## (a) Qwen3-ASR / SenseVoice / FunASR on Mac

### GEM: Whisper Notes benchmark, M4 Pro 32 GB (vendor blog, Chinese)
https://whispernotes.app/zh/blog/sensevoice-fastest-cjk-transcription
- 5 min English: Parakeet V3 2.91 s (103x), SenseVoice 5.8 s (52x), Whisper-large-v3-turbo 20.92 s (14.3x).
- 27 min Chinese: SenseVoice 13.83 s (118x), Whisper turbo 124 s (13.1x), Qwen3-ASR via sherpa-onnx CPU 224 s (7.2x, removed from app).
- SenseVoice via MLX GPU, 827 MB download, load 0.81 s. Vendor-run, single chip, no accuracy set for their audio. CommonVoice CER quoted from FunAudioLLM paper: zh 10.78 vs Whisper-v3 12.55; ko 8.28 vs 5.59; en 14.71 vs 9.39. SenseVoice = zh/yue/en/ja/ko only.
- Takeaway: Qwen3-ASR via sherpa-onnx CPU is slow (7x). MLX is the right runtime.

### GEM: mlx-qwen3-asr (moona3k), M4 Pro 48 GB, v0.4.x
https://github.com/moona3k/mlx-qwen3-asr , docs/BENCHMARKS.md
- 229 stars, Apache-2.0, pushed 2026-09-20. PyPI mlx-qwen3-asr 0.4.4, requires-python >=3.10 (3.14 untested, UNVERIFIED). CLI `mlx-qwen3-asr audio.wav -f srt`. Pre-quantized 4/8-bit HF repos (0.6B 4-bit 517 MB, 1.7B 4-bit 1.2 GB).
- 0.6B fp16 10 s clip 0.30 s (RTF 0.029 = ~34x); 4-bit 0.17 s (RTF 0.018 = ~55x). 1.7B fp16 10 s: 0.73 s (RTF 0.077 = ~13x).
- Accuracy: LibriSpeech test-clean 2.33% (0.6B), 1.94% (1.7B), n=100 samples only. FLEURS 10 langs x 10 samples: 0.6B 9.54%, 1.7B 6.70% (vi NOT in their 10 languages).
- Long-form 80 s: 0.6B mean 3.75 s. MLX is 4-7x faster than official PyTorch CPU on same Mac.
- Streaming mode, `--mic`, OpenAI-compatible server, diarize flag mentioned in acknowledgments.
- Caveat: tiny samples; M4 Pro is ~2x an M1 Pro on GPU, so expect M1 Pro slower. UNVERIFIED on M1.

### GEM: funasr-mlx (baicai-1145), MLX ports of FunASR zoo
https://github.com/baicai-1145/funasr-mlx , PyPI funasr-mlx 0.2.1 (Apache-2.0 on PyPI, GitHub license "other"), requires-python >=3.11. 0 stars, pushed 2026-10-14. Very young, single author.
- Ports: SenseVoiceSmall, Paraformer, Whisper large-v3 (RTF 0.058) / turbo (RTF 0.061), emotion2vec+, ct-punc, fsmn-vad, CAM++, Fun-ASR-Nano, Qwen3-ASR-0.6B, GLM-ASR-Nano, MOSS-Transcribe-Diarize. LLM-ASR pipeline-factory registration "pending" per README.
- Qwen3-ASR-0.6B w4 resident worker: RTF 0.079 on M4 Air (README). Fun-ASR-Nano MLX w4: RTF 0.104 (9.6x realtime), chip unstated (HF card https://huggingface.co/baicai1145/fun-asr-nano-mlx).
- Note: Whisper turbo RTF 0.061 here (16x) is comparable to mlx-whisper ~ framesift current.
- Verdict MEH for adoption (maturity), GEM as a datapoint.

### GEM: madeye/sensevoice-mlx (despite name = CoreML for ANE)
https://github.com/madeye/sensevoice-mlx
- SenseVoice-Small encoder (~234M, 70 SANM layers) rewritten ANE-style, CoreML fp16, cos-sim >0.998 vs ONNX, 1-30 s via EnumeratedShapes. Apache-2.0, 3 stars, last push 2026-03-05. NO speed numbers in README (fetched). Only CJK+en anyway. Interesting as a pattern for QUAN-like CoreML conversion, not as a model.

### GEM: macOS 26 SpeechTranscriber on ANE (Chinese dev PR, measured)
https://github.com/SoulZhong/voice-notes/pull/246 (merged 2026-10-05)
- M5 Pro, 10 min Chinese meeting, via same pipeline: Apple SpeechTranscriber CPU time 1.8 s vs SenseVoice 70 s; mem 140 MB vs 1.3 GB; per-segment latency <=0.25 s both. Chinese accuracy "same tier". Fewer punctuation, no language label. Needs language pack download.
- Swift API; 3 framework traps listed (must AssetInventory.reserve, Int16 input, check lang pack). Vietnamese availability UNVERIFIED (not in sources). Worth a one-line check by lead: `SpeechTranscriber.supportedLocales` for vi_VN. Needs macOS 26 (host here is Darwin 27, fine).

### MEH
- cnblogs xio1028 (https://www.cnblogs.com/xio1028/p/19011196): phone-call audio, SenseVoice fastest but least accurate vs whisper-large-v3. Qualitative only.
- Zhihu 2039392717323620590 "2026 实测主流 ASR": 16 s clip on RTX 4060 Ti / 5090, Dialect-focused (Qwen3-ASR 1.5-3 s, Dolphin-372M 0.57 s, Fun-ASR-Nano 2 s). No Mac, no vi.
- linux.do t/1755978: Qwen3-ASR on 4060 Ti gets only RTF ~1 (no vLLM). Confirms naive PyTorch Qwen3-ASR is slow; Mac numbers need MLX port.
- V2EX / segmentfault: no relevant Mac ASR benchmark threads found (V2EX sov2ex timed out; one V2EX thread about hosting qwen3-asr on EdgeOne = NOISE). SegmentFault: nothing on topic.

### NOISE
CSDN/devpress "Qwen3-ASR Mac M2/M3 Metal 保姆级" articles (templated, no numbers).

## (b) Other Chinese ASR with Vietnamese

| Model | Vi? | Vi accuracy (set) | Size | License | Mac path | Verdict |
|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | yes | FLEURS vi WER 5.55 (Flash API 3.64); 0.6B 8.52 (arXiv 2601.21337 appendix A.2b) | 1.7B (3.4 GB fp16; 1.2 GB 4-bit) | Apache-2.0 | mlx-qwen3-asr, funasr-mlx, sherpa-onnx CPU | GEM data. Accuracy good, speed ~13x M4 Pro: slower than QUAN. Better WER than Whisper turbo? Not directly compared on vi. |
| Fun-ASR-MLT-Nano-2512 | yes (31 langs) | Fun-ASR-ML-Nano FLEURS vi 7.75 vs Whisper-v3 6.51, dolphin-small 15.62, seamless 8.07, Fun-ASR-ML (big, closed) 5.50 (arXiv 2509.12508 Table 7) | 800M | Apache-2.0 | MLX port young (funasr-mlx); official FunASR PyTorch GPU | MEH. HF card says per-language results for MLT checkpoint not published; report's number is for "ML-Nano" and may differ. |
| Dolphin base/small | yes (40 Eastern langs) | vi FLEURS WER small 13.8, large 13.0; CommonVoice17 vi small 16.3 (arXiv 2503.20212 tables 7-8) | base 0.1B, small 0.4B | Apache-2.0 | `pip install dataoceanai-dolphin`, CUDA oriented, no MLX/CoreML found | NOISE for vi. Sees 2-3x worse than Whisper-v3 (8.9) on vi. |
| SenseVoice-Small | NO | zh/yue/en/ja/ko only | 234M | HF license "other" (FunASR model license) / repo MIT | MLX, sherpa-onnx, CoreML-ANE | irrelevant for vi |
| FireRedASR2S (Xiaohongshu) | NO ASR vi (zh/en/dialects). FireRedLID/VAD cover 100+ langs | n/a | AED 1B+, LLM 8B+ | Apache-2.0 | none for Mac | NOISE for vi. FireRedLID (100+ langs LID) is a possible whisper-tiny LID alternative: UNVERIFIED size/speed. |
| Kimi-Audio, Step-Audio 2 | no vi | n/a | 7B-class | open | no Mac path | NOISE |
| Paraformer | zh only | - | - | MIT | - | irrelevant |

Maturity: Dolphin repo 793 stars, last push 2026-06-11; Fun-ASR repo 1576 stars, push 2026-10-12; SenseVoice 9514 stars MIT push 2026-09-30; FunASR 20.7k stars push 2026-10-13; FireRedASR2S 716 stars push 2026-06-02; Qwen3-ASR-0.6B HF 678k downloads.

Note on FunASR site claims: funasr.com blog says "on CPU use SenseVoice"; SenseVoice 169.6x / 7.81% CER on 184 Chinese files is H100 not Mac.

## (c) Dasheng / CED tagging
- GEM: ced.cpp (https://github.com/mudler/ced.cpp, MIT, 107 stars, push 2026-10-04) ggml port; GGUF in https://huggingface.co/mudler/ced-gguf (tiny/mini/small/base, f16 + q8_0; 15.6k downloads). Claims f16 ~1.55x faster than PyTorch on CPU, q8_0 6.5x less memory; LocalAI lists "527 AudioSet classes tagged in about 55 ms" (chip unstated, UNVERIFIED). PR #2 adds any-ggml-GPU-backend (Metal would follow, not measured). Compare: framesift CED-small int8 sherpa-onnx 3 min audio in 4.8-7.4 s. No evidence ced.cpp beats it; skip unless Metal numbers appear.
- GEM-lite: CED model card (https://huggingface.co/mispeech/ced-small, Apache-2.0): CED-Tiny 5.5M params AS-2M mAP 48.1; Mini 9.6M 49.0; Small 22M 49.6; Base 86M 50.0. Original RicherMans/CED repo is GPL-3.0 (weights Apache-2.0 on HF). Tiny/mini offer a speed option; accuracy drop small (49.6 -> 48.1 mAP).
- MiDashengLM-7B (Xiaomi, Apache-2.0, 445 stars, https://github.com/xiaomi-research/dasheng-lm): 7B audio-captioning LLM, vLLM/CUDA only. Xiaomi roadmap says end-device offline deploy not done. MiDashengLM-Spatial released 2026-10-15 (7B, spatial). Too big for 16 GB M1 Pro. Not a tagger replacement.
- Dasheng encoders (https://github.com/xiaomi/dasheng, Apache-2.0, 208 stars, push 2025-11-07) base/0.6B/1.2B are embeddings, not tagging heads in the sources found. HF mispeech/dasheng-base 9.5k downloads. No Mac numbers anywhere in ZH.
- ZH forums (zhihu, cnblogs, segmentfault, v2ex, linux.do): zero Mac benchmark for CED/Dasheng. Only generic AudioSet/PANNs explainers (NOISE). SenseVoice has built-in AED labels (laughter, applause, etc.), 8 event classes only - not 527.

## (d) Diarization on Mac (3D-Speaker / CAM++)
- GEM: sherpa-onnx RTF table quoted in issue #3239 (https://github.com/k2-fsa/sherpa-onnx/issues/3239), from official docs: pyannote-seg + NeMo TitaNet INT8 RTF 0.110; + 3D-Speaker INT8 0.241; + 3D-Speaker FP32 0.297 (single-thread CPU; chip not stated). 21 min audio on RTX 3090 with GPU build took ~5 min (clustering CPU-only, GPU slow). Takeaway: TitaNet (framesift's choice) is 2.2x faster than 3D-Speaker; CAM++ gives no speed win.
- GEM: bili-scribe issue #47 (https://github.com/ZouZhao321/bili-scribe/issues/47, 2026-09): sherpa-onnx OfflineSpeakerDiarization pyannote-seg-3.0 + 3D-Speaker CAM++ Chinese embedding, CPU RTF ~0.09 (11x), 16.6 min audio in 89 s; chip not stated. INT8 segmentation model collapsed clusters (one speaker 86%) on some recordings; fp32 fine, only ~8% slower. Recommend fixed num_clusters=2 for interviews. User-reported, UNVERIFIED but specific. Relevant risk: framesift greedy clustering + int8 models.
- GEM: sherpa-onnx PR #4021 (https://github.com/k2-fsa/sherpa-onnx/pull/4021, open Oct 2026): batches 16 pyannote windows per ONNX call on CPU for float models; byte-identical output; quantized + CUDA models excluded (issue #3853 notes dynamic quant scale changes outputs). Not merged as of fetch; would speed fp32 diarization only.
- MEH: 3D-Speaker repo (https://github.com/modelscope/3D-Speaker, 3182 stars, Apache-2.0, last push 2025-12-08 = stale). cnblogs/zhihu CAM++ posts: Chinese speech, coarse segmentation, no overlap handling (zhihu 5960358286). funasr-mlx ports CAM++ to MLX; MOSS-Transcribe-Diarize MLX (end-to-end ASR+speaker, pipeline registration pending) - watch item.
- NOISE: CSDN 3D-Speaker run-through posts.

## Maturity / how to run summary
- mlx-qwen3-asr: pip, CLI, py>=3.10, Apple Silicon only. Best-maintained Qwen3 Mac path.
- funasr-mlx: pip 0.2.1, py>=3.11, 0 stars. Models need separate download via `funasr-mlx download`.
- sherpa-onnx (15.3k stars, Apache-2.0, push 2026-10-15): wheels, supports SenseVoice, Qwen3-ASR (CPU), FireRed; CoreML provider only coarse (framesift already measured CoreML EP slower for CED).
- Dolphin: pip dataoceanai-dolphin, PyTorch/CUDA.
- Python 3.14 compatibility: no source stated 3.14 support for any of these. UNVERIFIED all.

## Conflicts / caveats
- Zhihu 1986523121008779526 says Fun-ASR nano still beats Whisper/dolphin on small languages; the report's own table shows ML-Nano vi 7.75 > Whisper 6.51 (worse). Dolphin comparison holds; Whisper claim does not.
- Qwen3-ASR vi 5.55 (1.7B) is FLEURS read speech; QUAN author's set avg 7.01% is a different benchmark. Not comparable. Neither is on framesift's dub audio.
- All Mac speed numbers are M4 Pro / M4 Air / M5 Pro. None for M1 Pro. Scale down.

## Unresolved
1. Does macOS 26 SpeechTranscriber support Vietnamese, and what WER/speed on M1 Pro? (cheap to test in Swift.)
2. Qwen3-ASR-1.7B 4-bit on M1 Pro actual RTF and vi WER on framesift's 47-min dub vs QUAN 5.45% / Whisper 2.75%. Likely ~15-25x, not better than QUAN on speed; accuracy could beat QUAN on fillers/English names (LLM-ASR keeps punctuation + casing). Only worth testing if QUAN's dropped-fillers behavior is the pain.
3. FireRedLID (100+ langs) as whisper-tiny LID replacement: size, speed, accuracy unknown.
4. ced.cpp Metal speed vs sherpa-onnx CPU: no data.
5. Fun-ASR-MLT-Nano real vi WER (HF card says no checkpoint-specific results).

## Sources
- https://whispernotes.app/zh/blog/sensevoice-fastest-cjk-transcription
- https://github.com/moona3k/mlx-qwen3-asr and /blob/main/docs/BENCHMARKS.md
- https://pypi.org/pypi/funasr-mlx/json ; https://github.com/baicai-1145/funasr-mlx ; https://huggingface.co/baicai1145/fun-asr-nano-mlx
- https://github.com/madeye/sensevoice-mlx
- https://github.com/SoulZhong/voice-notes/pull/246
- https://arxiv.org/html/2601.21337v1 (Qwen3-ASR report)
- https://arxiv.org/html/2509.12508 (Fun-ASR report) ; https://zhuanlan.zhihu.com/p/1986523121008779526
- https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512
- https://arxiv.org/html/2503.20212 (Dolphin) ; https://github.com/DataoceanAI/Dolphin
- https://www.funasr.com/en/blog/which-funasr-model.html
- https://zhuanlan.zhihu.com/p/2039392717323620590
- https://www.cnblogs.com/xio1028/p/19011196
- https://linux.do/t/topic/1755978
- https://github.com/mudler/ced.cpp ; https://huggingface.co/mudler/ced-gguf ; https://huggingface.co/mispeech/ced-small
- https://github.com/xiaomi-research/dasheng-lm ; https://github.com/xiaomi/dasheng
- https://github.com/k2-fsa/sherpa-onnx/issues/3239 ; https://github.com/k2-fsa/sherpa-onnx/pull/4021
- https://github.com/ZouZhao321/bili-scribe/issues/47
- https://github.com/modelscope/3D-Speaker
- https://github.com/FireRedTeam/FireRedASR2S ; https://arxiv.org/html/2603.10420v1
- EmbeddingGemma 2 (ZH coverage only, no ASR/audio benchmark found): https://zhuanlan.zhihu.com/p/2091009926814496699 , https://www.ithome.com/1/010/125.htm
