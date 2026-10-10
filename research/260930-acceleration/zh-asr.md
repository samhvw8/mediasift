# zh-asr: Chinese-community ASR findings (Apple Silicon + CUDA) -- gathered 2026-09-30

Tags: GEM = decision-relevant, MEH = context, NOISE = ignore. Budget used: ~20 searches, ~14 fetches, ~10 gh calls. Elite-forum coverage was thin (V2EX/linux.do/SegmentFault returned almost nothing relevant; search engines mostly down: duckduckgo blocked, mojeek/mwmbl/marginalia cooling). Zhihu + cnblogs + GitHub issues carried the signal.

## Headline
1. SenseVoice-Small is by far the fastest measured on Mac (RTFx 118-289) but has NO usable word timestamps and only zh/yue/en/ja/ko (no Vietnamese). Fails the framesift "word timestamps + multilingual" bar. Use only as a fast zh/ja/ko/en segment-level option.
2. Qwen3-ASR (0.6B/1.7B) + Qwen3-ForcedAligner is the Chinese community's current favourite for accuracy and low hallucination; MLX port exists with word timestamps. ~RTF 0.03-0.08 on M4 Pro. Forced aligner capped at 180 s/call: chunk it.
3. transcribe.cpp (handy-computer, ggml; Metal+CUDA+Vulkan) is a NEW unified runtime found via Zhihu. 16+ model families, Python binding, Apple + CUDA, published per-machine speed matrices. Actively maintained (v0.2.4, 2026-09-25). Candidate single backend for both platforms. But timestamps: Whisper=segment only, Parakeet=token, SenseVoice/Qwen3-ASR/Fun-ASR = none.
4. sherpa-onnx CoreML provider is NOT a win: slower than CPU on M2 (RTF 0.457 vs 0.372). CUDA works but is ORT-version/CUDA-version fiddly. Maintainer says sherpa-mlx is being merged into sherpa-onnx.

## Findings

### Measured speed, hardware named

- [GEM] SenseVoiceSmall in transcribe.cpp, Apple M4 Max, Metal Q8_0: 11 s clip 45 ms (245x), 35 s clip 122 ms (289x). CPU 48-51x. Compute-only (mel+encode+decode), no load. Also Ryzen 4750U Vulkan 27x. Clip cap 30 s/call, no built-in long-form chunking. https://github.com/handy-computer/transcribe.cpp (docs/models/sensevoice-small.md)
- [GEM] Qwen3-ASR-0.6B in transcribe.cpp, M4 Max, Metal Q8_0: 75x (11 s), 63.5x (35 s); CPU ~17-20x. Same source, docs/models/qwen3-asr-0.6b.md.
- [GEM] Zhihu 945-clip FLEURS zh benchmark, transcribe.cpp on Apple M5 32GB, Metal, Q8_0, serial, greedy (2026-07-20): Qwen3-ASR 0.6B CER 7.65% at 10.5x; Fun-ASR-Nano-2512 8.59% at 14.0x; SenseVoice Small 10.11% at 57.8x; Moonshine Tiny zh 13.74% at 23x (5 generation failures, repeats, truncation, simp/trad switching). 100-clip common set: Qwen3 6.83, Whisper-large-v3-turbo 7.51, Fun-ASR 7.91, SenseVoice 9.19. Author warns 100 clips -> CIs overlap. NOTE: end-to-end 10.5x for Qwen vs compute-only 63-75x on M4 Max in the repo docs: large gap, probably per-file overhead/model load in his harness; do not treat 10.5x as kernel speed. https://zhuanlan.zhihu.com/p/2064499997513720008
- [GEM] Whisper Notes (vendor, zh page, M4 Pro 32GB, 27-min Chinese podcast): SenseVoice Small via MLX 13.83 s (118x); Whisper-large-v3-turbo 2m04s (13.1x); old Qwen3-ASR via sherpa-onnx CPU 224 s (7.2x). 5-min English: Parakeet v3 2.91 s (103x), SenseVoice 5.8 s (52x), Whisper-turbo 20.92 s (14.3x). Caveat: vendor marketing; Qwen3 number is CPU-only ONNX, not a fair MLX comparison. https://whispernotes.app/zh/blog/sensevoice-fastest-cjk-transcription
- [GEM] mlx-qwen3-asr (moona3k), M4 Pro, 0.6B: fp16 RTF 0.08 (10 s clip 0.83 s); q8 0.03; q4 0.02 (4.7x faster than fp16, +0.43 WER abs on 100 samples). Memory 1.2 GB fp16. Streaming RTF 0.08. 735 tests, benchmark JSON committed. v0.4.4 2026-09-20, 215 stars. Native MLX forced aligner for word timestamps (claims 2.6x faster than PyTorch aligner). https://github.com/moona3k/mlx-qwen3-asr/ , https://deepwiki.com/moona3k/mlx-qwen3-asr/5-performance-and-optimization
- [MEH] Reddit echo of above (English): Whisper turbo on M1 Pro, 66 s audio: turbo 24 s vs large-v3 130 s (mlx-whisper). https://www.reddit.com/r/LocalLLaMA/comments/1fvb83n/
- [MEH] cnblogs, M1 Pro, openai-whisper CPU: ~50 s on its test clip; author could not get faster-whisper/whisperx working on Mac. Confirms faster-whisper has no Metal path. https://www.cnblogs.com/aehyok/p/18183895
- [MEH] cnblogs (AWS blog, mlx-whisper large-v3-turbo on Apple Silicon): "1 hour ~ 5-10 min" and a known silence-segment hallucination (repeated fixed phrases); they pipeline a hallucination filter. Corroborates the English "mlx-whisper loop bugs" finding. https://www.cnblogs.com/AmazonwebService/p/19905468
- [MEH] Zhihu "FunASR 170x Whisper": 5-min meeting clip, Intel Xeon 28-core, 1 h audio: GPU 21 s, CPU 3.5 min; Whisper CPU 1.5 h. Intel server, not Mac/RTX; promo style. https://zhuanlan.zhihu.com/p/2046886874451644501
- [MEH] FunAudioLLM paper: SenseVoice-S 10 s audio in 70 ms vs Whisper-L-v3 1281 ms on A800, batch 1. Datacenter GPU only. https://zhuanlan.zhihu.com/p/713191354
- [MEH] Bilibili i5-11400H CPU-only Qwen3-ASR 0.6B: 32.5 min audio in ~5 min (RTF ~0.15) https://www.bilibili.com/video/av116447233711399
- [MEH] firered-realtime-voiceprint repo: FireRedASR2S + CUDA Graph decode RTF 0.024 (20 s in 0.47 s), GPU unnamed in snippet. Only FireRed datapoint found; no Mac data. https://github.com/aihaili/firered-realtime-voiceprint

### Timestamps

- [GEM] SenseVoiceSmall timestamps are broken/unsupported through FunASR: char count != timestamp count with output_timestamp+VAD (SenseVoice #215, open, "needs feedback", Apr 2025); SenseVoice+CAM++ diarization crashes on timestamp incompatibility (FunASR #2945, #2662); Chinese blogs resort to "FSMN-VAD segment then recognise each segment" to get segment-level times only. https://github.com/QwenAudio/SenseVoice/issues/215 , https://github.com/modelscope/FunASR/issues/2945 , https://blog.csdn.net/weixin_30872157/article/details/159700930 (CSDN, only for the workaround idea). transcribe.cpp lists SenseVoice capabilities as "-" (no timestamps).
- [GEM] FunASR own guide: Paraformer-zh gives per-CHARACTER timestamps (ms) and hotwords, Chinese-only; "CPU + Chinese-only + timestamps -> Paraformer". Vendor blog (updated 2026-07-15). https://www.funasr.com/en/blog/which-funasr-model.html . Not useful for English/Vietnamese.
- [GEM] Qwen3-ForcedAligner-0.6B: official input cap 180 s per align call (ASR itself 1200 s). Community report (X, Feiteng) says beyond 180 s the timeline "混乱不堪" and recommends LattifAI Lattice-1 for hours-long; WhisperJAV issue #440: Qwen3-ForcedAligner (LLM-style global aligner) gives trailing/drag on short lines, wav2vec2+CTC gives sharper boundaries. Cnblogs deployment write-up: chunk to 20-60 s blocks, align each, restore global timeline. Text errors shift the whole timeline ("多一个'的'字" pushes everything). Supports only 11 languages (per XiaoHu.ai summary; verify Vietnamese). https://x.com/FeitengLi/status/2055237563393466560 , https://github.com/meizhong986/WhisperJAV/issues/440 , https://www.cnblogs.com/shxjnpll/p/22940249 , https://www.xiaohu.ai/c/a066c4/qwen3-asr-52-whisper
- [MEH] WhisperJAV #417: users report Qwen3-ASR + VAD-only timing drifts 1-2 s on short subtitles; whisper-large-v2's built-in timing correction more accurate than VAD-only pipelines. https://github.com/meizhong986/WhisperJAV/issues/417
- [MEH] SegmentFault Qwen3-ASR packaging post: official streaming mode (vLLM only) does not return timestamps or batch; on Windows vLLM install failed. https://segmentfault.com/a/1190000047705939
- [NOISE] CSDN "star-map GPU platform" Qwen3-ForcedAligner reviews claiming +-15-20 ms, -47% error vs Whisper: auto-generated promo, no method. e.g. https://blog.csdn.net/weixin_42097508/article/details/157855599

### Language coverage (Vietnamese question)

- [GEM] SenseVoiceSmall: zh, yue, en, ja, ko only. No Vietnamese. https://github.com/QwenAudio/SenseVoice
- [GEM] Fun-ASR-MLT-Nano-2512: 31 languages incl. Vietnamese (zh,en,yue,ja,ko,vi,id,th,ms,tl,ar,hi,...). Flagship Fun-ASR-Nano-2512: zh/en/ja only + dialects. HF card admits no MLT per-language results published yet ("will be added when reproducible evaluation is published"). No Vietnamese WER exists from a primary source. https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512 , https://github.com/QwenAudio/Fun-ASR
- [GEM] Qwen3-ASR: 30 languages + 22 Chinese dialects (52 "languages and dialects"); Aliyun hosted qwen-audio-3.0-asr-flash lists Vietnamese. Vietnamese on the open weights not verified by me. https://help.aliyun.com/zh/model-studio/asr-model
- [MEH] transcribe.cpp also ships moonshine-base-vi (Vietnamese) and the Fun-ASR-MLT model; no Vietnamese accuracy numbers checked. https://github.com/handy-computer/transcribe.cpp
- [NOISE] CSDN "Fun-ASR-MLT-Nano Thai/Viet/Indonesian test" (tencentcloud.csdn.net/69eaa9920a2f6a37c5a4ea8c.html): qualitative, no numbers.

### sherpa-onnx (k2-fsa)

- [GEM] CoreML provider slower than CPU: FireRedASR int8 on M2, macOS 26.0.1, 74 s audio: CoreML RTF 0.470 (2 thr), 0.457 (10 thr) vs CPU 0.372 (10 thr). Maintainer csuk: "We will merge sherpa-mlx into sherpa-onnx, allowing you to use the NPU/GPU on Mac via mlx, which doesn't rely on onnx/onnxruntime." https://github.com/k2-fsa/sherpa-onnx/issues/2910
- [GEM] Fun-ASR-Nano with --provider=coreml crashes (open, May 2026) https://github.com/k2-fsa/sherpa-onnx/issues/3099 . macOS shared libs built with deployment target 15.5 crash every macOS<15 app at launch (#3840, open Aug 2026).
- [MEH] CUDA needs matching ORT/CUDA/cuDNN (issue #1465; CSDN guides). sherpa-onnx v1.13.x, very active.
- [MEH] Implication: the English finding "onnx-asr Parakeet CUDA/CoreML" deserves an M1 Pro benchmark before trusting CoreML EP; Chinese data says CoreML EP gives no speedup for transformer/conformer models.

### Repo health (gh, 2026-09-30)
- handy-computer/transcribe.cpp: 1970 stars, v0.2.4 (2026-09-25), pushed 2026-09-30. Python/TS/Rust/Swift bindings, CUDA flag -DTRANSCRIBE_CUDA=ON. Young (v0.2) -> API churn risk (docs/migrating-to-0.2.md exists).
- moona3k/mlx-qwen3-asr: 215 stars, v0.4.4 2026-09-20. Single maintainer. Issues (closed): "Duration caps at 4 minutes", "SRT line split".
- Blaizzy/mlx-audio: 7964 stars, v0.5.7 2026-09-28 (Whisper, Qwen3-ASR, others, OpenAI-compatible server). Chinese video reviews recommend it as the "Ollama of speech on Mac" (https://www.bilibili.com/video/av116709579167766, marketing-ish).
- modelscope/FunASR: ~20.5k stars, v1.4.16 2026-09-18. QwenAudio/SenseVoice (moved org): 9.4k stars, llama.cpp runtime releases. QwenAudio/Fun-ASR llama.cpp runtime v0.1.10 (2026-09-05). Fun-ASR-Nano quoted ~340x on vLLM (vendor).
- k2-fsa/sherpa-onnx v1.13.x active.
- QwenLM/Qwen3-ASR: 3.6k stars, no releases, pushed 2026-06-26.
- DataoceanAI/Dolphin: 792 stars, last release 2025-07. Only hype articles (CSDN/aliyun/baidu), no independent numbers. [NOISE]/[MEH]
- FireRedTeam/FireRedASR2S: zh/en + dialects, no release tags; no Mac RTF evidence found. [MEH]
- anvanvan/mac-whisper-speedtest: last push 2025-08, 50 stars, stale. [MEH]

### Community opinion (qualitative)
- [MEH] cnblogs personal compare on phone-call audio: speed funasr(SenseVoice) > faster-whisper > whisper; accuracy whisper-large-v3 > faster-whisper > SenseVoice; SenseVoice for clean standard Mandarin only. https://www.cnblogs.com/xio1028/p/19011196
- [MEH] X Feiteng: drop Whisper for Qwen3-ASR ("幻觉很少", longer audio up to 20 min more accurate), Whisper needs 30 s chunks and its timestamps "不是很准". https://x.com/FeitengLi/status/2055237563393466560
- [MEH] Yuanchang blog (EN, M4 Pro 24GB): openai-whisper CPU 2.5 h no output; mlx-whisper turbo ~3 min for 20 min. https://yuanchang.org/en/posts/apple-silicon-stt-guide/
- [MEH] V2EX VoxScripts: moved whisper.cpp -> faster-whisper because whisper.cpp lacked word-level timestamps; https://www.v2ex.com/t/1024340 (2024, dated)
- [NOISE] V2EX/linux.do hits: Typeless backend thread (linux.do/t/topic/2378299, pointer to a comparison repo only), CapsWriter-Offline (SenseVoice dictation), SubtitlePipeline (whisper small CPU). No Mac/CUDA numbers.
- [NOISE] CSDN/juejin/bilibili Qwen3.8 LLM posts (unrelated), "整合包" pan-link posts.

## What this adds / contradicts vs English findings

Adds:
- transcribe.cpp as a one-runtime option (Metal+CUDA, Python binding, 16+ families) with per-machine speed tables; English wave missed it. Weak point: only Whisper (segment) and Parakeet (token) give timestamps.
- Quantified Mac speeds for non-autoregressive SenseVoice (118-289x) vs Whisper-turbo (13x) vs Qwen3-ASR (7-75x depending on runtime). Shows autoregressive decoders are the bottleneck, not the framework.
- Qwen3-ASR MLX port with native forced aligner = a word-timestamp path on Mac that is not Whisper/Parakeet.
- Forced aligner failure mode: 180 s cap, drift on long audio, text-error timeline shift, trailing on short lines.

Contradicts / tempers:
- onnx-asr "CoreML" candidate: sherpa-onnx data says CoreML EP was slower than CPU for conformer/transformer models on M2. Verify on M1 Pro before adopting.
- SenseVoice's speed reputation does not translate to framesift needs: no word timestamps (open bugs), 5 languages, no Vietnamese, 30 s clip cap.
- mlx-whisper: Chinese sources confirm hallucination on silence and recommend dropping Whisper for Qwen3-ASR for hallucination, but Whisper still wins "mature long-audio handling" (Zhihu 945-clip author).
- Fun-ASR-Nano/MLT is the only Chinese-lab model with Vietnamese, but no independent Vietnamese accuracy and no timestamps in transcribe.cpp.

## Unresolved
- Real Vietnamese WER for Fun-ASR-MLT-Nano and Qwen3-ASR open weights (no source found).
- Does Qwen3-ForcedAligner support Vietnamese (claim "11 languages" unverified).
- Why 10.5x (Zhihu, M5) vs 63-75x (repo, M4 Max) for Qwen3-ASR 0.6B: harness overhead or quant/path difference.
- No RTX measurement for SenseVoice/Qwen3-ASR/transcribe.cpp CUDA found (only A800, 4090 mentions without numbers). transcribe.cpp docs may have CUDA tables for other machines; not checked.
- M1 Pro specific numbers for any of these (all Mac data is M4/M5).
- mlx-audio Qwen3-ASR / Whisper timestamp quality unchecked.
- Bilibili 4-way Mac Whisper comparison (BV1ZdSfBXEo3) not retrievable (HTTP 412).
- linux.do / V2EX / SegmentFault yielded nothing substantive; likely search-engine coverage gap, not absence of discussion (sov2ex/tieba failed).
