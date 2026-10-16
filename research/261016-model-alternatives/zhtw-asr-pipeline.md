# zh-TW practitioner reports: local speech pipeline on Apple Silicon (framesift model alternatives)

Date 2026-10-16. Language: zh-TW queries only. Report written in English.
Tags: GEM = actionable with numbers; MEH = useful context, thin; NOISE = skip. UNVERIFIED = social-only.

## Coverage summary

| Sub-area | GEMs | Verdict |
|---|---|---|
| (a) Mac ASR speed reports | 3 | One dense source (jt-live-whisper BENCHMARKS.md) carries almost everything. PTT / iThome 鐵人賽 / Medium / HackMD gave no usable Mac RTF numbers. |
| (b) Diarization | 3 | Nemotron 3 Diarization is the news. Real-meeting numbers (zh + en) plus Mac MPS and CoreML speeds. |
| (c) Sound-event tagging / LID | 1 | Tagging: EMPTY from zh-TW practitioners (only SenseVoice blurbs, MEH). LID: one GEM (Qwen3-ASR auto-LID misfires). |
| (d) Code-switching | 4 | Best-covered. Direct measurements, including a negative result for Breeze on English. |
| EmbeddingGemma 2 audio (extra) | 1 | CoreML audio encoder speed on M5 Pro. |

Search caveats: Threads source was down during the run (error since 2026-10-15), so Threads was effectively not searched. Facebook returned mostly announcements, no numbers. PTT/ithelp/Medium/HackMD `site:` searches ran (about 10 queries) and mostly returned tutorials (NOISE) or platforms other than Mac. zh-TW community mostly links to or runs the same few GitHub projects. Only the one `jasoncheng7115` repo published structured benchmarks.

## Key finding

Single best zh-TW source: https://github.com/jasoncheng7115/jt-live-whisper (574 stars, Apache-2.0, last push 2026-10-15, created 2026-03-03). Author Jason Cheng (Taiwan). Mac, Windows and Linux. Its BENCHMARKS.md contains a ground-truth methodology for Taiwan meeting audio. Read it: https://github.com/jasoncheng7115/jt-live-whisper/blob/main/BENCHMARKS.md and SOP.md ("Qwen3-ASR (experimental)" section).

Caveat: these are one author's self-measurements, not independent. The corpora are named (AISHELL-4, AMI, Common Voice, nan_tw_soap_opera), so they are reproducible.

## (a) Mac ASR speed

### GEM-a1. Qwen3-ASR 0.6B on Apple Silicon (MLX via mlx-audio), Taiwan meetings
- Source: jt-live-whisper SOP.md and README (URLs above).
- Speed: Mac Apple Silicon (M5): 37-min meeting in about 1.5 min total = ASR 64 s + forced alignment 22 s. That is about 35x realtime for ASR alone and about 26x end to end. Memory about 3.2 GB; model download 2.3 GB (MLX 8bit).
- Accuracy vs large-v3-turbo, same audio and same ground truth:
  - Real Chinese meetings, 20 AISHELL-4 sessions of about 38 min each: CER 28.78% (turbo) vs 15.75% (Qwen3-ASR 0.6B).
  - Low volume (-60 dBFS): 20.67% vs 14.33%.
  - Korean long file: 13.35% vs 3.54%.
  - Japanese long file: 6.97% vs 8.38% (Qwen worse).
  - Same 37-min meeting, Mac vs GPU server: CER 18.85% on both. Median sentence-start error 280 ms.
- Limits stated by author: Chinese, English and Korean only. Offline only. No Taiwanese Hokkien.
- License: Apache-2.0.
- Chip caveat: M5, not M1 Pro. Scale down for M1 Pro (M1 Pro memory bandwidth is lower than M5). Not directly comparable to the 156x QUAN number.
- Related, English-source: https://github.com/moona3k/mlx-qwen3-asr (229 stars, Apache-2.0, pushed 2026-09-20). On M4 Pro 48 GB: 0.6B fp16 RTF 0.029 on a 10 s clip, 4-bit RTF 0.018, 1.7B fp16 RTF 0.077. Pip package `mlx-qwen3-asr`. Python-friendly, which fits Py3.14 if the MLX wheel exists (not verified for 3.14).

### GEM-a2. Breeze-ASR-25/26 on Apple Silicon is SLOW
- Source: jt-live-whisper SOP.md. Breeze-ASR-26 (Taiwanese Hokkien) in mlx 4bit (877 MB): about 1.3x realtime on Apple Silicon, so about 45 min per hour of audio. CPU int8: 0.24x. Reason given: it is large-v2 fine-tuned, and the decoder has about 8x the layers of large-v3-turbo.
- Breeze-ASR-25 is the same architecture, 1.55B params (HF safetensors total 1,543,304,960). Expect the same slow regime.
- fredchu CoreML 4-bit (WhisperKit): https://huggingface.co/fredchu/breeze-asr-25-whisperkit-coreml. About 1.05 GB, Apache-2.0, 96 downloads. The card's own "Mac M1+ ~2-4 s per 30 s audio" is an ESTIMATE, not a measurement. Card also says first load ANE compile takes about 10-12 minutes per component. Conversion took 4.5 h on M1 Max.
- Verdict for framesift: Breeze is a 1.55B Whisper-large-v2 and cannot compete on speed with a 356M CTC at 156x. Only worth it for zh-TW or Taiwanese content.

### GEM-a3. Breeze-ASR-25 accuracy (paper numbers, relayed on the CoreML card; arXiv 2506.11130)
- Whisper large-v2 vs Breeze-ASR-25 WER: ASCEND-MIX (code-switching) 21.01 to 16.38; CommonVoice16-zh-TW 9.84 to 7.97; CSZS-zh-en (code-switching) 29.49 to 13.01 (-56%). Long-form: ML-lecture-2021-long 6.13 to 4.98; Formosa-Go 15.03 to 13.61.
- Maturity: HF MediaTek-Research/Breeze-ASR-25: 27,893 downloads, 153 likes, last modified 2025-07-08. GitHub mtkresearch/Breeze-ASR-25: 221 stars, last push 2025-07-01. License Apache-2.0. Many community conversions (GGUF, ggml, MLX, faster-whisper), all dated Sep-Oct 2026 and low downloads.

### MEH
- https://github.com/yotronai/breeze-asr-mac: Breeze-ASR-25 CoreML via whisperkit-cli (`brew install whisperkit-cli`) plus whisperkit diarize (speakerkit-coreml, about 11 MB) plus FastAPI. 1 star, 1 commit. Architecture reference only; no benchmarks.
- https://blog.gslin.org (2026-10-10): runs Breeze-ASR-25 ggml in Handy on macOS for dictation. Anecdote, "works smoothly". No numbers.
- Business Insider TW (2026-07-12), https://www.businessinsider.tw/article/4360: 4-min real press-conference test. OpenAI whisper-1 best overall; Breeze-ASR-25 and Groq large-v3 roughly tied; Breeze better on names and later narrative, Groq better on English titles (Founder, co-founder). Qualitative only. The editors keep Breeze as a "second opinion".
- FluidAudio (https://github.com/FluidInference/FluidAudio, 3,102 stars, Apache-2.0, pushed 2026-10-15): README says Parakeet TDT v3 at about 190x realtime on M4 Pro (1 h audio in about 19 s). It also lists SenseVoice and Paraformer for Mandarin and a `Diarizer` supporting pyannote, Sortformer, LS-EEND and Nemotron 3. Swift SDK, not pip, so you would call a CLI (`fluidaudiocli`). It has no Vietnamese ASR. Not a zh-TW report; surfaced by search.
- UNVERIFIED (Facebook video, will.fans): Breeze-ASR-26 Taigi run on an MBP M5 local. No numbers retrievable.
- NOISE: PTT MAC/Notebook threads (MacWhisper, WhisperDesktop on Intel AI PC), ithelp Whisper install tutorials, ithelp Day 2 MLX Whisper STT (M5 Pro, no speed numbers).

## (b) Diarization

### GEM-b1. NVIDIA Nemotron 3 Diarization, measured on Chinese and English real meetings
- Source: jt-live-whisper BENCHMARKS.md section 4 (2026-09-24 to 10-01). Became the default in v2.26.0.
- Model: 100M params, up to 8 speakers, Sortformer lineage, streaming or offline. Card says "ready for commercial or non-commercial use". License OpenMDW-1.1. No HF gate, no token. Needs transformers 5.18+.
- Test: real ASR segments, not ground-truth segments. Chinese: AISHELL-4 test, 20 meetings, 5-7 speakers. English: AMI test, 16 meetings.
- Segment speaker-error: Chinese 18.52% (resemblyzer+spectralcluster) to 3.07% (Nemotron); English 12.31% to 4.65%. Speaker count correct: Chinese 2/20 to 17/20; English 12/16 to 15/16. Rerun on final transformers 5.18: Chinese 2.92% (18/20), English 4.65% (15/16).
- Speed on a 37-min Chinese meeting, full product flow: NVIDIA GB10 about 15 s; Mac Apple Silicon (MPS) about 50 s (about 44x realtime); Windows 2-core i5 CPU about 2 min 45 s. Results are near-identical across platforms (887/904 segments the same).
- Gotchas: max 8 speakers. Intel Mac unsupported. `num_speakers` is a CAP, so under-specifying hurts (error 2.92% to 6.89%) while over-specifying is harmless. Do not pass a count unless sure.
- Comparison to the current framesift stack: framesift uses TitaNet-small via sherpa-onnx plus greedy clustering. The author's old method (resemblyzer d-vector + spectral clustering) failed the same way: undercounts speakers in many-speaker, short-turn meetings (a 7-person meeting split into 3 clusters). This is the same class of weakness as greedy clustering. Worth a test.
- Maturity: nvidia/Nemotron-3-Diarization has 87,010 downloads. Dated September-October 2026. Many ports in Oct 2026: mlx-community/Nemotron-3-Diarization-8bit (500 dl), audio-cpp GGUF (34k dl), several CoreML, ONNX and LiteRT ports (mostly 0-26 downloads).
- NVIDIA card (RTX PRO 5000, bf16): offline config 1340x eager / 4385x compiled at batch 1.

### GEM-b2. CoreML port of Nemotron 3 Diarization on Apple Silicon (FluidAudio route)
- Source: https://huggingface.co/aidsoid/nemotron-3-diarization-coreml (26 downloads; third-party port, English card, found via zh-TW search path). UNVERIFIED independently.
- Chip: M5 Pro. AMI-MHM test, 16 meetings, collar 0, overlap included.
  - `offline`: DER 9.47 (NVIDIA card 9.25), 904x wall realtime, 190 MB.
  - `fast128`: DER 9.36, speaker-count accuracy 16/16, 546x.
  - `fast32`: DER 9.53, 179x, "recommended default".
  - `c128-split-w8a8`: 95 MB, 100% ANE, 364x.
- FluidAudio README lists Nemotron 3 under its `Diarizer`. Needs a Swift CLI bridge. No pip wheel.
- If speed matters, CoreML is about 20x faster than the MPS path in b1 (904x vs 44x), though on different chips and different tests.

### GEM-b3. Why not pyannote, and the pyannote speed baseline
- jt-live-whisper README: pyannote weights are MIT/CC-BY-4.0 but gated (HF login, form, token), which breaks a zero-account local tool. So the author chose Nemotron.
- FluidAudio Benchmarks.md (M5 Pro run, 2026-07-03): pyannote community-1 pipeline in CoreML, VoxConverse DER 15.07% at 122x (step 0.2) or 13.89% at 65x (step 0.1). PyTorch baseline about 11% DER. pyannote community-1 on CPU about 1.5-2x realtime, on MPS about 20-25x.
- Source for Nemotron vs pyannote head-to-head on the same data: none found. Nemotron's own card compares only against Sortformer.

### MEH
- ithelp 17th 鐵人賽 "AI語音辨識系統" series (Harper team, Day 4-7): whisper + NeMo diarization + pyannote + speechbrain emotion, concept only, no Mac numbers.
- ithelp AI image processing Day 12: WhisperX plus pyannote tutorial. Concept only.

## (c) Tagging and LID

### GEM-c1. Qwen3-ASR auto language ID misfires on Taiwanese speech
- Source: BENCHMARKS.md section 2: "Qwen's auto language detection judges Taiwanese Hokkien as Thai or Cantonese; the 0.6B often runs as Cantonese." Forcing the language fixes it, but they do not offer Hokkien at all. Limited to zh, en, ko in their product.
- Implication: if framesift uses an ASR model's built-in LID, test on minority and accented speech. Keep a separate LID step (whisper-tiny) and pass the language explicitly.

### Tagging: empty
- No zh-TW practitioner report on sound-event tagging for AudioSet-class (CED, YAMNet, BEATs) on Mac. Only ithelp 2021 YAMNet intro (NOISE).
- SenseVoice-small (FunAudioLLM; repo now at https://github.com/QwenAudio/SenseVoice, 9,514 stars, pushed 2026-09-30): ASR + LID + emotion + event tags (applause, laughter, music) in one model; run via sherpa-onnx (k2-fsa/sherpa-onnx, 15,302 stars, pushed 2026-10-15). Claim from Deep-Learning-101 (Taiwanese author TonTon Huang) of "15x faster than Whisper-large" is a vendor-style claim: MEH, no Mac chip given. FluidAudio also lists SenseVoice for Mandarin on CoreML.
- Event vocabulary of SenseVoice is only a handful of classes (not 527 AudioSet classes), so it cannot replace CED-small.
- sherpa-onnx LID uses Whisper multilingual models only (per the k2-fsa doc found), the same family as framesift's current whisper-tiny approach. No zh-TW comparison found.

## (d) Code-switching (zh-en), informs Vietnamese-English

### GEM-d1. LLM-decoder ASR (Qwen3-ASR 0.6B) keeps the English words that Whisper turbo drops
- Source: jt-live-whisper SOP.md "Qwen3-ASR (experimental)" table. Same audio and ground truth.
  - zh-dominant mixed speech: English word recall 34.7% (large-v3-turbo) vs 74.9% (Qwen3-ASR 0.6B).
  - en-dominant mixed speech: Chinese character recall 25.6% vs 83.0%.
- Sample set: constructed mixed recordings, not a public corpus. Size not stated in SOP. Treat as indicative.
- Relevance: framesift's QUAN (CTC) drops English names (ChatGPT, AGI). This is the same failure shape, and an LLM-decoder reader keeps them.

### GEM-d2. Breeze-ASR-26 (Whisper-large-v2 fine-tune) translates English into Chinese
- Source: BENCHMARKS.md section 1 (2026-09-29). Mixed recording, 12 segments (6 Hokkien, 3 Mandarin, 3 English), one language per segment.
  - Breeze-26: English mostly turned into Chinese, sometimes with errors ("before Friday" became "星期天").
  - large-v3-turbo (Chinese mode): English segment dropped entirely.
  - Qwen3-ASR (Chinese mode): English dropped entirely, and the neighbouring segments vanished too.
- Not tested: intra-sentence mixing (author says no ground-truth corpus). Lesson: with a forced language, models either translate or delete the other language. Language setting matters.

### GEM-d3. Breeze-ASR-25 paper numbers on code-switching
- See GEM-a3: CSZS-zh-en WER 29.49 to 13.01 (-56%), ASCEND-MIX 21.01 to 16.38. Training recipe: TTS-synthesised code-switch data (arXiv 2506.11130). The synthetic-data approach is portable to Vietnamese-English, i.e. a fine-tune route if framesift ever trains its own reader.

### GEM-d4. Hotwords and prompts can hallucinate
- Source: BENCHMARKS.md section 6. faster-whisper `hotwords` with a list unrelated to the audio: on a 37-min Chinese meeting, "李主任" was inserted 14 times and "王經理" 4 times. Segments dropped from 1043 to 346 and runtime rose from 72 s to 401 s. On Japanese, CER rose 13.22% to 14.23%. Author rejected hotwords at ASR time and corrects after the fact instead. Relevant if framesift is tempted to add a glossary to fix English names.
- Also section 3: Whisper anti-hallucination parameters tuned for large-v3-turbo made Breeze-26 worse (CER 17.99% to 56.42%, 4.7x slower). Fine-tuned Whisper variants need their own decode parameters.

## EmbeddingGemma 2 audio on Mac

### GEM-e1. Core ML audio encoder
- Source: https://huggingface.co/FluidInference/embeddinggemma-2-coreml (883 downloads). Text on the ANE; audio and vision encoders on the GPU. Audio model is 561 MB fp16 (12-layer Gemma 4 USM conformer + projection), 10 s windows to 250 tokens.
- Chip: M5 Pro. 1 hour of audio: audio model alone 5.8 s (617x realtime); with the text model 10.2 s (353x); fp32 13.2 s (272x). fp16 window embedding cosine to fp32 is at least 0.977, mean 0.9993. On a 1-hour earnings-call index, top hit matched fp32 for 15/15 text queries; top-5 overlap 95%.
- Google model (google/embeddinggemma-2, 144,280 downloads): 740M total (270M text-only), Apache-2.0, 8K token context = about 5.5 min audio per call when audio only (iThome news 179475, 2026-10-07). Matryoshka dims 768/512/256/128; 256 keeps about the same retrieval quality.
- Not tested on M1 Pro. Expect a few times slower since the encoder is GPU-bound. English-language test; no zh-TW or Vietnamese retrieval numbers found.

## Recommended follow-ups for framesift (ranked, evidence-based)

1. Diarization: benchmark Nemotron 3 Diarization (ports: `mlx-community/Nemotron-3-Diarization-8bit`, CoreML via FluidAudio, or transformers on MPS) against TitaNet-small+greedy on a many-speaker clip. The zh-TW source reports undercounting in the old clustering approach and a large fix. Python route exists (transformers 5.18, MPS) at about 44x on M5; the M1 Pro number is unknown.
2. English-name dropping in Vietnamese: evidence (d1) says an LLM-decoder reader keeps mixed-language tokens. Qwen3-ASR (0.6B, MLX, Apache-2.0) is a candidate second pass for segments flagged as code-switched. The 156x QUAN pass could stay for bulk. Needs a Vietnamese-English test; no Vietnamese data in these sources.
3. Do not adopt Breeze-ASR for Vietnamese pipeline speed: it is a 1.55B Whisper-large-v2 at about 1.3x realtime on Apple Silicon (jt measurement for Breeze-26 mlx 4bit).
4. Do not add ASR hotwords for names (d4).
5. Tagging: nothing here beats CED-small. No change.

## Unresolved

- No M1 Pro, M2, M3 or M4 numbers for Breeze or Qwen3-ASR from zh-TW sources (only M2 turbo about 2.5 s for a 3-8 s buffer, and M5). Everything lands on M5 or M4 Pro.
- jt-live-whisper tests are self-run. Corpus sizes for the code-switch test are not stated.
- Nemotron 3 vs pyannote community-1 head-to-head: not found.
- Whether Vietnamese is covered by Qwen3-ASR auto-LID code-switching: not tested in any source.
- Threads was down, so Threads/Dcard-style posts from practitioners were not checked. PTT has no Mac ASR benchmark threads found.

## Sources

- https://github.com/jasoncheng7115/jt-live-whisper (README, BENCHMARKS.md, SOP.md, CHANGELOG.md)
- https://huggingface.co/fredchu/breeze-asr-25-whisperkit-coreml
- https://huggingface.co/sheep52031/breeze-asr-25-coreml-ane
- https://huggingface.co/c1ayvveng/Breeze-ASR-25-mlx
- https://huggingface.co/MediaTek-Research/Breeze-ASR-25 and https://github.com/mtkresearch/Breeze-ASR-25
- https://huggingface.co/MediaTek-Research/Breeze-ASR-26
- https://github.com/yotronai/breeze-asr-mac
- https://blog.gslin.org/archives/2026/10/10/13262/
- https://www.businessinsider.tw/article/4360
- https://github.com/Jeffrey0117/SpeakSlow/issues/4
- https://huggingface.co/nvidia/Nemotron-3-Diarization
- https://huggingface.co/aidsoid/nemotron-3-diarization-coreml
- https://github.com/FluidInference/FluidAudio (README, Documentation/Benchmarks.md)
- https://github.com/moona3k/mlx-qwen3-asr
- https://github.com/QwenAudio/SenseVoice and https://github.com/k2-fsa/sherpa-onnx
- https://huggingface.co/FluidInference/embeddinggemma-2-coreml
- https://www.ithome.com.tw/news/179475
- https://hackmd.io/@shaoeChen/r1gkK6JQ-g (WhisperLiveKit + Breeze, Docker/CUDA, NOISE for Mac)
- https://ithelp.ithome.com.tw/articles/10402637, /10382810, /10349152 (concept only)
- https://www.ptt.cc/bbs/Notebook/M.1763123165.A.D0B.html (Windows AI PC, NOISE)
- https://github.com/Deep-Learning-101/Speech-Processing-Paper (SenseVoice claim, MEH)
- Facebook (UNVERIFIED, no numbers): https://www.facebook.com/will.fans/videos/ (Breeze-ASR-26 on MBP M5)
