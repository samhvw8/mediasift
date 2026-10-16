# EN code/GitHub/X: failure modes for framesift audio-stack candidates

Slice: GitHub issues/discussions + X only. Tool calls used: 19 of 20 (incl. skill/brief load). Date 2026-10-16.
Tags: GEM = number/incident/error/workaround from an issue I opened. MEH = real but thin. NOISE = nothing useful.
X (Parallax twitter_search): 4 calls timed out (context deadline exceeded), 1 retry failed on bad args. X NOT RESEARCHED. Risk: missed community claims on FluidAudio, QUAN-Transcribe, Parakeet-Vietnamese.

Gh query forms used: `gh search issues "<q>" --repo <owner/repo> --sort comments`; `gh issue view N -R repo --comments`; `gh issue list`/`gh pr list` where search failed; GraphQL discussion search (0 hits on FluidAudio, QUAN, WhisperKit, mlx-audio returned only discussions, no relevant ones).

## Repo snapshot (gh repo view)

| Repo | Stars | Last push | License | Note |
|---|---|---|---|---|
| FluidInference/FluidAudio | 3102 | 2026-10-15 | apache-2.0 | Swift package; CLI `swift run fluidaudio` |
| senstella/parakeet-mlx | 987 | 2026-10-01 | apache-2.0 | |
| istupakov/onnx-asr | 384 | 2026-10-09 | mit | |
| Blaizzy/mlx-audio | 8051 | 2026-10-14 | mit | |
| qnguyen3/QUAN-Transcribe | 26 | 2026-10-14 | n/a | 0 issues ever |
| k2-fsa/sherpa-onnx | 15302 | 2026-10-15 | apache-2.0 | |
| argmaxinc/WhisperKit | 6405 | 2026-10-11 | mit | Issues live in argmaxinc/argmax-oss-swift; `gh search` on WhisperKit repo errors |

## FluidAudio (Swift; no Python API seen)

- GEM #154 (FluidInference/FluidAudio): `transcribe --output` unsupported, output empty on release CLI; user wanted a faster ANE-offloading replacement for parakeet-mlx. Fixed in PR #156. Maintainer: Python OSS wrapper by hamzaq2000 "planned" (date of that comment unknown, unverified whether shipped). Only CLI path is `swift run fluidaudio transcribe foo.wav`, which logs text in debug mode.
- GEM #512: Parakeet TDT v3 transcribes short Latin-script utterances as Cyrillic (Polish reproduced). `language` hint has no effect: decoder path exposes only token_id/token_prob/duration, not logits, so pl/ru/no-hint all give identical output. Open question from reporter: needs model re-export? Relevant to any multilingual Parakeet use.
- GEM #594: French drifts into English BPE at chunk seams in batch `AsrManager.transcribe` (regression from #264). Reporter's test of PR #596: flag ON fixes 1 of 3 French fixtures, breaks 2 (mid-sentence drift, 99.9 s clip). Root-cause guess: OOD joint-network pair at seam. Not fixed in what I read.
- GEM #842: Spanish decoded through English lexicon. Reporter retracted "CoreML port bug": upstream parakeet-tdt-0.6b-v3 fp32 via parakeet-mlx fails the same way, measurably worse (M1 Pro, macOS 26.5.2). So it is model behaviour; do not blame the port.
- GEM #951: Nemotron3 diarizer, `ANECCompile()` fails on M3 Max; CoreML falls back and rejects preallocated fp16 output buffers. `--compute-units cpu` works on M5 Pro. Fix PR #952 (retry without preallocated buffers). ANE compile failure is chip/OS specific.
- GEM #661: concurrent Core ML calls (ASR re-transcription + diarizer in parallel) corrupt shared E5RT/BNNS state, EXC_BAD_ACCESS in libBNNS on Apple Silicon. Workaround: serialize Core ML calls. Closed by reporter.
- GEM #739 (title + fix PR #744 seen in comment, body not read): Nemotron streaming int8 zero output on cold start, iPadOS 26.5 on M1; works on macOS M1 Pro.
- MEH #49 (model-request thread, 90 comments): NVIDIA parakeet v3 requested; mentions v3 Russian support; CoreML speaker-diarization models published at FluidInference/speaker-diarization-coreml (edit dated 09/30). No DER numbers.
- Gaps: no M1 speed numbers, no diarization DER numbers found in any issue I opened. Vietnamese: search "Vietnamese" returned only #49 (noise, no Vietnamese content).
- Diarization accuracy reports exist but not opened: #513, #253 (see leads).

## parakeet-mlx (Python; CTC/TDT)

- GEM #4 (memory, OOM on long files): base M1 MacBook Air OOM, reproduced even with `parakeet-tdt_ctc-110m`. Fixed in v0.2.3: long-form chunking, default `--chunk-duration` 2 min, `--overlap-duration`. Reporter: `model.transcribe(mp3, chunk_duration=60)` works, RAM +~1.7 GB. Closed as resolved. Note: confirms the CTC-110m variant loads and runs.
- GEM #34: MPS memory not released across chunked loop. Maintainer workaround: `mx.clear_cache()` + `gc.collect()`. Reporter: per-chunk clear_cache slows Parakeet; used it at lower frequency. Closed.
- GEM #27 (token corruption, default CLI): M2 Max podcast transcripts show garbage runs (e.g. "wella-awaiovapplyauaioatoaoa"). Maintainer could not reproduce on M2 Air; suggested `--fp32`. Outcome unknown in what I read.
- GEM #43: abnormal timestamps, ~15 s time jump in second sentence (sentence loss). Maintainer pointed to FluidAudio #128 and said `--decoding beam` should fix; asked to reopen if it persists. Outcome unknown.
- MEH #49: `mx.concatenate` axis kwarg rejected with MLX 0.30.6. Only hit for "python 3.14" query. No Python 3.14 evidence either way (wheel or pyobjc issues): NOT RESOLVED here.
- NOISE #25 (VITS alignment question) — maintainer mentions parakeet-CTC as an alignment option; not a support issue.

## onnx-asr (Python; ONNX Runtime)

- GEM #99 (macOS CoreML benchmarks):
  - grikdotnet (Apple M5): ONNX CoreML EP does not support Parakeet/Canary nodes/shapes. M5 CPU runs Parakeet TDT at 40 RTFx (2 threads) to 45 RTFx (5 threads); Canary 1B 12 RTFx.
  - EuanTop (M3 Pro, 18 GiB, macOS 15.6, ORT 1.28.0, Python 3.13.12): whisper-tiny FP32. CPU 37.65 RTFx; CoreML NeuralNetwork + CPU fallback 18.47 RTFx; CoreML MLProgram "Error in building plan" (encoder init fail). NeuralNetwork format works. WER 7.11% on first 100 LibriSpeech test-clean clips (670 s subset; not full benchmark).
  - Corroborates the brief's sound-tagging finding (onnxruntime CoreML EP slower).
- GEM #13 (GPU slower than CPU): Parakeet TDT v2 CPU RTFx 4.2 vs CUDA 57.5 (100 clips, same WER 6.51%). Maintainer fix: `cpu_preprocessing=True` in load_model; made default in next version. Reporter hardware: A100, ORT-gpu 1.22.0, Ubuntu 20.04.
- GEM #18 (long audio + CUDA): chunked subsampling breaks on long audio on CUDA (L4, Ubuntu 24.04, driver 575.51.03, CUDA 12.9). Workaround found by reporter: disabling subsampling chunking (`change_subsampling_conv_chunking_factor(1)` at export) fixes it, but that export setting may break ONNX on long audio. Maintainer outcome not read.
- Vietnamese: no issue mentions Vietnamese. `vietnamese` search returned nothing. Unverified whether Parakeet v3 covers it (not checked in this slice).
- Python 3.14: no hits.
- Qwen3-ASR support requested (#73) — title only, not opened.

## mlx-audio (Python; Qwen3-ASR, Apple Silicon)

- GEM #928 (Qwen3-ASR-1.7B bf16, `/v1/audio/transcriptions`, default `chunk_duration=30`, no `language`): auto-detected language from first chunk is reused for all later chunks. Quiet first ~25 s of a 934 s recording: 1,108 chars, 33 segments with `language: "None"`, 30 empty; HTTP 200, no warning. Same file with `language=English`: 10,469 chars. Cause: no-speech chunk returns `language None<asr_text>`, stored and fed to later chunks. Synthetic repro with macOS `say` + 30 s silence. Directly relevant to framesift's per-chunk language handling: do not let auto-language persist across chunks.
- GEM #544: realtime WebSocket `language=None` crashes Qwen3-ASR `_build_prompt()`. Fixed in PR #546 (per maintainer). #546 body empty when read.
- GEM #331 (not opened, title only): GLM-ASR-Nano extremely slow + memory leak. See leads.
- Qwen3-ASR speed on M1: no measured numbers in any issue I opened. Gap.
- Roadmap #1 (71 comments) not opened.
- Vietnamese: only #796 (Confucius4-TTS, a TTS model), no Vietnamese ASR.

## QUAN-Transcribe (qnguyen3/QUAN-Transcribe)

- GEM (fact): 0 issues ever (`gh issue list --state all` empty). 0 discussions. Failure modes cannot be assessed from tracker. Risk: no public bug trail for the candidate framesift already uses.
- MEH PRs (no state checked): #1 live microphone `--mic`; #2 README restructure; #3 and #4 both "Add an OpenAI-compatible serving layer (quan-asr-server)" (duplicate title, one may be superseded).
- v1.1 roadmap: not found in issues/PRs. Unresolved.

## sherpa-onnx (CoreML provider, audio tagging, LID, Python 3.14)

- GEM #3689 (macOS sdist metadata, uv lock): sdist `install_requires` is conditional on env var `SHERPA_ONNX_SPLIT_PYTHON_PACKAGE`; when uv builds metadata from sdist, the `sherpa-onnx-core` dependency is dropped. Wheels are fine, so `uv lock` and `uv pip install` disagree. Workaround: add `sherpa-onnx-core==1.13.3` explicitly. Relevant to any uv-locked framesift install.
- GEM #3459 (Qwen3-ASR fails to run): closed via PR #3460.
- CoreML provider speed: no numbers in issues. Titles seen: #1792 Kokoro TTS with CoreML provider (not opened), #1188 macOS deployment target build failure (not opened). Consistent with onnx-asr #99 (CoreML EP unsupported ops), but no direct sherpa CoreML speed data. Gap.
- Audio tagging and spoken language ID: search returned only noise (TTS bugs, OpenHarmony, CUDA wheel builds). NOISE.
- Python 3.14 wheels: no 3.14 evidence. Hits are cp312 wheel requests (#2750, #3177 win_amd64 missing). NOT RESOLVED.

## WhisperKit (now argmaxinc/argmax-oss-swift)

- GEM #98 (non-English): with language unset, distil-whisper large-v3 594MB translates Russian/Spanish to English; with language set, still translates/transliterates (Russian "попытка распознать речь" to transliteration). `base` model works. Maintainer: small models struggle; use explicit language. Note: distil-large-v3 is English-only (my knowledge, unverified in this run).
- GEM #265 (M1, memory): memory leak with `ModelComputeOptions .cpuAndGPU` on Turbo model on M1. Repro code only read; leak numbers not read.
- MEH #7, #39: "Unable to load models" / "Initializing models... unknown error" (titles only).
- Language detection thread #226, #29 (titles only, see leads).
- Gaps: no M1 speed numbers in opened issues. #301 "performance degradation vs whisper.cpp" (not opened) is the best speed lead.
- New bug titles from issue list (not opened, not GEM): #539 greedy sampler tied top-k sums → near-zero-prob token at temperature > 0 → empty transcription; #542 word timestamps wrong when promptTokens set; #551 SpeakerKit diarize memory grows with recording length (no backpressure).

## Cross-cutting findings

1. Parakeet multilingual is unreliable on non-English: FluidAudio #512, #594, #842 (last one shows upstream fp32 has the same problem). Framesift's routing of Vietnamese to QUAN and others to parakeet should not assume v3 handles off-language cleanly.
2. Long-form memory: parakeet-mlx needed chunking (#4), MLX cache clearing (#34). Expect chunking for long files.
3. Auto-language persistence is a real bug class: mlx-audio #928 shows it silently drops 90% of a long file. Pin language per file or reset per chunk.
4. ANE/CoreML is chip-, OS- and concurrency-sensitive: FluidAudio #951, #739, #661; onnx-asr #99 (CoreML EP and MLProgram failures). CPU is often the working fallback.
5. Python packaging: sherpa-onnx #3689 (uv lock drift) and parakeet-mlx #49 (MLX version breakage).

## Unresolved questions

- Python 3.14: no direct evidence for parakeet-mlx, onnx-asr, sherpa-onnx wheels. Need pip/PyPI check (other slice).
- Parakeet v3 Vietnamese support: not checked here.
- FluidAudio diarization DER: no number in any issue opened. Need paper/benchmark (other slice).
- Qwen3-ASR and WhisperKit M1 speed: no numbers in issues opened.
- X: not researched (timeouts).
- Outcomes of parakeet-mlx #27, #43; onnx-asr #18 maintainer reply; FluidAudio #594 root cause.

## UNOPENED LEADS

1. https://github.com/FluidInference/FluidAudio/issues/415 — 29 comments, Swift data race on Xcode RC26.4
2. https://github.com/FluidInference/FluidAudio/issues/513 — OfflineDiarizerManager fails to identify speakers (diarization quality)
3. https://github.com/FluidInference/FluidAudio/issues/253 — inaccurate diarization on iOS/macOS (DER proxy)
4. https://github.com/argmaxinc/argmax-oss-swift/issues/301 — WhisperKit slower than whisper.cpp (speed comparison)
5. https://github.com/argmaxinc/argmax-oss-swift/issues/226 — how to get auto language (LID workaround)
6. https://github.com/argmaxinc/argmax-oss-swift/issues/539 — greedy sampler bug gives empty transcript at temp > 0
7. https://github.com/Blaizzy/mlx-audio/issues/331 — GLM-ASR-Nano slow inference and memory leak on Mac
8. https://github.com/k2-fsa/sherpa-onnx/issues/1792 — Kokoro via CoreML provider (only CoreML EP data point in sherpa)
9. https://github.com/senstella/parakeet-mlx/issues/49 — MLX 0.30.6 breaks mx.concatenate (Python dependency pin)
10. https://github.com/FluidInference/FluidAudio/issues/889 — Kokoro ANE SIGSEGV on iOS 27 (ANE stability trend)
