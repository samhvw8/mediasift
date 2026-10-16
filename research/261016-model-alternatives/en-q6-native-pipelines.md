# Q6: Native (Swift/Rust/Go/C++) on-device audio pipelines as subprocess CLI (EN, 2026-10-16)

## Bottom line
- No native stack beats current Vietnamese path (QUAN CoreML 156x RT, 6.87% avg). Best native Vietnamese ASR candidates are far slower (FluidAudio Cohere 1.55x RT) or CPU-only (Moonshine Tiny 9.4%).
- Real GEM = **FluidAudio `fluidaudiocli`** (ANE-resident Parakeet v3/Ultra + pyannote-community-1 diarization + Silero VAD) and **Kesha Voice Kit** (Rust binary wrapping FluidAudio; JSON + word timestamps + audio language-ID + diarization). Both would replace parakeet-mlx (~25x) with ~150-190x RT, and speakers/LID.
- Apple SpeechAnalyzer: **no Vietnamese** (verified locally, see below). yap/speech-analyzer-cli = dead end for vi.
- No native stack here offers sound tagging (CED replacement). Keep sherpa-onnx CED.

## Coverage: GEM counts per sub-area
| Sub-area | GEM | Notes |
|---|---|---|
| (a) Swift/ANE | 3 | FluidAudio CLI, Kesha (Rust wrapper over FluidAudio), argmax-cli (WhisperKit+SpeakerKit+serve) |
| (b) C++/ggml | 1 | Moonshine (vi 9.4%, MIT, CPU). whisper.cpp CoreML = MEH, sherpa-onnx CoreML = NOISE for speed |
| (c) Rust/Go bindings | 1 | Kesha engine (Rust). parakeet-rs / transcribe-rs = MEH (ONNX, no CoreML ANE evidence) |
| (d) Apple SpeechAnalyzer CLIs | 0 | vi unsupported, verified |
Total 5 GEM >= 3 threshold, no extra search round required. (I did extra checks on Vietnamese anyway.)

## Verified local fact (GEM, primary evidence)
Ran a Swift probe on this Mac (macOS 27.0.1): `SpeechTranscriber.supportedLocales` = 45 locales (en_*, de_*, es_*, fr_*, it_*, ja_JP, ko_KR, zh_CN/HK/TW, yue_CN, pt_*, hi_IN, ta_IN... ). `vi: false`. Probe: scratchpad/sw/main.swift. (Old SFSpeechRecognizer lists vi-VN but that is the server/legacy API, [SO](https://stackoverflow.com/questions/45407986/does-speech-framework-recognize-only-english).)

## Candidate table
| Tag | Project | Lang | License | Stars / last push | Models | ANE? (evidence) | Speed | Output | Install | Risk |
|---|---|---|---|---|---|---|---|---|---|---|
| GEM | [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio) | Swift | Apache-2.0 (models MIT/Apache; Sortformer = NVIDIA Open Model License) | 3102 / 2026-10-15; releases v0.17.7 (10-08), ~weekly | Parakeet TDT v3 / Ultra / Redux / v2 / Phonon-2 / TDT-CTC-110M / JA / EOU streaming, Cohere Transcribe (14 langs incl vi), SenseVoice, pyannote community-1 offline diarizer (WeSpeaker+VBx), LS-EEND, Sortformer, Silero VAD, TTS (Kokoro, Pocket), LocalVQE | YES: README "inference offloaded to ANE, avoiding GPU/MPS entirely"; Models.md says "ANE" per model; Kokoro v3 explicitly documented as NOT ANE-only (honest scoping) | README: ~190x RT on M4 Pro (1 h in ~19 s); Phonon-2 159x vs v3 149-152x RTFx on ANE (chip unstated in Models.md); no M1 Pro number published | `fluidaudiocli transcribe a.wav [--output-json out.json]` (word timings in JSON only for `--model-version unified`, English); `process a.wav --output r.json` (diarization JSON); `vad-analyze` | **No release binary** (release assets empty); `swift build -c release` from source, or SwiftPM dep. Models auto-download from HF (Parakeet Ultra int8 595 MB) | Medium: v0.x, fast API churn, 49 open "Model Support Requests". No sound tagging, no LID (LID absent from Models.md grep). Word timestamps only on one model. |
| GEM | [drakulavich/kesha-voice-kit](https://github.com/drakulavich/kesha-voice-kit) | Rust engine (~65 MB single binary) + Bun CLI wrapper | MIT | 77 / 2026-10-14 | Parakeet TDT v3 (25 European langs), audio language ID, Silero VAD, diarization (`--speakers`, macOS only), TTS | YES on macOS arm64 (README table "CoreML / ANE" via FluidAudio per FluidAudio README row); Linux/Win ONNX CPU | "up to ~19x faster than Whisper on Apple Silicon" (self-claim, UNVERIFIED independent) | `kesha --json --timestamps a.ogg`; `words` array in `--json` on all OS; MCP server | `bun add -g @drakulavich/kesha-voice-kit` or brew/Docker; `kesha install` ~0.6 GB on Apple Silicon | Medium-high: 77 stars, single maintainer, needs Bun. **No Vietnamese** (25 European only). Useful as proof/shortcut for other langs + LID + speakers. |
| GEM | [argmaxinc/argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift) (WhisperKit + SpeakerKit + TTSKit) | Swift | MIT | 6405 / 2026-10-11; v1.1.1 (10-08) | Whisper family CoreML incl large-v3-turbo 626 MB; SpeakerKit = pyannote v4 community-1 CoreML | YES (CoreML Whisper encoder on ANE; paper: encoder latency measured "on M3 Max on the Neural Engine", [arXiv 2507.10860](https://arxiv.org/html/2507.10860v1)) | Paper table: original Whisper encoder 612 ms on M3 Max ANE per 30 s window; MacParakeet blog (secondary): large-v3 "15-30x RT with CoreML" . No M1 Pro number verified. First ANE compile 85-98 s for Turbo (keybumps issue, secondary) | `brew install whisperkit-cli`; `argmax-cli transcribe --audio-path ...`; `argmax-cli serve` = OpenAI-Audio-API-compatible local server with word + segment timestamps, language detection, SSE | **brew** (`whisperkit-cli`), server needs `BUILD_ALL=1 swift build` | Low: biggest community, MIT, active. But Whisper large-v3-turbo via mlx-whisper already in framesift; gain = ANE offload (less CPU/GPU contention) not accuracy. Vietnamese = Whisper-turbo quality (framesift measured 2.75% on dub) |
| MEH | [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine) | C++ core, Python/Swift/etc bindings | MIT (models MIT; legacy non-streaming non-English = community non-commercial) | 11239 / 2026-10-02 | Streaming Tiny/Small/Medium; **Vietnamese Tiny Streaming 34M, 9.4% WER (FLEURS+LSVSC, 400-clip sample)** | NO (own C++ runtime / ORT; no ANE claim found) | no vi speed number found; designed for low-latency streaming | `pip install moonshine-voice` (Python wheel; check cp314 wheel), `moonshine-voice mic` | pip | 9.4% vs QUAN 6.87% avg: worse. Only upside is tiny size and a 2nd vi opinion. Python 3.14 wheel UNVERIFIED |
| MEH | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | C++ | MIT | 54420 / 2026-10-12 | Whisper | YES for encoder only via Core ML ("more than x3 faster vs CPU-only", README); ANE first-run compile slow | README claim only; framesift already has faster mlx-whisper path | `whisper-cli -oj -ojf` JSON full w/ word timing | brew `whisper-cpp` / build | Redundant with mlx-whisper; Metal GPU path usually preferred |
| MEH | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (12 language bindings incl Go, Rust via C API) | C++ | Apache-2.0 | 15302 / 2026-10-15; v1.13.8 | ASR, diarization, VAD, KWS, audio tagging, LID, TTS | `--provider=coreml` exists but **slower than CPU**: [issue #2910](https://github.com/k2-fsa/sherpa-onnx/issues/2910) FireRedASR M2 macOS 26: RTF 0.457-0.47 coreml vs 0.372 CPU; [#3099](https://github.com/k2-fsa/sherpa-onnx/issues/3099) funasr-nano coreml errors; [#3840](https://github.com/k2-fsa/sherpa-onnx/issues/3840) macOS<15 crash on shared libs. Matches framesift's own 18.8 s CoreML-EP result | n/a | already in use | already in use | Not an ANE route. Keep CPU. |
| MEH | [altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs) | Rust (ort + experimental pure-Rust burn) | MIT | 410 / 2026-10-13; v0.4.1 | Parakeet TDT, Nemotron 3.5 multilingual streaming (40 locales), Cohere, Sortformer v3 diarization (8 spk), word/sentence timestamps | NO ANE evidence: README lists cuda/tensorrt/webgpu/directml EPs, no CoreML feature mentioned | not given | Rust lib, examples (not a polished CLI) | cargo | CPU ORT only on Mac: no win vs parakeet-mlx |
| MEH | [Frikallo/parakeet.cpp](https://github.com/Frikallo/parakeet.cpp) | C++ (axiom) | MIT | 306 / 2026-05-04 (stale 5 months) | Parakeet 110M/600M TDT-CTC, EOU, Sortformer, diarized transcriber, word timestamps | NO: Metal GPU via MPSGraph ("96x speedup" vs its own CPU, self-claim) | self-claim | C++ examples | build | stale, small |
| MEH | [cjpais/transcribe-rs](https://github.com/cjpais/transcribe-rs) | Rust | MIT | 274 / 2026-10-02 | Engine wrapper: Parakeet, Canary, Cohere, Moonshine, SenseVoice, GigaAM (ORT), Whisper via whisper.cpp Metal | NO (ORT CPU + whisper.cpp Metal GPU) | none | lib, powers Handy | cargo | Library not CLI; 0.3.0 "large migration" |
| MEH | [huggingface/candle](https://github.com/huggingface/candle), [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Rust | Apache/MIT | 21173 / 7751 | candle examples: whisper, voxtral, silero-vad; no parakeet/moonshine example | NO: Metal GPU only, no CoreML | n/a | n/a | cargo | No vi benefit; skipped deeper |
| NOISE | [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples), [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | Swift | MIT/Apache | 2682 (last push 2026-07-20) / 1360 | MLX = GPU, not ANE | NO | | | | Not ANE |
| MEH | [pykeio/ort](https://github.com/pykeio/ort) CoreML EP | Rust | MIT/Apache | 2565 | any ONNX | CoreML EP partial-graph, same pitfalls as sherpa #2910 | | | | no |
| NOISE | [finnvoor/yap](https://github.com/finnvoor/yap), [simonw/speech-analyzer-cli](https://github.com/simonw/speech-analyzer-cli) (23 stars), DravenYe/swift-speech-analyzer (1 star), FluidInference/swift-scribe (313) | Swift | - | yap 1628 / 2026-07-20 | Apple SpeechTranscriber | Apple-managed (not controllable) | ~38x RT claim in a third-party skill (fuyao-mac-stt-lite, UNVERIFIED) | yap: `brew install yap`; `--srt --vtt --json --word-timestamps`; good CLI shape | brew | **vi_VN absent from supportedLocales (verified)**. Good for en/zh/ja/ko/es/fr/de only; no diarization/LID/tagging |
| NOISE | Handy (33884 stars, Tauri app, uses transcribe-rs), vibe (7796, whisper.cpp), VoiceInk (6819, Swift app) | - | - | - | apps, not CLIs | - | - | - | - | Apps, not subprocess-able |
| MEH | FluidAudio Cohere Transcribe (inside FluidAudio) | Swift | Cohere model: Apache-2.0 per release announcement (verify) | n/a | 14 langs incl **Vietnamese**; INT8 encoder 1.8 GB + ANE decoder | YES (ANE decoder, ANE cold-start compile ~6 min) | **Vietnamese FLEURS WER 9.55%, CER 6.87%, RTFx 1.55x on M4 Pro**; en LibriSpeech test-clean 1.77% WER 2.04x on M2 ([Cohere.md](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ASR/Cohere.md)); 35 s hard cap per call | via fluidaudiocli cohere-benchmark (no general transcribe doc seen) | ~100x slower than QUAN (156x) for a worse vi WER than QUAN's. Dead for vi. |

## Key facts for framesift
1. Vietnamese remains QUAN. Native alternatives: Cohere/FluidAudio 9.55% FLEURS @1.55x RT; Moonshine Tiny 9.4% CPU; Apple: none; Parakeet v3/Ultra: no vi (25 European langs).
2. Non-Vietnamese ASR upgrade path: `fluidaudiocli` / FluidAudio Parakeet Ultra on ANE (LibriSpeech test-clean 2.13%, test-other 3.81%, FLEURS 24-lang mean 11.67% vs v3 14.81%; ANE int8 595 MB) vs parakeet-mlx ~25x. FluidAudio claims 150-190x on ANE. Gain ~6x if it holds on M1 Pro (UNVERIFIED on M1 Pro; ANE gen on M1 is older, M4 Pro number likely optimistic). Also frees GPU/CPU.
3. Diarization: FluidAudio offline = pyannote community-1 (segmentation + WeSpeaker + VBx) with `process --output json`, RTTM scoring; SpeakerKit = pyannote v4 community-1 CoreML. Both are the upgrade over TitaNet-small+greedy clustering. Benchmark numbers not captured here (see FluidAudio Documentation/Benchmarks.md; not fetched).
4. LID: Kesha has audio LID (CoreML); FluidAudio has none seen; SenseVoice (50+ langs auto-detect, no vi listed in its FunASR set: zh/en/ja/ko/yue) and Whisper-via-argmax `detect language`.
5. Sound tagging: none of these. SenseVoice emits `<|event|>` tags only (speech/music/applause type), not 527-class AudioSet. Keep CED.
6. Distribution cost: FluidAudio has no prebuilt release asset; framesift would need a swift build step or Homebrew tap (none found). argmax `brew install whisperkit-cli` is the only brew-ready one; yap too (but no vi).
7. Subprocess shape: all emit file JSON, so subprocess + JSON is viable; startup cost matters (ANE first-compile: Turbo 85-98 s, Cohere ~6 min, Redux "several minute first compile"); the compiled cache persists after first run (per whisper.cpp README "first run slow").

## UNVERIFIED / social-only
- Kesha "19x faster than Whisper": self-claim.
- "~38x RT" SpeechAnalyzer: third-party repo README, not measured by me.
- Reddit snippets (r/LocalLLaMA, Parakeet "Metal GPU higher peak throughput than ANE") seen in search result snippets only; Reddit not fetched per instruction.
- HN: only Show HN titles seen for FluidAudio (id 44546863, 48415709); comments not read. X search timed out.

## Unresolved
- FluidAudio RTFx on M1 Pro actual (not published; would require clone + build + model download to measure).
- Whether M1 Pro ANE (16-core, older gen) runs Parakeet v3 near the 150-190x quoted (M2/M4 numbers).
- Moonshine Vietnamese RTF and Python 3.14 wheel availability.
- FluidAudio diarization DER numbers vs TitaNet-small baseline.
- Cohere Transcribe license for FluidAudio's CoreML conversion.
- Whether FluidAudio's `transcribe` JSON gives word times for non-unified (multilingual) models.

## Sources
- https://github.com/FluidInference/FluidAudio (README, Documentation/CLI.md, Models.md, ASR/Cohere.md), releases v0.17.7
- https://github.com/argmaxinc/argmax-oss-swift README; https://arxiv.org/html/2507.10860v1
- https://github.com/drakulavich/kesha-voice-kit README, docs/languages.md
- https://github.com/k2-fsa/sherpa-onnx issues 2910, 3099, 3840
- https://github.com/ggml-org/whisper.cpp README (Core ML section)
- https://github.com/moonshine-ai/moonshine; https://moonshine-voice.readthedocs.io/en/latest/models/available-models/ and /accuracy/
- https://github.com/altunenes/parakeet-rs, https://github.com/Frikallo/parakeet.cpp, https://github.com/cjpais/transcribe-rs
- https://github.com/finnvoor/yap, https://github.com/simonw/speech-analyzer-cli
- https://news.ycombinator.com/item?id=44546863 (Show HN FluidAudio, title only)
- https://macparakeet.com/blog/whisper-to-parakeet-neural-engine/ (secondary)
- Local probe of SpeechTranscriber.supportedLocales on macOS 27.0.1
