---
type: Decision
title: Speech and speed - transcription, the GPU box, and keeping the Mac responsive
description: How mediasift see transcribes (parakeet-mlx, QUAN-Transcribe for Vietnamese, or whisper, on the Mac or a GPU box over ssh), how whisper words become cues, how the box's wait overlaps the local passes and falls back when the box is busy, and the resource measurements behind "the Mac must not lag".
tags: [see, asr, whisper, gpu-box, resources, lag]
status: draft
generated:
  by: samhv
  at: "2026-10-17T13:35:00Z"
resource: src/mediasift/asr.py
sources:
  - id: lagprobe
    resource: bench/lagprobe.py
    title: Timer-lateness probe - "does the Mac feel slow while this runs"
  - id: box
    resource: ~/workspace/infra/.okf/servers/me-ubuntu-home.md
    title: gpu-box (RTX 3060) - mediasift installed as an editable uv tool
  - id: quan
    resource: https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0-CoreML
    title: QUAN-Transcribe-1.0 Core ML (Vietnamese CTC on the EmbeddingGemma 2 audio encoder)
---

# Transcription

Auto-picked via the shared `siftasr` library (`~/workspace/libs/siftasr`): parakeet-mlx on Apple Silicon for its 25 languages (MLX buffer cache capped at 0),
Vietnamese through QUAN-Transcribe on Core ML (below), Chinese/Cantonese through `qwen3-mlx` when installed, and whisper (`mlx-whisper`, `faster-whisper` on
CUDA or CPU) for everything else or when language-detection probability is `< 0.7`. With `--on HOST` or `MEDIASIFT_ON`,
the audio goes as FLAC over ssh to `mediasift asr -` on that host[^box], which returns JSON cues and
speaker letters; 1h45m podcast: box transcription 99 s (64x realtime), transfer 2 s. The first box run
of that podcast cost the Mac 390 s of CPU against 785 s all-local, and none of its GPU.

**Whisper cues.** faster-whisper returns words; `words_to_cues` groups them. The plain 8 s cap cut
mid-phrase ("like web..." | "apps with the Django framework"), and "..." counted as a full stop. A
long cue now splits at its last comma, else a pause >= 0.15 s, else before a capital where whisper
began a sentence without a period ("lắm Mình nghĩ"), else at the cap; the first part keeps >= 3 s.
A widest-gap rule left Vietnamese fragments ("Thì cái lời khuyên không"): whisper's word gaps are
nearly uniform. Result: python_100s lines are whole sentences; the 2:44 Vietnamese reel has 0 lines
under 30 characters. Parakeet returns sentences itself.

**Vietnamese: QUAN-Transcribe on Core ML.**[^quan] The EmbeddingGemma 2 audio encoder with a CTC reader,
one forward pass per <= 30 s piece, no decoder loop. It runs through pyobjc's CoreML (coremltools has no
Python 3.14 build), with one function (`len_30s`) for every piece; line times come from the 40 ms CTC frames
(`asr.ctc_words`), then `words_to_cues` (only its pause rules fire: no punctuation, no capitals).
Measured 2026-10-16 on the M1 Pro, against mlx-whisper large-v3-turbo:

| file | QUAN | mlx-whisper | WER vs reference |
|---|---|---|---|
| 47 min talk (clean TTS dub, script SRT as reference) | 18.2 s (156x) | 128 s (22x) | 5.45% vs 2.75% |
| 2:44 reel (real speech, no reference) | 1.3 s (123x) | 12.2 s (13.5x) | about even by reading |

Whole `see` runs, Mac only (box unset): the 47 min talk 259 s -> **120 s** (faster than the 182 s
with the box), peak footprint 3.3 -> 2.2 GB, stutter p99 1.40 ms, none over 16 ms; the 2:44 reel 31 s
(speech no longer the slow part: picture and OCR are). The first build leaked: Core ML returns
autoreleased objects and a Python loop never drains the pool, so each 30 s piece kept ~380 MB and the talk
reached a 36 GB footprint; each prediction now runs in `objc.autorelease_pool()`.

QUAN's extra errors on the talk are deletions (152 vs 22): fillers ("ừ", "ừm": 33) and English names
(ChatGPT, AGI, OpenAI). On the reel it got "rào cản gia nhập ngành" right where whisper did not, and spelled
English loanwords phonetically ("hô tây cà phê nêu" for homestay, cafe, nail). Output is lowercase, without
punctuation, with numbers as words. First use on a Mac downloads 665 MB and compiles once (~28 s); later
runs load in ~1.7 s. The box keeps faster-whisper: there speech hides behind the picture pass, so its
better English terms cost no time; QUAN serves the Mac when speech is the critical path.

The English counterpart (eg2-transcribe reader v2, open) was not adopted: LibriSpeech test-clean / other
4.21% / 10.42% against parakeet v3's ~1.9% / ~3.6%, and it can only write words from a fixed 200,000-word
audiobook lexicon (no "Django", no "ChatGPT"). The 288x Mac app it reproduces has closed weights.

# Overlap and fallback

Local passes run one after another so the Mac never holds a decoder and a speech model at once.
Speech on the box is the exception: the Mac only waits on ssh.

| behaviour | measured |
|---|---|
| audio decoded and tagged first; if the tagger hears speech, the box starts in a background thread while the picture pass and OCR run; speech lines are collected last | 2:44 reel 29 s → 18-20 s, 34 speech lines identical, wait left 0.5-1.6 s |
| audio without speech is never sent | silent screen recordings wait for nothing |
| any box failure (error, timeout) is logged in one line and the Mac transcribes instead | ComfyUI held 10.6 of 12 GB on the 3060; whisper hit CUDA out of memory and the whole card had failed |
| speaker embeddings in parallel on the box | 1h45m podcast 318 s → 213 s |

# The Mac must not lag (16 GB M1 Pro)

| finding | measurement[^lagprobe] | action |
|---|---|---|
| CPU scheduling is not what lags the desktop | a normal-priority 5 ms timer stayed as punctual as idle during a full-speed run (p99 2.6 ms) | plain `nice 10`, pools capped at 4 (`MEDIASIFT_FULL_SPEED=1` skips both) |
| macOS background QoS band | 5x slower (48 s → 256 s), no gain | not used |
| MLX buffer cache | 5.5 GB kept after a 2-min transcription, on a Mac deep in swap | `mx.set_cache_limit(0)`: same speed, swap churn per run 3.9 GB → 125 MB |
| GPU | 74-96% busy during MLX speech | offload with `--on HOST` |
| overlap with the box | lateness p50 1.26 ms, p99 1.7 ms, none over 16 ms (idle 1.26 / 5.8 ms) | kept |
| OCR (Apple Vision) | 50 s reading whole frames on a 2:23 montage; 14.5 s with region of interest | region of interest, ~1 event per 2 s |
| frame grabs | one ffmpeg seek per frame ~30 s for 230; one decode pass with `select` 0.8 s | `decode.frames_at` |

# Long files on the Mac alone (box down)

Measured 2026-10-10 with the GPU box unreachable, so both runs took the fallback: the box check
costs 3 s, then speech runs on the Mac after the picture and OCR passes.[^lagprobe] Memory is the
physical footprint (what Activity Monitor shows): `ps` RSS leaves out MLX's GPU buffers (31 MB vs
1040 MB for a process holding 1 GB).

| file | wall | CPU time | peak memory | stutter (5 ms timer, p99) |
|---|---|---|---|---|
| 47 min Vietnamese talk (mlx-whisper, 23x realtime) | 259 s | ~319 s | 3.3 GB | 2.4 ms, none over 16 ms |
| 1h45m English podcast (parakeet, 25x realtime) | 485 s | ~722 s | 3.3 GB | 3.4 ms, 0.02% over 16 ms |

| phase (podcast) | time | CPU mean | memory mean / max | GPU mean |
|---|---|---|---|---|
| picture, sound, OCR | 165 s | 281% | 1.0 / 2.0 GB | 11% |
| speech (local) | 256 s | 96% | 2.2 / 3.3 GB | 61% |
| card and images | 28 s | 38% | 1.5 / 1.9 GB | 22% |

The Mac stayed responsive (stutter as at idle) on a machine already 12.4 GB into its 12 GB swap
before the run; swap grew 0.6 GB over the podcast run. Overlapping local speech with the picture pass
would cut the wall time to about the longer of the two, but would stack the 2 GB and 3.3 GB peaks
on a Mac already in swap: not done. With the box up the same podcast takes 213 s.

Rechecked 2026-10-12 with the box up (same files, card see-11):

| file | Mac only | with box | Mac peak memory | box GPU busy, VRAM peak |
|---|---|---|---|---|
| 47 min Vietnamese talk | 259 s | 182 s | 3.3 → 1.6 GB | 35 s at 91%, 4.5 GB |
| 1h45m podcast | 485 s | 286 s | 3.3 → 2.7 GB | 74 s at 97%, 4.5 GB |

The box's CPU stayed under 14% and 49 GB of its RAM stayed free. The podcast took 286 s, not 213 s:
the Mac started with 15.3 GB of swap used (12.4 GB two days earlier), and its picture pass took
174 s (115 s in the Mac-only run). Speech was back at 246 s; the Mac waited 26 s for it. The box run
also gave the podcast 2 speaker letters, where the Mac run gave 4.

The run also found a card bug: per-minute speech lines joined voice tags into
`[voices A, male, B, male, male]`. They now list letters (`[voices A, B]`), and `transcript.txt`
lines carry the speaker letter (`B: ...`) when more than one person speaks (card see-11).

# Parallel passes and Core ML (2026-10-16)

Profile of the 47 min talk after QUAN (per step, seconds): OCR 31, CED tagging 27.9, QUAN 21.4, picture decode +
grid 20.7, events 3.7, speaker letters 4.3, language ID 1.1. Almost all of it is model or decoder compute, so a
rewrite in Swift, Rust or Go would buy little; what helped was moving work to the GPU and overlapping passes.

| change | 47 min talk (Mac only) | 2:44 reel | notes |
|---|---|---|---|
| QUAN, passes in sequence | 120 s | 31 s | |
| audio + tagging beside the picture pass; Core ML speech starts when the tags hear speech | 79 s | | peak 2.2 -> 1.8 GB |
| CED-small on Core ML ([sound sensing](/decisions/sound-sensing.md)), mel frames shared | **70-73 s** | **20 s** | CPU time 259 -> 133 s; peak 2.2 GB; stutter p99 2.7 ms, 0.01% over 16 ms |

A run during another session's GPU job took 150 s (OCR 94 s, QUAN 37x): Vision, QUAN and CED share the one
GPU, so a busy GPU slows every pass. MLX speech still runs after the local passes: its 3.3 GB peak would stack
on the picture pass.

| tried | result |
|---|---|
| `-skip_loop_filter all` for the picture decode | 23.3 -> 18.5 s decode on the 1080p talk, nothing on short files, but 7-32% of picture events changed (the reel +15% noise events): not worth a bench re-run for 3% |
| VideoToolbox decode | still slower for this tiny-output scan (as measured before) |
| parakeet-ctc-0.6b-Vietnamese (ONNX int8, onnx-asr) | 1.5x realtime on the CPU and broken syllables on the reel ("đúngôm", "h hômờ tâyà phê"); punctuation is its only gain |
| QUAN + its KenLM (author: ~2 WER points better) | blocked: `kenlm` is a source-only package and the Mac's Command Line Tools cannot link (SDK missing); retry after reinstalling them |
| FluidAudio (Swift, ANE parakeet, 129-210x on M4 Pro) for non-Vietnamese | blocked by the same toolchain; a subprocess to keep in sync if adopted |
| Qwen3-ASR-1.7B as an accurate Vietnamese mode | not adopted: 10-13x realtime; mlx-whisper turbo already gives 2.75% on the dub with `--asr mlx-whisper` |

Model survey behind these choices: `research/261016-model-alternatives/report.md`.

# OCR cost

OCR is now the slowest local pass (~9 s of 18 s on the reel: 164 accurate reads at ~51 ms). A
text-presence pre-check (`VNDetectTextRectanglesRequest`, 15x faster) missed real text (63 of 70
regions on python_100s, 11 of 21 on ui_flow); 2-4 Vision threads gave 8.7 → 7.9 s. Left as is: the
card is cached after its first run.

Card times (before 2026-10-16; see above for the Core ML numbers): 32 s UI clip ~6 s, 2:23 explainer ~20 s, 2:44 reel 18-20 s, 6 min lecture ~56 s,
1h45m podcast 213 s (with the box).

[^box]: The box's install and service notes live in the infra bundle, not here.
[^quan]: `qnguyen3/QUAN-Transcribe-1.0-CoreML` pinned at revision `13c3a2c` in `asr.py`; author's Vietnamese benchmark (avg WER 6.87% vs turbo 17.66%, the turbo average inflated by one set at 59.6%) on the model card of `qnguyen3/QUAN-Transcribe-1.0`. Gemma Terms of Use.
[^lagprobe]: `uv run python bench/lagprobe.py [--csv FILE] -- mediasift see VIDEO --force`; samples (CPU, footprint, swap, GPU) and stamped logs in `bench/results/res/`.
