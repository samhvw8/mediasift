---
type: Architecture
title: Analysis pipeline
description: How a video becomes a sensor card (`see`, the default since v0.3) or segments, keyframes and a storyboard (`analyze`) — modules, data flow, outputs.
tags: [pipeline, architecture]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-18T16:15:00Z"
resource: src/mediasift/see.py
---

# `see`: the default (v0.3)

One cached command; why each part exists: [mediasift see](/decisions/see-sensor.md), with detail in [picture](/decisions/picture-sensing.md), [sound](/decisions/sound-sensing.md) and [speech and speed](/decisions/speech-and-speed.md); measured by [the sense bench](/benchmarks/sense-bench.md).

| stage | module | output |
|---|---|---|
| cache key | `see.content_key` | size + first/last MiB + card version → `~/.cache/mediasift/see/<key>/`; URLs via yt-dlp into `cache/url/` (site subtitles in the video's own language replace ASR; a subtitle failure never costs the video, see [URL fetch](/decisions/see-sensor.md)) |
| picture | `events.build_grid`, `events.detect`, `events.cuts`, `events.camera_moves`, `events.detect_layout` | 256x144 frames at 15/5/2 fps; active-viewport matte-trimmed outer 25% border-strip consensus shift `(dx, dy)` warps `prev` before computing 32x18 colour blocks + largest local change → motion-compensated residuals, cuts, space-time regions (`change` / `blink` / `motion` / `animating` / `scene`) with lifetimes, rolling-median camera pans / tilts (screen: scrolls), shot letters (`events.shot_letters`), and viewport layout (`letterbox` / `pillarbox` / `split-vertical` / `static-slide`) |
| sound | `sound.tag`, `sound.hits`, `sound.beds`, `sound.silences`, `sound.bursts`, `sound.quality`, `sound.voices` | CED-small on 3 s windows (Core ML on the GPU in batches of 64 when `mise run ced-coreml` built it, else sherpa-onnx on the CPU; numpy mel with frames shared across overlapping windows), spans trimmed to audible audio; counted short sounds with pitch words (syllables and a song's drums excluded); lasting sounds, "maybe X" below 0.3; silences; TitaNet voice letters; music beds carry CED's own mood and genre words when sure, and a mood arc when long ([music and voice mood](/decisions/music-and-voice-mood.md)) |
| speech | `asr.transcribe` / `asr.remote` | parakeet-mlx (cache capped), QUAN-Transcribe on Core ML for Vietnamese (CTC frames give word times), or whisper (words grouped into cues split at commas, pauses or sentence capitals); `--on HOST` sends FLAC over ssh to `mediasift asr -`, overlapped with the picture and OCR passes, falling back to the Mac if the box fails |
| text | `ocr.read`, `see.scan_persistent_text` | Apple Vision on macOS, falling back to Indra OCR (`datalab-to/indra-ocr` ONNX Runtime CPU) on Linux (`MEDIASIFT_OCR=indra`), on changed regions (`_ocr_events`) plus settled full-frame anchors for persistent banners/headlines (`scan_persistent_text`); captions that repeat speech dropped; platform watermarks named once (`see.strip_watermarks`) |
| images | `see.flow_sheets`, `see.event_sheets`, `events.select_bucket_peaks`, `decode.frames_at` | ~1 fps flow sheets (stratified intra-bucket residual peak snapping on camera, change-weighted on screens), before/after crops; one decode pass or parallel seeks on long files |
| card | `see.card` | `card.md` timeline (`pic` / `snd` / `say`) + `Layout:` and `Lasting text (persistent banner/overlay):` lines, budgets per section; `sensed.json` (with `frame_peaks`, `layout`, `lasting_text`); `transcript.txt`; `usage.jsonl` log of calls |
| zoom | `see.look`, `events.snap_look_times`, `see._active_crop_row` | frames over a span (or `--say "phrase"`), snapped to cached `0.25 s` residual peaks + a 4-tile 2x active-region Codec Canvas crop row (`ACT CROP`) + a sound strip on the same axis |
| corrections | `feedback.record`, `feedback.learned` | `mediasift fix` logs a wrong sound name to `feedback.jsonl`; 3 agreeing fixes add "(usually: X)" on later cards |

Runs politely: `nice 10`, capped pools (`polite.py`). Audio decoding and tagging run beside the picture pass; speech starts as soon as the tags hear it when it holds little Mac memory (the box, or Core ML QUAN) and is collected after OCR; MLX speech (3.3 GB) still waits for the local passes. Core ML calls go through `coreml.py` (pyobjc, one autorelease pool per prediction).

# `analyze` (v0.2, kept)

# Principle

Analyse dense and cheap, emit sparse and rich. Every frame is examined at
128×72; only frames that add information are extracted at 768 px.

# Data flow

| stage | module | output |
|---|---|---|
| probe + packets (no decode) | `decode.py` (`packet_activity`, `packets`) | duration, fps, packet sizes, P/B byte-ratio spikes, I-frame times |
| visual track | `visual.py` | per frame: 32×18 luma + edge blocks, 64-bin histogram, sharpness, dHash, step distance |
| audio track | `audio.py` | 50 ms log-mel + RMS, pauses |
| signals | `segment.build_signals` | visual/audio novelty as robust-z, codec activity, silence on one grid |
| boundaries | `segment.find_boundaries` | hard cuts + novelty peaks + codec scene-cuts, snapped to pauses, min/max duration enforced |
| keyframes | `keyframes.py` | drift-from-last-kept selection, duplicate refs, budget |
| speech | `transcript.py` | cues from file or faster-whisper on the same PCM buffer |
| outputs | `render.py` | `storyboard.md`, contact sheets, `manifest.json`, `signals.npz` |

ffmpeg is the only decoder (see [ASR input](/decisions/asr-shares-pcm.md)).

# Search (optional, `[embed]` extra)

| stage | module | output |
|---|---|---|
| embedding model | `embed.py` | EmbeddingGemma 2 (text + vision) run by llama.cpp's `llama-server`, started per command on a private Unix socket and stopped after it; `index --on HOST` runs it on another machine over ssh |
| index | `index.py` | `vectors.npz`: 768-d float16 vectors for keyframes + a 1 fps probe (≤ 600) and for ~8 s speech windows; model identity; `index/` probe JPEGs |
| query | `query.py` | `find` (text or `--like T`) and `look` (exact times) → one labelled grid each in `find/`, `look/`; every call appended to `queries.jsonl` |

Ranking and its measurements: [Embedding search](/decisions/embedding-search.md).

# Related

- [Codec as evidence](/decisions/codec-as-evidence.md)
- [Keyframe drift](/decisions/keyframe-drift.md)
- [Screen vs camera detection](/decisions/screen-vs-camera.md)
- [Hard-cut floor tuning](/decisions/cut-threshold.md)
- [Embedding search](/decisions/embedding-search.md)
- [The sense bench](/benchmarks/sense-bench.md)
- Agent entry point: `skills/mediasift/SKILL.md`
