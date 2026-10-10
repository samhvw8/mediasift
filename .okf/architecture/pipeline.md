---
type: Architecture
title: Analysis pipeline
description: How a video becomes a sensor card (`see`, the default since v0.3) or segments, keyframes and a storyboard (`analyze`) — modules, data flow, outputs.
tags: [pipeline, architecture]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-10T18:00:00Z"
resource: src/framesift/see.py
---

# `see`: the default (v0.3)

One cached command; why each part exists and what it measured: [framesift see](/decisions/see-sensor.md).

| stage | module | output |
|---|---|---|
| cache key | `see.content_key` | size + first/last MiB + card version → `~/.cache/framesift/see/<key>/`; URLs via yt-dlp into `cache/url/` (site subtitles replace ASR) |
| picture | `events.build_grid`, `events.detect`, `events.cuts` | 256x144 frames at 15/5/2 fps → 32x18 colour blocks + largest local change → cuts, then space-time regions of change classified change / blink / motion / animating / scene, with reverts linked; global shift per frame → camera pans / tilts (screen: scrolls) |
| sound | `sound.tag`, `sound.hits`, `sound.beds`, `sound.silences`, `sound.bursts`, `sound.quality`, `sound.voices` | CED (sherpa-onnx, 22 MB, CPU) on 3 s windows, spans trimmed to audible audio; counted short sounds with pitch words; lasting sounds; silences; voice letters |
| speech | `asr.transcribe` / `asr.remote` | parakeet-mlx (cache capped) or whisper (words grouped into cues split at commas, pauses or sentence capitals); `--on HOST` sends FLAC over ssh to `framesift asr -` |
| text | `ocr.read` | Apple Vision on changed regions only; captions that repeat speech dropped; platform watermarks named once (`see.strip_watermarks`) |
| images | `see.flow_sheets`, `see.event_sheets`, `decode.frames_at` | ~1 fps flow sheets (uniform on camera, change-weighted on screens), before/after crops; one decode pass or parallel seeks on long files |
| card | `see.card` | `card.md` timeline (`pic` / `snd` / `say`), budgets per section; `sensed.json`; `transcript.txt`; `usage.jsonl` log of calls |
| zoom | `see.look` | frames over a span + a sound strip on the same axis |
| corrections | `feedback.record`, `feedback.learned` | `framesift fix` logs a wrong sound name to `feedback.jsonl`; 3 agreeing fixes add "(usually: X)" on later cards |

Runs politely: `nice 10`, capped pools, audio and picture passes in sequence (`polite.py`).

# `analyze` (v0.2, kept)

# Principle

Analyse dense and cheap, emit sparse and rich. Every frame is examined at
128×72; only frames that add information are extracted at 768 px.

# Data flow

| stage | module | output |
|---|---|---|
| probe + packets (no decode) | `decode.py` | duration, fps, packet sizes, I-frame times |
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
- Agent entry point: `skills/framesift/SKILL.md`
