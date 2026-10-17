---
type: Decision
title: ASR consumes mediasift's decoded PCM
description: faster-whisper gets the 16 kHz float32 buffer ffmpeg already produced instead of opening the file itself.
tags: [asr, dependencies]
status: stable
generated:
  by: claude-code/opus-5.5
  at: "2026-09-30T10:15:00Z"
resource: src/mediasift/transcript.py
---

# Decision

`transcript.transcribe(pcm)` passes the numpy buffer to faster-whisper.

# Why

faster-whisper 1.2.1 opens files through PyAV and passes `metadata_errors`,
which PyAV 19 removed (`TypeError`). Passing PCM removes the PyAV path, keeps
ffmpeg the single decoder, and avoids decoding the audio twice.

Cost: ~54 s of CPU ASR (`small`) for 6 min of speech on an M-series Mac.
