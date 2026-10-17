---
type: Decision
title: Use the codec's own structure as evidence
description: Off-cadence I-frames and P-frame packet sizes are read from ffprobe without decoding and fused as an independent change signal.
tags: [codec, segmentation]
status: stable
generated:
  by: claude-code/opus-5.5
  at: "2026-09-30T10:15:00Z"
resource: src/mediasift/segment.py
---

# Context

The encoder already ran motion estimation over every frame. Its decisions are
stored in the container and cost nothing to read.

# Decision

- An I-frame whose gap to the previous one is < 0.8× the median GOP is an
  encoder scene-cut (`codec_scenecut` reason). Near an existing boundary it
  annotates; alone it becomes a boundary with strength = sensitivity.
- Bytes per non-key packet, normalised by the median, is `activity`.
- `--fast` decodes only I-frames; falls back to full decode when there are
  fewer than max(4, duration/30) I-frames (screen recordings, VP9).

# Consequences

Codec scene-cuts double as a weak ground truth for tuning
([hard-cut floor](/decisions/cut-threshold.md)). Missing for streams with a
fixed GOP and scenecut disabled — the signal then simply adds nothing.
