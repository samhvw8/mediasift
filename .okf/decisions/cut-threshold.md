---
type: Decision
title: Hard-cut absolute floor = 0.12
description: The cut floor was swept against codec scene-cuts across the corpus; the local-outlier test does the real work.
tags: [segmentation, tuning]
status: stable
generated:
  by: claude-code/opus-5.5
  at: "2026-09-30T10:15:00Z"
resource: src/mediasift/visual.py
---

# Decision

A frame is a hard cut when its step distance ≥ 0.12 **and** ≥ local median +
6×MAD over ±5 s, is a local max, and the frame 0.4 s later still differs from
the frame before (flash guard).

# Sweep (recall vs [codec scene-cuts](/decisions/codec-as-evidence.md), ±0.4 s)

| floor | BBB | CS50 | Fireship | Tears of Steel |
|---|---|---|---|---|
| 0.25 | 32 cuts, 22/22 | 0, 0/22 | 7, 3/9 | 27, 10/17 |
| 0.18 | 33, 22/22 | 7, 5/22 | 12, 4/9 | 37, 13/17 |
| **0.12** | 33, 22/22 | 21, 15/22 | 16, 4/9 | 43, 15/17 |
| 0.08 | 36, 22/22 | 23, 16/22 | 21, 7/9 | 43, 15/17 |

0.25 came from synthetic solid colours and missed every same-stage camera
switch in the lecture (true cuts measured 0.14–0.21 vs median 0.004).
