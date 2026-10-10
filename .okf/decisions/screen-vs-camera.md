---
type: Decision
title: Screen vs camera detection
description: Classify content by the fraction of exactly identical consecutive frames; it sets drift, area weight and keyframe gap.
tags: [keyframes, heuristics]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-09-30T10:15:00Z"
resource: src/framesift/visual.py
---

# Decision

`screen` iff ≥ 25% of consecutive analysis frames have step distance < 0.001.

| content | drift | area weight | min gap |
|---|---|---|---|
| screen | 0.06 | 1.0 | 0.4 s |
| camera | 0.12 | 0.5 | 1.0 s |

Overridable with `--content` and `--drift`.

# Evidence

| video | flat blocks | identical frames | truth |
|---|---|---|---|
| 3 screen recordings | 0.24–0.69 | 0.70–0.85 | screen |
| Fireship montage | 0.67 | 0.45 | screen/graphics |
| CS50 lecture | 0.28 | 0.06 | camera |
| BBB, Tears of Steel, 2 phone clips | 0.00–0.03 | 0.00–0.05 | camera |

The first heuristic (flat-block fraction ≥ 0.15) misclassified the lecture
(dark stage) as screen and produced 158 keyframe candidates for 6 minutes.

# Revisit when

A screen recording with a live webcam overlay or constantly animated content
(games) — identical-frame fraction will drop.
