---
type: Decision
title: Keyframes by drift from the last kept frame
description: I/P-frame analogy — keep a new frame once accumulated change since the last kept frame crosses a threshold; localized change measured with a motion-tolerant block test.
tags: [keyframes]
status: stable
generated:
  by: claude-code/opus-5.5
  at: "2026-09-30T10:15:00Z"
resource: src/mediasift/keyframes.py
---

# Decision

- Distance is taken against the **last kept** frame, so slow pans and slides
  that build up bullet by bullet accumulate.
- On crossing `drift`, pick the most stable, sharpest frame in the next 0.6 s;
  skip if it settles back below threshold (flash, bounce).
- `content_distance = max(global, area_weight × changed_area)`, where
  changed_area is the fraction of 4×4 blocks whose luma (>0.04) or edge
  density (>0.03) changed after a ±1-block motion search.
- Minimum gap between keyframes: 0.4 s screen, 1.0 s camera.
- Duplicates (dHash ≤ 5 bits and distance ≤ 0.05) reference an earlier frame.
- Budget drops smallest drifts first, then anchors of least eventful segments.

# Why localized change

A whole-frame mean missed a ChatGPT screen recording entirely: new messages
cover ~10% of a dark UI and the wallpaper diluted them. Adding changed_area
took that video from 6 to 23 keyframes, each a distinct conversation state.

# Known limit

Presenter gestures in close-ups still earn frames; pixel statistics cannot
separate motion from new information. Planned: embedding-distance veto. The model and per-frame vectors now exist
([embedding search](/decisions/embedding-search.md)); the veto itself is not built or measured yet.

See [screen vs camera](/decisions/screen-vs-camera.md) for the per-content parameters.
