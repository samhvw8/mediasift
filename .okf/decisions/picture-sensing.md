---
type: Decision
title: Picture sensing - what the card says about the picture, and why
description: How framesift see turns frames into card lines and images - flow-sheet tiles, event crops, camera moves, shot letters, drawn-in overlays, watermarks - with the real-file failure that prompted each rule and the measurement that kept it.
tags: [see, picture, events, camera, shots, ocr]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-10T21:00:00Z"
resource: src/framesift/events.py
sources:
  - id: sense-real
    resource: bench/sense/real.py
    title: Real-footage temporal questions, 6 videos (rounds r2-r18)
  - id: real-use
    resource: ~/Downloads (two phone videos, run locally)
    title: First real-use run on a 26 s English reel and a 2:44 Vietnamese TikTok
---

# Base: events on a block grid

Every frame (15 fps; 5 / 2 fps for files over 10 / 40 min) is read at 256x144 as a 32x18 grid of
colour blocks plus the largest local brightness change per block. Space-time connected regions of
change become `change`, `blink` (changed and back), `motion` (with a path), `animating`, `scene`
or `cut`. A change that a later event undoes reports how long it lasted. Why the frames are uniform
on camera footage and change-weighted on screens: [framesift see](/decisions/see-sensor.md).

# Rules, each from a measured failure

| rule | the failure it fixes | measured |
|---|---|---|
| **flow tiles take the video's shape** at the same pixel area (16:9: 320x180 in 4 x 3; 9:16: 180x320 in 6 x 2) | a phone reel in 16:9 tiles filled a third of each tile: same tokens, a small face[^real-use] | burned-in captions readable at the same sheet size |
| **event crops: screens keep the most telling events, camera footage only events the card lists** (`see.listed`, one crop per group) | a phone reel got 7 face-zoom crops no line pointed to; applying the same rule to screens hid the overlays answerers counted | real r7 lost python_100s q04 + q08 in 2 of 2 runs with the rule on screens; restored, 2-3 of 4 right (r2-r6: 10/10)[^sense-real] |
| **tile `#id`s only for listed events on camera footage** | tiles named events the card never listed | test: tile ids ⊆ listed ids |
| **camera moves**: global shift between 128x72 luma frames 0.5 s apart (phase correlation, sub-pixel); sustained one-way shift (>= 4% of the frame/s, peak >= 0.1, >= 1.5 s) → `pic camera tilts down`; on screens `pic page scrolls down` | ~1 fps frames cannot show direction: Big Buck Bunny q02 "pan or tilt?" missed in every round | q02 right from r13 on; an editing punch-in faked ~1.1 s "pans" on a reel (gone at 1.5 s); a CS50 pan checked by eye; cost 2.2 s on a 1h45m file |
| **shot letters**: each shot's colour layout (9 x 16, averaged over the shot); shots closer than 0.06 share a letter; the cut line reads `153.5 Y, 160.4 Z, 165.3 Y` | "how many times does it cut back to X" needed places recognised across sheets (q07) | checked by eye: Y = rodents by the log, Z = rabbit alone; q07 right in r14 |
| **a single cut back to the opening place** reads `CUT: same place again (shot A, as at the start)` | "CUT: new shot A" on a punch-in read as a contradiction | wording test |
| **drawn-in overlays are changes**: a `motion` whose net change covers >= 85% of its region | a red X drawn in over 0.5 s came out as motion, no lifetime | net/union: moving balls 0.20-0.80, the X 0.91; synthetic balls stay motion (test) |
| **colour of a change = its most-changed block** | the thin red X over a dark page averaged to "black" | synthetic toast now "red" (drawn 200,40,40; was "pink"), lamp flash "white" (was "gray") |
| **an overlay >= 2% of the frame wiped away by a later change gets a lifetime** | the X went together with the card under it, so "undone" never matched | the X reads red, 104.6-107.8 s (key: about 3 s) |
| **platform watermarks named once** in the header ("TikTok", "CapCut" everywhere; an "@handle" only on camera footage; OCR variants folded) | the mark jumps corners and each jump read as new text: 2-4 lines per reel | 0 watermark lines in TikTok timelines |
| **zoom hint**: the card says to `look` for a gesture, a face or small text at one moment | answerers guessed gestures from ~1 fps tiles | reel C 6/8 → 7/8 in both repeat runs (r18) |

# Tried and left out

- **Ranking bonus for coloured overlays**: the red X ranks 111 of 318 groups (card budget 36 for a
  2:23 video); a bonus reached rank 43 while lifting 46 groups. The ranking was left alone; the X's
  lifetime is in `sensed.json` and reaches the card when the budget allows.
- **OCR speed-ups**: see [speech and speed](/decisions/speech-and-speed.md).

[^sense-real]: Rounds under bench/sense/real/r7-r18; method and current scores in [the sense bench](/benchmarks/sense-bench.md).
[^real-use]: Cards for the phone videos in ~/.cache/framesift/see; the videos are now bench items `reel_en` and `tiktok_vi`.
