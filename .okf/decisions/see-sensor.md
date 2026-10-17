---
type: Decision
title: mediasift see - a sensor card instead of a storyboard
description: Why v0.3 replaced change-pruned keyframes with one cached command that writes a timeline card (picture events, camera moves, shots, named and counted sounds, speakers, speech, on-screen text) plus ~1 fps frame sheets; the parts, the first measurements, and where each part's detail lives.
tags: [see, sensor, decision]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-10T21:00:00Z"
resource: src/mediasift/see.py
sources:
  - id: council
    resource: .council/20261010-074112
    title: 5-member council (Contrarian, Daily operator, First principles, Agent harness, Film perception) - develop and stress files
  - id: sense-synth
    resource: bench/sense/make_synth.py
    title: Synthetic clips with scripted ground truth
  - id: review
    resource: tests/test_review_fixes.py
    title: Regression tests for the independent code review's findings
---

# Problem

Agents (Claude Code) read text and still images; they cannot hear and cannot take video. v0.2 picked
keyframes by drift and wrote a storyboard. Asked about time and sound it scored 14/26 on scripted
questions, below plain uniform 1 fps sheets plus a transcript (22/26): pruning dropped the sequence
(4 keyframes for a 32 s UI flow) and nothing described non-speech sound.[^sense-synth] It was also
awkward daily: an output folder, 2-3 flags, 63 s for a 2:23 video, and work that made the Mac lag.

# Decision

`mediasift see VIDEO|URL` (also `mediasift VIDEO`): no flags, cached by file content under
`~/.cache/mediasift/see/<key>/`, prints `card.md`. Frames carry the sequence (~1 fps sheets, the
strongest baseline); the card carries what frames cannot: brief events, exact counts, which way the
camera went, which shots repeat, sound, sync, who speaks, small text. The shape came out of a
five-lens council; every part was then kept or cut by measurement.[^council]

| part | what it is | detail |
|---|---|---|
| picture events | every frame (15/5/2 fps) as a 32x18 block grid; change / blink / motion / animating / cut, with lifetimes | [picture](/decisions/picture-sensing.md) |
| camera moves and shot letters | pans, tilts, scrolls from frame-to-frame shift; a letter per place, repeated on a return | [picture](/decisions/picture-sensing.md) |
| sound | CED on 3 s windows; counted short sounds with pitch words; lasting sounds, "maybe X" when unsure; silences; bursts | [sound](/decisions/sound-sensing.md) |
| sync | a sound within 150 ms of a picture change is written on the change ("together with a beep: #3, #6 of the 8") | the only arm that answered the audio-visual question |
| speakers | TitaNet letters per line, one gender per letter | [sound](/decisions/sound-sensing.md) |
| speech | parakeet or whisper, on the Mac or a GPU box over ssh, overlapped with the local passes | [speech and speed](/decisions/speech-and-speed.md) |
| on-screen text | Apple Vision on changed regions; captions that repeat speech and platform watermarks left out | [picture](/decisions/picture-sensing.md) |
| images | ~1 fps flow sheets shaped like the video (uniform on camera, change-weighted on screens, every shot's first frame), before/after crops | [picture](/decisions/picture-sensing.md) |
| `look VIDEO A-B` | frames over a span with a sound strip; the card says when to use it | - |
| `fix` / `feedback` | agent corrections that teach sound names | [sound](/decisions/sound-sensing.md) |

# First measurements

| round | B: uniform 1 fps + transcript | C: mediasift see |
|---|---|---|
| synthetic r1 (first build) | 22/26, 11 images | 24/26, 7 images |
| synthetic r2 | - | 26/26, 7 images |
| real r2 (first build) | 29/32 | 27/32 |
| real r5 | 32/32 | 31/32 |

What moved C to parity on real footage: ~1 fps flow frames and every shot instead of ~24
change-weighted frames per 3 min; uniform frames on camera footage (change-weighting starved a slide
animation while a lecturer walked); only cuts and text as pixel lines on camera footage (2-7 KB of
texture noise removed); 320 px tiles (44% fewer tokens, same accuracy). Current scores and every
later round: [the sense bench](/benchmarks/sense-bench.md).

# Review fixes

An independent review found 12 issues the bench never reached; each has a regression test:[^review]
variable-frame-rate files gave the next state as the "before" frame; rotated phone videos were
squashed; subtitle de-duplication merged real repeated lines; `look` crashed on audio-only files;
the cache ignored `--asr`; a half-finished download or interrupted run could be served as complete
(the card is now written last, atomically); the card claimed sound tagging on too-short audio; a
picture failure waited for speech; remote ASR could hang or choke on a login banner; onset
detection copied whole recordings (~1 GB per hour). Later real-use fixes: test runs no longer write
into the real usage log (`tests/conftest.py`).

# Not measured or open

- Speaker letters on noisy multi-party audio (crosstalk, music under speech): the voice bench is clean
  read speech; the podcast result (two voices, host 373 lines, guest 1,271) has no answer key.
- Sound names: the remaining misses are the tagger's own confusions; only a better tagger or
  confirmed `fix`es close them.
- Timing a gesture to a spoken phrase (phone reel q06) is missed by both arms.
- Screen vs camera is a heuristic (identical-frame share); files over 20 minutes always get the lean
  picture policy.

[^sense-synth]: bench/sense/r1 and r2: `uv run python bench/sense/report.py r1`.
[^council]: The council's shortlist (sound made legible, synthetic bench, event pairs, zero-invocation card) and its stress tests set the order: measure first, expect the simple arm to win.
[^review]: `tests/test_review_fixes.py`, one test per finding.
