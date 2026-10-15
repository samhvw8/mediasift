---
type: Benchmark
title: The sense bench - how framesift see is measured
description: The four benches that decide every change to framesift see - synthetic clips with scripted answers, real-footage questions written from dense frames (6 videos), sound naming on 200 held-out ESC-50 clips in silence, speech and music, and voice letters on real dialogues - how to run each, the current scores, and the history of rounds.
tags: [bench, evaluation, see, sound, voices]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-10T21:00:00Z"
resource: bench/sense
sources:
  - id: synth
    resource: bench/sense/make_synth.py
    title: Synthetic clips whose script is the answer key
  - id: real
    resource: bench/sense/real.py
    title: Real-footage rounds - writer, answerers, blind grader
  - id: sound
    resource: bench/sense/real_sound.py
    title: ESC-50 sound naming, scored by script
  - id: voices
    resource: bench/sense/voices.py
    title: LibriSpeech dialogues, scored by pairwise agreement
---

# The benches

| bench | what it tests | how it is scored | run |
|---|---|---|---|
| synthetic | 26 questions on 5 clips (UI flow, motion, counting, audio scene, pointing): counts, brief events, sync, sound, order | the script is the key; Sonnet answerers, a blind Sonnet grader | `uv run python bench/sense/prep.py rN C`, then the briefs in `rN/prompts/`; `report.py rN` |
| real footage | 48 questions on 6 videos: 4 landscape (Big Buck Bunny, Tears of Steel, a Fireship explainer, a CS50 lecture) + 2 portrait phone videos (`reel_en`, `tiktok_vi`) | a writer agent sees only 2 fps oracle sheets + transcript and writes the key; arm B = uniform 1 fps sheets + transcript, arm C = framesift; a blind grader | `real.py oracle [names]`, `real.py arms rN`; `report.py real/rN` |
| sound naming | ESC-50 clips spliced at known times over a still picture; 40 clips (folds 1-2) or 200 held out (all 50 classes, folds 3-5); beds: none, LibriSpeech speech, synthetic chords, a real CC BY song | a clip counts as named when a sound line over it carries an accepted word; stray lines are lines over no clip; no LLM | `real_sound.py [--set full] [--bed speech\|music\|song] [--snr 6]` |
| voices | 8 LibriSpeech dialogues, 2-3 real speakers, 12 turns | pairwise agreement of speaker letters, voices found | `voices.py` |

No agent can hear, so real-footage questions test sight and speech only: there the bar is parity
with arm B, and framesift's gains are the senses B lacks. Run-to-run noise on real footage is 1-3
questions in 32: compare repeat runs before calling a change.

# Current scores (measured on card see-10; see-11 changed only speaker letters on long talks)

| bench | arm B: images + transcript | framesift |
|---|---|---|
| synthetic | 22/26, 11 images (r1) | **26/26, 7 images** (r16) |
| real, 4 landscape videos | 31/32, 58 images | **32/32, 55 images** (r15, one run) |
| real, 2 phone videos | 15/16 | 15/16 (r17 + r18; the shared miss is a gesture at a spoken phrase) |
| sound, 200 held-out clips | 0 by construction | silence 164, under speech 138, synthetic music 136, real song 134 (82 / 69 / 68 / 67%) |
| sound, 40 clips | 0 | 36/40 |
| voices | 0 | 1.00 agreement, 8/8 voice counts |

# History of real-footage rounds

| round | B | C | what changed in C |
|---|---|---|---|
| r2 | 29/32 | 27/32 | first build |
| r3-r4 | 32/32 | 30/32 | uniform frames on camera footage, ~1 fps flow |
| r5-r6 | 32/32 | 31/32 | lean camera cards, 320 px tiles |
| r7 | 31/32 | 28/32 | listed-only crops applied to screens: a regression |
| r10 | 31/32 | 31/32 | screen crops restored |
| r12 | 31/32 | 31/32 | whisper cue splitting, watermarks |
| r13 | 31/32 | 31/32 | camera moves: q02 right |
| r14 | 31/32 | 31/32 | shot letters: Big Buck Bunny 8/8 |
| r15 | 31/32 | 32/32 | drawn-in overlays, change colours |
| r17-r18 | phones 15/16 | 14 → 15/16 | zoom hint, single-cut wording |

Lessons from the rounds: a cleanup that removes images an agent can see is a behaviour change and
must be measured like one (r7); a bench that counts only "named" and "stray" hides confident wrong
names (sound); synthetic music is too easy a bed, so invented sounds are counted on a real song.

Raw records (keys, answers, grades, logs, result JSON) are committed under `bench/sense/`; media is
regenerated. Why each rule exists: [picture](/decisions/picture-sensing.md),
[sound](/decisions/sound-sensing.md), [speech and speed](/decisions/speech-and-speed.md).
