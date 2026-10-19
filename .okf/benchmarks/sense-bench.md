---
type: Benchmark
title: The sense bench - how mediasift see is measured
description: The four benches that decide every change to mediasift see - synthetic clips with scripted answers, real-footage questions written from dense frames (6 videos), sound naming on 200 held-out ESC-50 clips in silence, speech and music, and voice letters on real dialogues - how to run each, the current scores, and the history of rounds.
tags: [bench, evaluation, see, sound, voices]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-18T16:15:00Z"
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
  - id: harmony
    resource: bench/sense/harmony_rhythm.py
    title: Harmony, key/mode shift, BPM, and cut-to-beat sync bench
  - id: emotion
    resource: bench/sense/segment_emotion.py
    title: Speaker-normalized segment talk emotion bench (RAVDESS + calm/VN controls)
  - id: modes
    resource: bench/sense/mode_bench.py
    title: Adaptive, probe, and preset sensing modes + incremental cache upgrade bench
  - id: codec_motion
    resource: bench/sense/codec_motion_bench.py
    title: Codec-aligned motion-compensated P-frame residuals & intra-bucket peak snapping bench (HF MotionBench / TempCompass / Reels; visual artifact at https://mediasift-see-14.art.3cxo.work/)
---
# The benches

| bench | what it tests | how it is scored | run |
|---|---|---|---|
| synthetic | 26 questions on 5 clips (UI flow, motion, counting, audio scene, pointing): counts, brief events, sync, sound, order | the script is the key; Sonnet answerers, a blind Sonnet grader | `uv run python bench/sense/prep.py rN C`, then the briefs in `rN/prompts/`; `report.py rN` |
| real footage | 48 questions on 6 videos: 4 landscape (Big Buck Bunny, Tears of Steel, a Fireship explainer, a CS50 lecture) + 2 portrait phone videos (`reel_en`, `tiktok_vi`) | a writer agent sees only 2 fps oracle sheets + transcript and writes the key; arm B = uniform 1 fps sheets + transcript, arm C = mediasift; a blind grader | `real.py oracle [names]`, `real.py arms rN`; `report.py real/rN` |
| sound naming | ESC-50 clips spliced at known times over a still picture; 40 clips (folds 1-2) or 200 held out (all 50 classes, folds 3-5); beds: none, LibriSpeech speech, synthetic chords, a real CC BY song | a clip counts as named when a sound line over it carries an accepted word; stray lines are lines over no clip; no LLM | `real_sound.py [--set full] [--bed speech\|music\|song] [--snr 6]` |
| voices | 8 LibriSpeech dialogues, 2-3 real speakers, 12 turns | pairwise agreement of speaker letters, voices found | `voices.py` |

| harmony & rhythm | 18 chords/modes (`major`/`minor`/`drone`), 10 harmonic shifts, 10 BPMs, 8 speech/ambient false-BPM controls, 6 cut-sync cases | deterministic script judge | `uv run python bench/sense/harmony_rhythm.py` |
| segment emotion | 12 RAVDESS actors (48 segments) + 4 calm multi-turn controls + 1 real VN narration clip | macro F1 across `calm`/`heated`/`animated`/`subdued` + 0 false flags | `uv run python bench/sense/segment_emotion.py` |
| sensing modes | `resolve_passes` (10 cases), `classify_signal` (6 real videos), `probe` vs `full`, `podcast` vs `full`, and incremental `+ocr` upgrade | deterministic script judge | `uv run python bench/sense/mode_bench.py` |
| codec motion, layout & persistent text | 30 real clips across 8 HF datasets (`MotionBench`, `TempCompass`, `reels-clips`, `ig-reels`, `funa21/tiktok-data`, `Shofo/shofo-tiktok-general-small`, `videoSALMONN2`, `nyu-visionx/vstat`, `facebook/show3d-dataset`) + `reel_en` / `tiktok_vi` gesture apexes + layout (11 cases) + persistent headline OCR (10 cases) + `decode.packet_activity` | deterministic script judge | `uv run python bench/sense/codec_motion_bench.py` |
No agent can hear, so real-footage questions test sight and speech only: there the bar is parity
with arm B, and mediasift's gains are the senses B lacks. Run-to-run noise on real footage is 1-3
questions in 32: compare repeat runs before calling a change.

# Current scores (measured on card see-10; see-11 changed only speaker letters on long talks; see-12 added music mood and genre words, rechecked on the sound bench only)

| bench | arm B: images + transcript | mediasift |
|---|---|---|
| synthetic | 22/26, 11 images (r1) | **26/26, 7 images** (r16) |
| real, 4 landscape videos | 31/32, 58 images | **32/32, 55 images** (r15, one run) |
| real, 2 phone videos | 14/16 (r20) | **16/16** (r20, `see-14`: `reel_en` **8/8**, `tiktok_vi` **8/8**) |
| sound, 200 held-out clips | 0 by construction | Core ML tagger (2026-10-16): silence 163, under speech 144, synthetic music 143, real song 138 (82 / 72 / 72 / 69%), stray 17 / 21 / 7 / 32; sherpa on the same code 164 / 139 / 138 / 134, stray 11 / 20 / 6 / 35; see-13 real song unchanged (138, stray 32) |
| sound, 40 clips | 0 | 36/40 (5 stray; sherpa 3) |
| voices | 0 | 1.00 agreement, 8/8 voice counts |
| harmony & rhythm | 0 | **18/18** mode, **10/10** shift, **10/10** BPM, **0/8** false BPM, **6/6** cut sync |
| segment emotion | 0 | **79.8% macro F1** (speaker-normalized prosody + CED vocal bursts + pauses vs 14.2% unnormalized), **0/5** false flags |
| sensing modes | - | **10/10** pass resolution, **6/6** adaptive signal classification, **67% faster** `probe` on `python_100s`, **66% faster** & **61% fewer lines** `podcast` on `cs50_lecture` |
| codec motion, layout & persistent text | - | **3/3** MotionBench pan recall, **0/3** still FP, **40.2%** pan residual reduction, **1.47x** intra-bucket peak gain (30 clips), **2/2** gesture apex hits, **11/11** layout classification, **10/10** persistent headline OCR recall (vs 1/10 change-only) |

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
| r19 | phones 13/16 | **15/16** | `look --say "phrase"`, harmony/rhythm, speaker-normalized emotion (`see-13`) |
| r20 | phones 14/16 | **16/16** | motion-compensated P-frame residuals, intra-bucket peak snapping (`flow_sheets` & `look`), and `look` Codec Canvas active-region zoom row (`see-14`) |

Lessons from the rounds: a cleanup that removes images an agent can see is a behaviour change and
must be measured like one (r7); a bench that counts only "named" and "stray" hides confident wrong
names (sound); synthetic music is too easy a bed, so invented sounds are counted on a real song.

Raw records (keys, answers, grades, logs, result JSON) are committed under `bench/sense/`; media is
regenerated. Why each rule exists: [picture](/decisions/picture-sensing.md),
[sound](/decisions/sound-sensing.md), [speech and speed](/decisions/speech-and-speed.md).
