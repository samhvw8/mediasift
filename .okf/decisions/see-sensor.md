---
type: Decision
title: framesift see - a sensor card instead of a storyboard
description: Why v0.3 replaced change-pruned keyframes with one cached command that writes a timeline card (picture events, named and counted sounds, speech, on-screen text) plus ~1 fps frame sheets, and how each part was kept or cut by the sense bench.
tags: [see, sensor, bench, audio, ocr, resources]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-10T15:00:00Z"
resource: src/framesift/see.py
sources:
  - id: council
    resource: .council/20261010-074112
    title: 5-member council (Contrarian, Daily operator, First principles, Agent harness, Film perception) - develop and stress files
  - id: sense-synth
    resource: bench/sense/make_synth.py
    title: Synthetic clips with scripted ground truth (UI flow, motion, counting, audio scene, pointing)
  - id: sense-real
    resource: bench/sense/real.py
    title: Real-footage temporal questions written from 2 fps oracle sheets by a separate agent
  - id: lagprobe
    resource: bench/lagprobe.py
    title: Timer-lateness probe for "does the Mac feel slow while this runs"
  - id: ced
    resource: https://github.com/RicherMans/CED
    title: CED, distilled AudioSet tagger (sherpa-onnx export, ced-small int8)
---

# Problem

Agents (Claude Code) read text and still images; they cannot hear and cannot take video. v0.2 picked
keyframes by drift and wrote a storyboard. Asked about time and sound, it failed: on 26 questions
with scripted answers it scored 14/26, below plain uniform 1 fps sheets plus a transcript (22/26),
because pruning dropped the sequence (4 keyframes for a 32 s UI flow) and nothing described
non-speech sound.[^sense-synth] It was also awkward daily: an output folder, 2-3 flags, 63 s for a
2:23 video, and full-priority work that made the Mac lag.

# Decision

`framesift see VIDEO|URL` (also `framesift VIDEO`): no flags, cached by file content under
`~/.cache/framesift/see/<key>/`, prints `card.md`. The shape came out of a five-lens council, and every part
below was then kept or cut by measurement.[^council]

| part | what it is | kept because |
|---|---|---|
| picture events | every frame at 15 fps (5/2 fps for long files) as a 32x18 block grid at 256x144; space-time connected regions of change become `change`, `blink` (changed and back), `motion` (with path), `animating`, `cut`; a change undone later reports how long it lasted | counts and brief events: "flashed 8 times at 1.0, 2.1, 2.9 ..." matched the script exactly |
| sound | CED tagger[^ced] on 1 s windows (music, siren, telephone...), short sounds found by a jump-and-fall-back level test and counted, pitch/decay words, silences that clicks break, burst counts inside a lasting sound, rough voice letters | sound questions went from 4-5/6 to 6/6; the transcript alone cannot answer them |
| sync | a sound within 150 ms of a picture change is written on the change ("together with a beep: #3, #6 of the 8") | the only arm that answered the audio-visual question |
| OCR | Apple Vision on changed regions only (region of interest), text appeared/vanished; burned-in captions that repeat speech are dropped | "Error 503: upstream timeout" from a 0.7 s toast |
| flow sheets | about one frame per second (12 per sheet; 320x180 tiles in 4 x 3, portrait video 180x320 in 6 x 2, same pixels), every shot's first frame; screen recordings change-weighted with still frames dropped, camera footage clock-uniform | uniform 1 fps was the strongest baseline on real footage |
| `look VIDEO A-B` | frames over a span plus a strip of the sound on the same axis | zoom without re-sensing |

## Measured (Sonnet answerers, blind Sonnet graders)

| round | questions | B: uniform 1 fps + transcript | C: framesift see |
|---|---|---|---|
| synthetic r1 (first build) | 26 | 22/26, 11 images | 24/26, 7 images |
| synthetic r2 (final) | 26 | - | 26/26, 7 images |
| real r2 (first build) | 32 | 29/32 | 27/32 |
| real r3-r4 | 32 | 32/32 | 30/32 |
| real r5 (final) | 32 | 32/32 | 31/32 (the miss is an item B also missed once) |

On real footage the writer could only ask visual and speech questions (no agent hears), so parity
there is the bar; the card's gains are the senses B lacks.[^sense-real] Session tokens on real
footage: B ~86k, C ~92k (first build ~98k). What moved C to parity, each a measured change:

- flow frames: ~24 change-weighted frames per 3 min lost to 180 uniform ones; now ~1/s and every shot.
- camera footage: change-weighting starved a slide animation while a lecturer walked; now uniform.
- card size: pixel events on camera footage were texture (2-7 KB of noise); now only cuts and text.
- tiles 384 -> 320 px: 44% fewer tokens per sheet, no accuracy change.

## Resources on a 16 GB M1 Pro

| finding | measurement | action |
|---|---|---|
| CPU scheduling is not what lags the desktop | full-speed run: a normal-priority 5 ms timer stayed as punctual as idle (p99 2.6 ms)[^lagprobe] | plain `nice 10`, worker pools capped at 4 |
| macOS background QoS band | 5x slower (48 s -> 256 s), no measured gain | not used |
| MLX buffer cache | 5.5 GB kept after a 2-minute transcription on top of a 3.1 GB peak, on a Mac already deep in swap | `mx.set_cache_limit(0)`: same speed, swap churn per run 3.9 GB -> 125 MB |
| GPU | 74-96% busy during MLX speech recognition (idle desktop 27-33%) | `--on HOST` / `FRAMESIFT_ON` sends audio as FLAC to `framesift asr -` over ssh |
| CoreML for the tagger | sherpa-onnx CoreML provider 18.8 s vs CPU 4.8-7.4 s for 3 min (measured under load) | CPU |
| OCR | 50 s on a 2:23 montage reading whole frames; 14.5 s with Vision's region of interest | region of interest, budget ~1 event per 2 s |
| frame grabs | one ffmpeg seek per frame (~30 s for 230); one decode pass with `select` 0.8 s; long files use parallel seeks | `decode.frames_at` |

Timings after the changes: 32 s UI clip 1.5 s (6 s with OCR), 2:23 explainer 19 s, 6 min lecture
56 s, 1h45m podcast 9.4 min before the long-file fixes (speech recognition alone ~263 s at 24x
realtime; offload is the answer there).

# Real recordings (sound names, voices) and the GPU box

| check | method | result |
|---|---|---|
| sound names on real recordings | 40 ESC-50 clips (20 classes) spliced at known times, scored by a synonym table, no LLM (`bench/sense/real_sound.py`) | **36/40 (90%) with 3 s tagging windows**, 3 stray lines; 1 s windows: 25/40, 2 s: 31/40, 5 s: 36/40 with blurrier times. CED was trained on 10 s clips, so 1 s windows starved it. Still missed: both coughs, one mouse click, one alarm clock. CED-base tied CED-small at 1 s (25/40) at 4x the size: not adopted. Lasting sounds are trimmed to the audible part of their span afterwards, which keeps edges exact (siren 14.0-18.0 s, ring 23.0-27.2 s on the synthetic scene). Through a 3 s window a lone thud reads as "music" and its hit was dropped as a beat (synthetic fell to 23/26); now only music longer than 4.5 s or with 4+ onsets suppresses hits, and short music runs around a hit are dropped: back to 26/26. |
| voice letters on real speech | 8 LibriSpeech dialogues, 2-3 real speakers, 12 turns (`bench/sense/voices.py`) | TitaNet-small speaker embeddings: 1.00 pairwise agreement, right voice count 8/8. The mean-spectrum heuristic shipped first: 0.73. CAM++ exports could not separate speakers under sherpa-onnx 1.13.8 (same-speaker and different-speaker similarities overlapped). On the 1h45m podcast: two voices (host 373 lines, guest 1,271). |
| `--on gpu-box` | real ssh, RTX 3060 | whisper large-v3-turbo on CUDA, 58x realtime incl. transfer; the host also returns voice letters. 1h45m podcast: 309 s wall, Mac CPU time 390 s (785 s all-local), Mac GPU unused for speech. |

Two pipeline fixes came from the real-sound check: music-branch classes that are everyday
sounds (a church bell) are named instead of folded into "music", and one strong window (>= 0.5)
is enough for intermittent sounds (a baby's cries). A substitution for implausible winners
("Heart murmur" -> best other class) made labels worse ("cricket" for thunder) and was reverted.

# Review fixes (2026-10-10)

An independent review found 12 issues the bench never reached; each fixed, the first four with a
failing test first (`tests/test_review_fixes.py`): variable-frame-rate files gave the next state as
the "before" frame (now `fps=30` before `select`, non-discarding seek with absolute timestamps, and
no cap at the file's average rate); rotated phone videos were squashed (probe swaps width and
height); subtitle de-duplication merged real repeated speech lines; `look` crashed on audio-only
files; the cache ignored `--asr`; a half-finished yt-dlp download or an interrupted run could be
served as complete (card written last, atomically; `url.txt` marks a finished download); the card
claimed sound tagging when audio was too short; a picture failure waited for speech recognition;
remote ASR could hang (ssh keepalive, timeout) or choke on a login banner; onset detection and
pitch words copied the whole recording (~1 GB per hour).

# First real-use run (2026-10-10)

framesift on two downloaded phone videos (a 26 s English reel, a 2:44 Vietnamese TikTok), the kind
of file daily use brings. Benches are all landscape, so these found what they could not:

- **Portrait tiles.** A 9:16 video in a 16:9 tile filled a third of it: same tokens, a small face.
  Tiles now take the video's shape at the same pixel area (portrait 6 x 2 sheets); the burned-in
  captions became readable.
- **Dangling references (camera footage only).** Camera footage lists no plain pixel events, yet
  crops were made for them (the reel: 7 face-zoom crops, none referenced) and flow tiles named their
  ids. On camera footage, crops and tile `#id`s now come only from the events the card lists (shared
  `listed()`). The same rule on screen recordings was a regression: the real bench (r7) lost the
  python_100s emoji-count and red-X-duration questions in 2 of 2 runs, because the unlisted crops
  were how answerers counted brief overlays. Screen recordings keep the old selection, and the card
  says some crops have no line. q04 + q08 right: r2-r6 10/10; r7 rule everywhere 0/4; restored 2/4
  (whisper ASR) and 3/4 (parakeet ASR) — the ASR difference is within noise.
- **One voice.** A single speaker got `[voice male]` on all 34 lines; it is now said once below
  the timeline.

Real-footage bench r7 (all four videos, Sonnet answerers, blind graders): B 31/32, C 28/32. Besides
python_100s (above), C missed Big Buck Bunny q02 (pan vs tilt, missed in r6 too) and q07 (cut-back
count; the CUTS line and landscape sheets are unchanged, so noise). Lesson: a cleanup that removes
images an agent can see is a behaviour change; measure it like one.
After the fix, full round r10 in the daily setup (ASR on the box): B 31/32, C 31/32, C reading
51 images vs 58; C's one miss is the python_100s emoji count. Synthetic round r11 with the final code: 26/26 on
7 images (unchanged from r5).

# Speech cues from whisper (2026-10-10)

faster-whisper (the CUDA box, and the CPU fallback) returns words; `words_to_cues` groups them. Its
8 s cap cut mid-phrase ('like web...' | 'apps with the Django framework'; 'spam and' | 'eggs'), and
it took whisper's hesitation '...' for a full stop. A cue that grows too long now splits at its last
comma, else a pause of at least 0.15 s, else before a capital where whisper began a sentence without
a period ('lắm Mình nghĩ'), else at the cap, always keeping the first part at least 3 s. A first try
that split at the widest gap left fragments on Vietnamese ('Thì cái lời khuyên không'), because
whisper's word gaps are nearly uniform. Result on the real files: python_100s speech lines are whole
sentences again; the 2:44 Vietnamese reel has 0 lines under 30 characters. Parakeet returns
sentences itself and is unaffected. Card version see-4.

Platform watermarks ("TikTok", the creator's "@handle") jump between corners, and OCR read every
jump as new text: 2-4 timeline lines per real reel. They are now named once in the card header and
dropped from events. "TikTok"/"CapCut" are stripped everywhere; an "@handle" only on camera footage
(in a chat recording "@sam" is content). OCR variants of one handle fold into one. Card see-5.
Real bench r12 with both changes (C re-answered on the three speech videos): B 31/32, C 31/32,
54 images vs 58 — no regression.

# Corrections teach sound names (2026-10-10)

The tagger's misses are fixed by use, not rules: `framesift fix VIDEO TIME "label"` logs what the card
said there and what it was (`<cache>/feedback.jsonl`); the skill tells agents to run it only when a
frame or the user shows the truth. A tagger label fixed the same way at least 3 times, by at least
75% of its fixes, becomes a learned name, shown next to the tagger's (`throat clearing (usually:
cough)`) on cards sensed afterwards; it never silently replaces what the tagger heard. A rule-based
runner-up alias was tried first and reverted (it missed the cough and added noise to 12 lines);
learning from fixes needs no threshold tuned on the bench. Runs: one local JSONL file, nothing else.

# Speech on the box overlaps the local passes (2026-10-10)

Passes run one after the other so the Mac never holds a decoder and a speech model at once. Speech
on a GPU box (`--on` / `FRAMESIFT_ON`) is the exception: the Mac only waits on ssh, and that wait
was 11.5 s of a 29 s card on the 2:44 Vietnamese reel. Now the audio is decoded and tagged first;
if the tagger hears speech, the box gets it in a background thread while the picture pass and OCR
run, and the speech lines are collected last. Same reel: 29 s -> 18-20 s, the 34 speech lines
identical, the wait left at the end 0.5-1.6 s. Lag probe during the run: wakeup lateness p50
1.26 ms, p99 1.7 ms, none over 16 ms (idle: 1.26 / 5.8 ms). Audio without speech is never sent.

# Not measured or open

- Voice letters on noisy multi-party audio (crosstalk, music under speech): the bench is clean read
  speech; the podcast result (two voices) is plausible but has no answer key.
- Sound names: 4 of 40 real clips still missed; names stay hedged where doubtful ("thump
  (heartbeat-like)"). Each miss is CED's, not the card's: one cough reads as Throat clearing (Cough
  0.15 vs 0.33 in the track), one as Plop, a mouse click as Computer keyboard, a bell alarm as
  Jackhammer. Tried and reverted: naming a close runner-up class ("A / B" when the second class is
  at least 0.6 of the first). It missed that cough (ratio 0.45) and added synonym noise to 12 other
  lines ("computer keyboard / typing"). Closing these needs a better tagger, not card rules. The other sherpa-onnx tagger family does not
  help: Zipformer-small (27 MB int8) alone named 35/40 with 6 stray lines; averaged with CED it named
  34/40 (both coughs fixed, a sneeze, a cat, a fire and a click lost). CED-small alone stays.
- Screen vs camera classification is a heuristic (identical-frame share); a static podcast can read
  as a screen, so files over 20 minutes always get the lean picture policy.

[^sense-synth]: bench/sense/r1 and r2: `uv run python bench/sense/report.py r1`.
[^sense-real]: bench/sense/real/r2..r5: `uv run python bench/sense/report.py real/r5`.
[^lagprobe]: `uv run python bench/lagprobe.py -- framesift see VIDEO --force`.
[^council]: The council's shortlist (sound made legible, synthetic bench, event pairs, zero-invocation card) and its stress tests set the order: measure first, expect the simple arm to win.
[^ced]: CED small, int8, 22 MB, pinned by SHA-256 in `sound.py`.
