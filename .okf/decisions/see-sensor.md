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

OCR is then the slowest local pass (about 9 s of 18 s on that reel: 164 accurate Vision reads at
~51 ms; frame extraction 1 s). Tried and rejected: a text-presence pre-check
(`VNDetectTextRectanglesRequest`, 15x faster) missed real text — 63 of 70 regions on python_100s
("import matplotlib"), 11 of 21 on ui_flow ("Password"); 2-4 Vision threads gave 8.7 -> 7.9 s
(Vision serializes internally). Left as is: the card is cached after its first run.

Long files: on the 1h45m podcast the box took 182 s, of which transcription was 99 s (64x) and the
transfer 2 s; speaker embeddings for ~1,900 lines took ~76 s on 4 threads of a 64-core box. Lines
now embed in parallel (one thread each, up to 16 workers, 4 on a polite Mac): same embeddings,
300 lines 11.0 s -> 3.2 s, voice bench still 1.00. Podcast card: 318 s -> 213 s.

A busy box failed the whole card: ComfyUI held 10.6 of 12 GB on the 3060, whisper hit CUDA out of
memory, and `see` exited. Now any box failure (error, timeout) is logged in one line and the Mac
transcribes instead, as it already did when the box did not answer.

# Bigger sound bench: 200 held-out clips, under speech and music (2026-10-10)

`bench/sense/real_sound.py --set full` uses 4 clips of each of the 50 ESC-50 classes from folds 3-5
(the 40-clip set and every earlier change used folds 1-2); `--bed speech|music|song --snr DB` plays
them under LibriSpeech speech, synthetic chords, or a real CC BY song. Scored by script, no LLM.

| condition | named | stray lines |
|---|---|---|
| silence | 161/200 (80%) | 11 |
| under speech, 6 dB | 138/200 (69%) | 55 -> 20 after the fix below |
| under synthetic music, 6 dB | 128/200 (64%) -> 136 | 7 -> 6 |
| under a real song (Kevin MacLeod, CC BY), 6 dB | 124/200 (62%) -> 134 | 48 -> 32 -> 35 (rules below) |

Under speech, syllables the 3 s windows did not mark as speech came back as hits. Where the tagger
hears some speech nearby, each onset is now checked on its own 1 s window and dropped when CED
calls it speech (>= 0.3): stray lines 55 -> 20, names unchanged; silence unchanged (161, 11). A
filter on the ASR's spoken lines was rejected: it removed 44 stray hits but also 36 right ones.
Under music, hits inside long music are dropped as beats and lasting sounds score 0.2-0.3 beside
"Music", below the naming threshold; the right class is still CED's top non-music guess for 44 of
60 clips. Open: a looser naming rule there must be checked on the real song for invented sounds.
On the real song, most stray lines were lone onsets the tagger could not name: the song's own
drums and plucks, leaking because the clips split it into music runs shorter than the 4.5 s beat
rule. A lone hit with tagger score < 0.1 heard while music plays is now dropped: song stray lines
48 -> 32, names unchanged (124); silence unchanged (161, 11), since the rule needs music.

Naming under music: while music plays, a non-music class is named from 0.18 (not 0.25), over at
least 1.5 s (the song's strays at the lower threshold were all 1.0 s runs). Song 124 -> 134 named,
stray lines 32 -> 35; synthetic music 128 -> 136, 7 -> 6; silence and the 40-clip set unchanged.
The bench counts a wrong name on a real clip as neither: on the song, names below 0.3 were wrong
more often than right (23 right, 30 wrong). So every lasting sound below 0.3 now reads "maybe X"
on the card, in silence too: low confidence means the same there. Card see-6.

Silent misses on the 200 clips are mostly tagger confusions (wind as "ocean", a washing machine as
"pump", a can opening as "slap"). One was wording: church bells read "change ringing (campanology)",
right but obscure, so the card says "church bells ringing": silence 161 -> 164 (scoring unchanged).

Tried and reverted: letting hits inside the tagger's speech stretches through to the per-onset
speech check instead of dropping them: speech bed 138 -> 143 named but stray lines 20 -> 27, about
one stray per gain, and real conversation would leak more syllables than this bed.

CED-base retested fairly (3 s windows, the 200-clip bench; the earlier tie was at 1 s on 40 clips):
silence 162 named / 20 stray vs small's 164 / 11; under speech 131 / 27 vs 138 / 20; under the
song 131 / 31 vs 134 / 35. Twice the tagging time (6.4 vs 3.3 s on a 2:44 reel), 87 vs 22 MB, no
gain anywhere (the rules were tuned on small, but none of the gaps favour base). CED-small stays.

# Camera moves (2026-10-10)

~1 fps frames cannot show which way the camera went: Big Buck Bunny q02 ("pan or tilt?") was missed
in every real round. `build_grid` now measures the global shift between each 128x72 luma frame and
the one 0.5 s earlier (phase correlation, sub-pixel), and `events.camera_moves` turns sustained,
one-way shifts (>= 4% of the frame per second, correlation peak >= 0.1, >= 1.5 s) into card lines:
`pic camera tilts down`, or on a screen recording `pic page scrolls down`. Content moving left means
the camera turned right; a subject crossing a still background does not count. Synthetic pan, tilt,
still and editing-jump clips are tests. On real files: Big Buck Bunny's first scenery shot reads
"camera tilts down" (the q02 key); a CS50 line checked by eye is a real pan; a talking-head reel went
from 4 false "pans" at a 1 s minimum to 0 at 1.5 s (one editing punch-in looks like ~1.1 s of motion).
Cost: one small FFT pair per analysed frame. Card see-7.
Real bench r13 (C re-answered on all four videos): B 31/32, C 31/32, 54 vs 58 images; C answered
q02 "tilts down" (sure) from the camera line, first time in any round; its one miss was q07
(cut-back count, missed in r7 too).

Shot letters: q07 asks how many times the film cuts back to one place; the card listed 34 cut times
and the agent had to recognise places across sheets. `events.shot_letters` describes each shot by
its colour layout (9 x 16, averaged over the shot) and gives shots closer than 0.06 the same
letter; the cut line reads "153.5 Y, 160.4 Z, 165.3 Y, ...". Checked by eye on Big Buck Bunny: Y is
the rodents by the log (4 shots), Z the rabbit alone, AA its close-up; after Z the film returns to Y
3 times, the q07 key. A synthetic A-B-A-C-A clip is a test. Card see-8.
Real bench r14 (C re-answered on all four videos): B 31/32, C 31/32, 56 vs 58 images. Big Buck
Bunny 8/8 for the first time in any round: q07 "3" (sure) from the shot letters, q02 "tilts down"
(sure) from the camera line. C's one miss: python_100s q08 (red X duration, variable since r7).

That red X: drawn in over 0.5 s it came out as "motion" (no lifetime) coloured "black", and it went
together with the card under it, so "undone" never matched. Now (1) a "motion" whose net change
covers >= 85% of its region is a change (measured net/union: moving balls 0.20-0.80, the X 0.91);
(2) a change's colour is the most-changed block's (the synthetic toast is red, not "pink"; the lamp
flash white, not "gray"); (3) an overlay >= 2% of the frame that a later change wipes away gets a
lifetime too. The X now reads red, 104.6-107.8 s (key: about 3 s). It still misses the card's line
budget (rank 111 of 318, budget 36; a colour bonus reached rank 43 while lifting 46 groups), so the
ranking was left alone.
Real bench r15 with all of the above (C re-answered on all four videos): C 32/32, B 31/32, 55 vs 58
images: C's first full score in any round. One run; run-to-run noise is 1-3 questions in 32, so the
claim is "no systematic miss left on this bench", not "ahead of the baseline here".

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
