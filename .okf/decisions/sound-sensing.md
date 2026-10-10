---
type: Decision
title: Sound sensing - naming, counting and hedging sounds, and who speaks
description: Why framesift see tags sound with CED-small on 3 s windows, how short sounds are counted without counting syllables or a song's drums, when names are hedged, how speakers get letters, and how agent corrections teach names - with every tagger and rule that was tried and dropped.
tags: [see, sound, audio, tagger, voices, feedback]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-10T21:00:00Z"
resource: src/framesift/sound.py
sources:
  - id: ced
    resource: https://github.com/RicherMans/CED
    title: CED, distilled AudioSet tagger (sherpa-onnx export, ced-small int8, 22 MB, SHA-pinned in sound.py)
  - id: esc50
    resource: https://github.com/karolpiczak/ESC-50
    title: ESC-50 environmental sounds (CC BY-NC 3.0, Piczak) - the sound bench's clips
  - id: librispeech
    resource: https://www.openslr.org/12
    title: LibriSpeech dev-clean (CC BY 4.0) - the voice bench and the speech bed
---

# Tagger

CED-small[^ced] (int8, 22 MB, CPU via sherpa-onnx) on **3 s windows** every 0.5 s. CED was trained
on 10 s clips: on the 40-clip set 1 s windows named 25/40, 2 s 31/40, 3 s 36/40, 5 s 36/40 with
blurrier times. Lasting sounds are trimmed to the audible part of their span afterwards, which keeps
edges exact (synthetic siren 14.0-18.0 s).

Every tagger sherpa-onnx ships was compared; CED-small wins or ties at the lowest cost:

| tagger | 40 clips | 200 held-out clips (silence / speech / real song) | cost |
|---|---|---|---|
| **CED-small** | 36/40, 3 stray | **164 / 138 / 134** named, 11 / 20 / 35 stray | 3.3 s on a 2:44 reel, 22 MB |
| CED-base | tied at 1 s | 162 / 131 / 131, 20 / 27 / 31 stray | 6.4 s, 87 MB |
| Zipformer-small | 35/40, 6 stray | - | 27 MB |
| CED + Zipformer averaged | 34/40, 5 stray | - | both |

# Counting short sounds (hits)

An onset is a spectral-flux peak or a 15 dB level jump that also falls back 10 dB (a ring or siren
fails the fall-back test; warble inside it fails too). Onsets group by timbre; each group is named by
what CED hears at its onsets, with pitch and decay words.

| rule | the failure it fixes | measured |
|---|---|---|
| beats inside music are not hits, but only for music longer than 4.5 s or with 4+ onsets; short "music" runs around a hit are dropped | through a 3 s window a lone thud reads as music and its hit vanished | synthetic 23/26 → 26/26 |
| **per-onset speech check**: where the tagger hears some speech nearby, each onset's own 1 s window is tagged; CED speech >= 0.3 → dropped | syllables the 3 s windows missed came back as hits | speech bed stray lines 55 → 20, names unchanged; silence unchanged |
| **a lone hit scored < 0.1 while music plays is dropped** | a real song's drums and plucks leaked as "sharp sound" when the clips split it into short music runs | real-song stray lines 48 → 32, names unchanged; silence unchanged |

# Naming lasting sounds (beds)

| rule | why | measured |
|---|---|---|
| everyday classes in the music branch (a church bell) are named, not folded into "music"; one window >= 0.5 is enough for intermittent sounds (a baby's cries) | real-sound bench misses | part of the 25 → 36 / 40 gains |
| while music plays, a non-music class is named from 0.18 (not 0.25), over >= 1.5 s | beside "Music" CED scores real sounds 0.2-0.3; the right class was still its top non-music guess for 44 of 60 clips | real song 124 → 134, synthetic music 128 → 136; the song's strays at 0.18 were all 1.0 s runs |
| **names below 0.3 read "maybe X"** everywhere | on the real song they were wrong more often than right (23 right, 30 wrong); the bench counts a wrong name on a real clip as neither hit nor stray | card wording |
| plain words for obscure AudioSet names ("church bells ringing" for "Change ringing (campanology)") | right but unreadable | silence 161 → 164 (scoring unchanged) |
| a single speaker is said once below the timeline, not `[voice male]` on every line | 34 identical tags on a Vietnamese reel | card wording |

# Speakers

TitaNet-small embeddings (40 MB) per speech line, greedy clustering at 0.6; letters A, B... A
one-line "speaker" in a long talk folds into the nearest main voice; one gender per letter by
majority. Voice bench (8 LibriSpeech dialogues, 2-3 speakers): 1.00 pairwise agreement, right count
8/8. The mean-spectrum heuristic shipped first scored 0.73; CAM++ exports could not separate
speakers under sherpa-onnx. Lines embed in parallel (one thread each, up to 16 workers, 4 on a
polite Mac): same embeddings, 3.4x faster on the GPU box.

# Corrections teach names

`framesift fix VIDEO TIME "label"` logs what the card said there and what it was
(`<cache>/feedback.jsonl`); the skill tells agents to run it only when a frame or the user shows the
truth. A label fixed the same way at least 3 times, by at least 75% of its fixes, shows on later
cards as `throat clearing (usually: cough)`; the tagger's word stays. `framesift feedback` lists
fixes. Runs: one local JSONL file.

# Tried and dropped

| idea | result |
|---|---|
| name a close runner-up class ("A / B" when the second is >= 0.6 of the first) | missed the target cough (ratio 0.45 in context) and added synonym noise to 12 lines |
| replace implausible winners ("Heart murmur" → best other class) | worse labels ("cricket" for thunder) |
| drop hits inside the ASR's spoken lines | removed 44 stray hits but also 36 right ones |
| let hits inside the tagger's speech stretches through to the per-onset check | speech bed 138 → 143 named, stray 20 → 27: one stray per gain |
| CoreML provider for the tagger | 18.8 s vs CPU 4.8-7.4 s for 3 min |

Remaining misses are CED's own confusions (wind as "ocean", a washing machine as "pump", a can
opening as "slap", a bell alarm as "jackhammer"): a better tagger or confirmed fixes, not card rules.
Bench method and every run: [the sense bench](/benchmarks/sense-bench.md).

[^ced]: CED small, int8, pinned by SHA-256 in `sound.py`; fetched once to `~/.cache/framesift/models`.
