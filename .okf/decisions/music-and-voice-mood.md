---
type: Decision
title: Music and voice mood - what the card can say about how a video feels
description: Why music beds now carry CED's own mood and genre words (88% right when stated, no new model), a mood arc on long scores, and why per-line speech emotion (emotion2vec+, SenseVoice) and open-vocabulary music words (CLAP, MTG-Jamendo) were measured and left out.
tags: [see, sound, music, mood, emotion, bench]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-17T12:00:00Z"
resource: src/framesift/sound.py
sources:
  - id: audioset-eval
    resource: https://huggingface.co/datasets/agkphysics/AudioSet
    title: AudioSet eval split (held out from CED training), mirrored as parquet; 742 clips with one mood or one genre label
  - id: ravdess
    resource: https://zenodo.org/records/1188976
    title: RAVDESS speech, 24 actors, 8 acted emotions (CC BY-NC-SA 4.0)
  - id: emonet
    resource: https://huggingface.co/datasets/t1a5anu-anon/emonet-voice-bench
    title: EmoNet-Voice bench (May 2025, CC BY 4.0), human intensity ratings; newer than emotion2vec+
  - id: emotion2vec
    resource: https://github.com/ddlBoJack/emotion2vec
    title: emotion2vec+ - the seed model was fine-tuned on EmoBox, which includes RAVDESS
  - id: essentia
    resource: https://essentia.upf.edu/models.html
    title: Essentia Discogs-EffNet and MTG-Jamendo mood/theme heads (ONNX)
  - id: scripts
    resource: research/261017-audio-mood/README.md
    title: Every script, pick list and data source behind the numbers here
---

# Question

Can the card say what the music feels like (epic, sad, scary) and how the speakers feel, cheaply
enough to run on every video?

# Decision

| what | on the card | why |
|---|---|---|
| music mood (CED's 7 AudioSet moods) | `[music] (scary)` when the bed's mean score >= 0.10; `maybe sad` at 0.05-0.10; nothing below | right 88% / ~65% of the time on held-out clips (chance 14%) |
| music genre (CED's genre classes 216-265) | `(scary, electronic)` when >= 0.15 | right 83% (9 genres, chance 11%) |
| mood arc | a bed of 60 s or more lists its sure moods per 15 s stretch, merged, when the mood changes: `(tender 00:01.2-01:01.2, then scary 01:01.2-02:01.2)` | where a score turns is where a film turns |
| beds merged "on and off" | keep each stretch's words, in order: `scary, then sad` | Tears of Steel lost its `scary` in the merge |
| speech emotion | not shown | see below |

Cost: none. CED already scores these classes in every window; the card used to throw them away.
The tagger still keeps its top 12 classes per window (top-1 60% vs 65% with all 527, but the
>= 0.10 rule is unchanged: 35% of clips, 88% right). Runs: nothing new.

# Music: the bench

742 AudioSet eval clips (10 s, 7 moods x 42-53 and 9 genres x 31-53), labels from AudioSet's own
raters.[^audioset-eval] Top-1 among the classes; tags also z-scored over the set so a tag's base rate
cannot win alone.

| method | moods top-1 / macro AUC | genres top-1 / macro AUC | cost per audio minute |
|---|---|---|---|
| **CED-small (already running)** | **65%** / 0.87 | **70%** / 0.91 | 0 extra |
| Discogs-EffNet + MTG-Jamendo mood/theme (56 tags incl. epic, dramatic, trailer), mapped to the 7 | 43-45% / 0.80 | 53-57% / 0.89 (Discogs-400) | 0.2 s, 18 + 3 MB |
| CLAP zero-shot (`laion/clap-htsat-unfused`, "X music") | 28-41% / 0.64 | 38-40% / 0.71 | 0.4 s, 600 MB, torch |

The judge is AudioSet's own taxonomy, home ground for CED; EffNet is read through a hand map (fixed
before results). Still, the open-vocabulary words the user asked for ("epic, heroic") come only from
EffNet or CLAP, and those were wrong more often than CED's plain words on the classes both can name.
They stay out until a judge with those words exists (MTG-Jamendo test split: ~150 GB of audio).

Per-bed precision by score on the same clips: mood >= 0.05 56% of clips / 78% right, >= 0.10 35% /
88%, >= 0.2 21% / 94%; genre >= 0.15 44% / 83%. On real samples the rule is quiet: Tears of Steel
reads `scary` on all three beds, Big Buck Bunny's opening `maybe sad`, a pizzicato track and two
reels nothing. Sound bench after the change: real song 138 named, 32 stray (unchanged).

# Speech emotion: measured, left out

| model | RAVDESS 7-way (bal. acc.) | EmoNet-Voice 6-way, strong-consensus clips | arousal (EmoNet) | cost per audio minute |
|---|---|---|---|---|
| emotion2vec+ base fp32 | 88% (contaminated)[^emotion2vec] | **45%** | 65% | 6.4-7.6 s, 373 MB |
| emotion2vec+ base int8 | 82% | 43% | 63% | 2.7-3.6 s, 94 MB |
| probe on CED scores (trained on RAVDESS actors 1-16) | 56% (actors 17-24) | 26% | 61% | 0 |
| probe on 9 prosody numbers (pitch, loudness, rate) | 42% | 16% (chance) | 51% | ~0 |
| SenseVoice-Small int8 (sherpa-onnx) | - | - | - | every window `NEUTRAL`, language `yue`, also on English and Vietnamese |

RAVDESS is inside emotion2vec+'s training lineage (its seed was fine-tuned on EmoBox), so 88% is an
upper bound. On EmoNet-Voice (newer than the model, 237 clips every rater scored 2) it drops to
45%,[^emonet] and confidence does not help: at p >= 0.95 it is still right 56-60%. On a Vietnamese
reel it printed `surprised 1.00` over calm narration. A label wrong about half the time on the
easiest clips would mislead an agent that cannot hear; the card's rule for sound names (hedge or
drop below even odds) applies.

What the card already says about a voice's state stays: CED names laughter, shouting, yelling,
screaming, crying and whispering as sounds (they are outside the speech classes).

# Segment-level harmony, rhythm, beat sync, and talk emotion

Per-utterance absolute speech emotion failed because pitch and loudness vary more across speakers,
microphones, and Vietnamese lexical tones than across emotions. Moving from isolated 2 s clips to
**segment-level changes relative to each speaker's own modal baseline** works without adding any new
model weights:

| sense | method | bench (`bench/sense/harmony_rhythm.py`, `bench/sense/segment_emotion.py`) | result | cost per audio min |
|---|---|---|---|---|
| Harmony mode & shift | 12-bin constant-Q row-normalized chromagram (`C2..C6`, spectral whitening) + bass (`C2..F4`) + Krumhansl-Kessler profile + CUSUM two-segment chroma distance (`audio.analyze_harmony_rhythm`) | 18 progressions (`major`, `minor`, `drone`) + 10 major<->minor shifts | **18/18 (100%)** mode, **10/10 (100%)** shift within +/- 1.5 s | < 0.05 s CPU |
| Rhythm BPM & beat sync | Dual-band (full + bass/kick) onset-flux autocorrelation (`64..180 BPM`, 3/2 dotted-subharmonic guard, >= 68% active-beat coverage) + cut-to-beat alignment (`audio.cut_beat_sync`) | 10 rhythmic beds + 8 ambient/speech-only controls + 6 cut-sync cases | **10/10 (100%)** BPM within +/- 4%, **0/8** false BPMs, **6/6 (100%)** cut sync | included above |
| Segment talk emotion | Per-speaker Shortest-Half (Rousseeuw modal cluster) baseline over utterance RMS, voiced spectral tilt (`1-3.5 kHz` vs `60-400 Hz`), and `siftasr` word pace (`syll/s`), gated by multi-cue convergence + CED non-verbal vocal bursts (`sound.talk_emotion_segments`) | 48 multi-segment streams across 12 held-out RAVDESS actors (`calm`, `heated`, `animated`, `subdued`) + 4 calm controls + real Vietnamese narration (`tiktok_vi.mp4`) | **79.8% macro F1** (vs **14.2%** global unnormalized prosody, **68.6%** prosody-only), **0/5** false flags on calm talk | < 0.05 s CPU |

Card rendering (`card.md` and `sensed.json`):

- Music beds >= 5 s append mode (or `minor -> major at MM:SS` when a shift occurs), `~NNN BPM`, and `cuts sync K/N` to the existing `snd [music] (...)` line.
- Non-calm talk segments emit `MM:SS.s-MM:SS.s talk A [heated (+3.8 dB, sharp tone)]` (or `animated`, `subdued`) on `card.md` and `talk_emotions` in `sensed.json`.
- `siftasr` word timestamps (`words: [(start, end, text), ...]`) are preserved in `sensed.json` across local and `--on HOST` ASR passes.

# Tried and dropped

| idea | result |
|---|---|
| `laion/larger_clap_music` | loads with all-zero logits under transformers 5 |
| SenseVoice emotion tags | constant `NEUTRAL` in this sherpa-onnx build; not debugged further |
| unnormalized single-utterance prosody / CED probes for emotion | 14.2% macro F1 on multi-segment actor streams; speaker identity and microphone gain swamp emotion |
| linear FFT bin summation in `_chroma_basis` without $1/f$ Jacobian or spectral whitening | upper octave (`1-2 kHz`) had 16x more FFT bins than bass and broadband noise flattened chroma (`top2 = 0.20`), misclassifying real songs as `B drone` |
| single-band onset autocorrelation without 3/2 subharmonic guard | syncopated hi-hats locked onto $2/3 \times \text{BPM}$ (`93.5 BPM` on `140 BPM`) and speech bursts triggered false BPMs |
| 15th-70th percentile mean for speaker baseline | when >= 50% of a clip is heated/animated, the 70th percentile absorbs the heated utterances (+3 dB bias) and triggers false `heated` flags on calm control segments; Shortest-Half (45% modal cluster) eliminated all false flags |
| YouTube fetch of AudioSet clips | bot check after ~160 clips; used the HF parquet mirror (label column first, then the rows) |

# Open

- An "epic / tense / uplifting" judge on real film scores, and a speech-emotion judge on real
  (not acted, not synthetic) talk in Vietnamese and English. Without one, a new model cannot earn
  a card line.
- Licences if EffNet ever ships: Essentia models are non-commercial; emotion2vec+ is the FunASR
  model licence; the ONNX export used here is third-party (re-export from the pinned original).

[^audioset-eval]: `research/261017-audio-mood/as_hits.json`, `score_music.py`; AudioSet eval is not in CED's training set.
[^emotion2vec]: emotion2vec README: the seed is fine-tuned on EmoBox academic data, the base model on pseudo-labels from it.
[^emonet]: `research/261017-audio-mood/score_emonet.py`; probes trained on all of RAVDESS, tested on EmoNet-Voice.
