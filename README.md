# framesift

A video and sound sensor for agents. Claude Code and similar harnesses read text and still images;
they cannot hear and cannot take video. `framesift see` watches and listens for them, cheaply and
locally, and writes down what happened over time.

```
framesift see bug-repro.mov          # or a YouTube URL; no flags; cached by file content
```

```
## Timeline   (verbatim from the card of bench/sense/clips/ui_flow.mp4)
00:01.0-00:19.8 snd [click x38] at 1.0, 1.3, 1.7, 2.1, 2.4, 2.8, 3.2, 3.5, 3.9, 4.3, 4.6, 6.0, ... (+26) s (very high-pitched (~5528 Hz), short, sharp)
00:01.0-00:04.7 pic #1..#11   tiny white area at top-left changed 10 times, at 1.0, 1.4, 1.7, 2.1, 2.5, 2.9, 3.2, 3.6, 4.3, 4.7 s | each with a click [ev_01.jpg row 1]
00:09.6-00:11.8 pic #15       small pink area at bottom changed; it lasted 2.9 s (until 00:12.5) | text appeared: "Error 503: upstream timeout" | with sound: click [ev_05.jpg row 1]
00:11.8-00:11.9 snd [tone x2] at 11.8, 11.9 s (very low-pitched (~113 Hz), ringing)
00:11.9         pic #16       medium pink area at bottom changed; it lasted 0.7 s (until 00:12.5) | with sound: tone [ev_05.jpg row 2]
00:12.5         pic #17       tiny white area at top changed; it lasted 4.7 s (until 00:17.2); this undid #16 | text vanished: "sam@acme. io" [ev_06.jpg row 1]
```

## What it senses

| sense | how | cost |
|---|---|---|
| **picture events** | every frame (15 fps on short files) as a 32x18 grid of colour blocks; space-time regions of change become *changed*, *flashed and changed back*, *moved (path)*, *kept changing*, *cut*; a change that is undone later says how long it lasted | ~50x realtime |
| **sound** | [CED](https://github.com/RicherMans/CED) (AudioSet tagger, 22 MB, CPU, 3 s windows: 36/40 real recordings named right) names lasting sounds (music, siren, telephone); short sounds are found by a jump-and-fall-back level test, counted, timed and described (pitch, sharp or ringing); silences; bursts inside a sound | ~150x realtime |
| **sound with picture** | a sound within 150 ms of a picture change is written on it ("together with a beep: #3, #6 of the 8") | free |
| **speech** | parakeet-mlx / whisper, only when the tagger hears speech; site subtitles for URLs; `--on HOST` runs it (and the voice grouping) on a GPU box over ssh | 20-30x realtime on an M1 Pro; 58x via an RTX 3060 |
| **voices** | TitaNet-small speaker embeddings group lines into voice A, B, ... | 1.00 on real two- and three-speaker dialogues |
| **on-screen text** | Apple Vision OCR on the changed regions only; captions that repeat the speech are dropped | ~0.1 s a region |
| **frames to look at** | flow sheets at about one frame per second (12 per image, every shot's first frame; screen recordings skip still frames), before/after crops of events, and `framesift look VIDEO 11-14` for any span with a sound strip under it | images only when opened |

## Does it work? The sense bench

Questions with known answers, answered by Sonnet sub-agents who only get one arm's material,
graded blind (`bench/sense/`). **B** is the shape other "Claude watches video" tools use: uniform
1 fps contact sheets plus a transcript.

| | B: uniform 1 fps + transcript | framesift v0.2 storyboard | **framesift see** |
|---|---|---|---|
| synthetic: brief events, counts, sync, non-speech sound, screen text (26) | 22/26 | 14/26 | **26/26** (7 images) |
| real footage: order, motion, counts, duration, speech + picture (32) | 32/32 | - | 31/32 |

Synthetic clips come from a script, so the answer is exact (`make_synth.py`). Real questions were
written by a separate agent from 2 fps oracle sheets (`real.py`); no agent can hear, so real
footage tests sight and speech only. On real footage `see` costs about 6% more tokens a session
than B (92k vs 86k) and adds the senses B does not have.
Details and every rejected variant: [.okf/decisions/see-sensor.md](.okf/decisions/see-sensor.md).

| video | length | `see` wall time (M1 Pro, first run) |
|---|---|---|
| UI screen recording (synthetic) | 0:32 | 6 s |
| Big Buck Bunny | 3:00 | 22 s |
| Python in 100 s (screen montage) | 2:23 | 19 s |
| CS50 lecture | 6:00 | 56 s |

## Kind to the machine

Measured on a 16 GB M1 Pro (`bench/lagprobe.py`): the lag people feel came from memory, not CPU.
MLX kept a 5.5 GB buffer cache after transcribing, on a Mac already in swap; capping it costs no
speed and cut swap churn per run from 3.9 GB to 125 MB. framesift runs at `nice 10` with capped
worker pools and does picture and sound one after the other. macOS's background QoS band was
tried and rejected: 5x slower for no measured gain. Speech is the heaviest part on long files;
`see --on gpu-box` (or `FRAMESIFT_ON=...`) sends the audio as FLAC to `framesift asr -` on a
GPU box and falls back to local when the box does not answer in 3 s.

Every `see` and `look` call is logged locally to `~/.cache/framesift/usage.jsonl`: which spans agents
had to zoom into shows where the card falls short, the data for the next round of tuning.

## Use

```bash
uv tool install -e .                      # once; Apple Silicon gets the MLX speech backends
framesift see VIDEO_OR_URL                # card on stdout; files in ~/.cache/framesift/see/<key>/
framesift look VIDEO 11.5-13 [-n 8]       # frames over a span + a sound strip
framesift fix VIDEO 21.0 "cough"          # a sound name was wrong; 3 agreeing fixes teach later cards
framesift feedback                        # recorded fixes and the names they taught
mise run test                             # unit + sense tests
uv run python bench/sense/make_synth.py bench/sense/clips   # regenerate the synthetic clips
```

Point your agent at [`skills/framesift/SKILL.md`](skills/framesift/SKILL.md).

## Known limits

- Region words ("small red area at bottom") come from pixel statistics; the card says to check an
  image before relying on them. On camera footage the card lists only cuts and text; motion is in
  the flow sheets.
- Sound names are AudioSet classes: on 40 real ESC-50 recordings 36 were named right (90%); coughs,
  one mouse click and one alarm clock were not. Doubtful names are hedged.
- Voice letters (A, B) come from a speaker-embedding model (TitaNet-small, 40 MB): right on every
  real LibriSpeech dialogue in the bench; they are groupings, not names.
- OCR needs macOS (Apple Vision). `--on HOST` needs framesift installed on the host.
- Long files (over 20 minutes): speech is summarised per minute in the card, full text in
  `transcript.txt`; on-device speech recognition dominates the time (~4.5 min per hour of audio).

## v0.2: segments, keyframes, search (still available)

`framesift analyze VIDEO -o OUT [--asr] [--index]`, `find`, `look OUT --at T`, `frames`, `split`
work as before. Their design and measurements:

### The idea, expanded

#### 1. Three independent sources of evidence

| source | what is measured | cost |
|---|---|---|
| **Picture** | every frame at 128×72: colour histogram, 4×4-block brightness and edge density | ~50× realtime |
| **Sound** | 16 kHz log-mel timbre, loudness, pauses | ~500× realtime |
| **Codec** | packet sizes and I-frame positions, read from the container **without decoding** | ~free |

The codec source was the surprise. The encoder already did motion estimation on
every frame. **Bits spent on a P-frame measure how much changed**, and an
**I-frame off the regular GOP cadence** means the encoder decided the next frame
could not be predicted: a scene cut, detected by x264's `scenecut` for free.
framesift reads both from `ffprobe` packets and uses them as independent evidence
(`codec_scenecut`, `activity`). `--fast` decodes *only* I-frames for a quick map of
long files.

#### 2. Two levels of "important"

- **Segments** answer *where does the story change?* They are what you cut, chapter, or cite.
- **Keyframes** answer *what must I look at?* They are the I/P idea applied directly.

A segment boundary is fused from:

- **hard visual cuts**: frame-to-frame distance that is both above a floor and an
  outlier against the local 5 s baseline (so a shaky shot does not cut every frame),
  with a flash-frame guard;
- **gradual change**: novelty, the distance between the mean of the previous 1.5 s
  and the next 1.5 s, which catches fades, dissolves and slides building up;
- **acoustic change**: the same novelty on log-mel timbre (speaker, music, room,
  silence→sound);
- **pause snapping**: soft boundaries move into the nearest silence so they never split a word;
- **min/max duration**: a montage of rapid cuts becomes one segment with many
  keyframes, and a 10-minute static stretch is split at its best pauses.

Every boundary records *why* it exists (`visual_cut`, `audio_change`, `at_pause`, …),
so an agent can weigh it.

#### 3. Keyframes: drift from the last *kept* frame

Keyframes use drift from the last kept frame, not from the previous frame. A slow
pan or a slide that gains one bullet at a time never crosses a frame-to-frame
threshold, but it does accumulate. When drift crosses the threshold, framesift waits
for the most stable, sharpest frame in the next 0.6 s (no mid-transition blur) and
keeps it. Two details from testing on real footage:

- **Localized change.** A new chat message covers 10% of the screen. A whole-frame
  mean dilutes that to noise, so framesift also measures the *fraction of blocks
  that changed*, after a ±1-block motion search (a tiny codec-style search, so
  camera shake does not count as new content).
- **Screen vs camera** is auto-detected. Screen content repeats itself exactly
  between changes, while a camera never does (measured: screens 0.45–0.85
  identical consecutive frames, camera 0.00–0.06). Screens get a lower threshold
  and full weight on localized change. Camera footage gets a higher threshold, a 1 s
  minimum gap, and half weight on localized change, because there it is mostly motion.

After selection, frames that match an earlier keyframe (a cut back to the host)
are **referenced, not repeated**, and a `--max-frames` budget drops the smallest
drifts first, then the anchors of the least eventful segments.

### Measured on real footage

| video | length | segments | keyframes | vs 1 fps | analysis |
|---|---|---|---|---|---|
| Big Buck Bunny (animation, music) | 3:00 | 30 | 56 | 180 | 4.6 s |
| Tears of Steel (live action, dialogue) | 4:00 | 43 | 80 (93 before budget) | 240 | 4.2 s |
| CS50 lecture (stage, slides) | 6:00 | 40 | 66 | 360 | 6.4 s (+54 s ASR) |
| Fireship "Python in 100 s" (montage) | 2:23 | 20 | 80 (84) | 143 | 3.2 s |
| ChatGPT screen recording | 3:05 | 7 | 23 | 185 | 4.9 s |

M-series Mac, CPU only. The hard-cut floor was tuned by sweeping it against the
codec's own scene cuts as an independent reference.

### Search: pull pictures per question

With llama.cpp's `llama-server` (pinned in `mise.toml`), `framesift index OUT` embeds the keyframes,
a 1 fps probe and the speech with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
(local, Apache 2.0, the `ggml-org/embeddinggemma-2-GGUF` Q8_0 build) into one vector space. An agent then reads the storyboard
text and asks for pictures only when it needs them:

```bash
framesift find OUT "chart ranking programming languages"   # 6 distinct moments as one grid + the speech there
framesift find OUT --like 84.5                               # moments that look like 84.5 s
framesift look OUT --at 80                                   # one moment at full width, to read small text
framesift index OUT --on gpu-box                             # embed on another machine over ssh
```

A frame scores by its similarity to the query plus half the similarity of the
speech heard with it. Each command starts its own `llama-server` and stops it
when done: `find` loads the text model only (about 2 s per call, 0.5 GB), `index`
also loads the vision projector (about 3 s). Nothing keeps running. The first
start downloads the model (~0.9 GB) into the Hugging Face cache. `index --on HOST`
runs the server on HOST over ssh (HOST needs `llama-server` on its PATH and the
model in its cache); Metal and CUDA vectors agree to cosine 0.9997, so an index
built on a GPU box is searched from a laptop. Measured on the
four sample videos ([decision](.okf/decisions/embedding-search.md), [bench](bench/qa/README.md)):

| | result |
|---|---|
| retrieval, 46 visible-answer questions | a correct moment in the top 3: 87% (random frames: 19%) |
| agent QA, 55 questions, 4 rounds | 96-100% correct, same as reading every contact sheet (96%) |
| one question per session | same accuracy, 3x fewer image tokens (0.8k vs 2.3k), but the same total session tokens (71.6k vs 71.7k): `find`'s text output offsets the images. Short videos gain nothing; the target is long videos, where the storyboard alone is ~60k tokens |
| indexing a 6-minute video (396 frames) | 59 s on an M1 Pro (7.5 frames/s), 28 s with `--on` an RTX 3060 (19 frames/s) |

