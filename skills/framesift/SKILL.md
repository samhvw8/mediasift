---
name: framesift
description: Sense a video or audio file (or a web video URL) you cannot watch or hear. One command turns it into a card - a timeline of what changed on screen, which sounds happened and how many times, speech with times, and text that appeared - plus frame sheets to look at and a zoom command for any span. Use when the task involves a video, screen recording or audio file: a bug-repro recording, a UI walkthrough, a tutorial, a lecture, a meeting or podcast, a YouTube link, "what happens in this video", "when does X happen", "what did the error say", "how many times", "what was on screen when they said ...".
---

# framesift

You cannot watch video or hear audio. framesift does both for you, cheaply, and writes down what it
sensed so you can reason over time instead of guessing from a few stills.

## 1. Sense it (one command)

```bash
framesift see VIDEO_OR_URL          # prints the card; first run 10 s - 1 min per few minutes of video, then cached
```

No flags needed. It reads every frame (15 fps on short videos), tags the soundtrack, counts short
sounds, transcribes speech when it hears some, reads text that appears on screen (macOS), and caches
everything by file content, so asking again is instant. `--on HOST` sends speech recognition to a GPU
box over ssh (falls back to local if the box does not answer).

## 2. Read the card first

The card's timeline interleaves three kinds of line, all timestamped:

| line | means |
|---|---|
| `pic` | the picture changed: where (`small red area at bottom`), how (`changed`, `flashed ... and changed back`, `moved: left (...)`, `kept changing in place`), how long it stayed (`it lasted 0.7 s`), how many times (`changed 10 times, at ...`), text that appeared (OCR), and a sound that landed with it. `CUTS (n) at ...` lists new shots. |
| `snd` | a sound: `[music]`, `[siren]`, `[telephone tone, 2 separate bursts]`, `[beep x5] at 1.5, 2.9 ... s (mid-pitched (~1000 Hz), short, sharp)`, `[silence 2.9 s]` |
| `say` | speech with times; `[voice A]`/`[voice B]` group lines by speaker (not identity); with one speaker the tag is left out and said once below |

Many questions are answered by the card alone: what an error said, how many times something
happened, which flash came with a beep, what sound followed the intro, what changed after a click.
Counts and times of flashes and short sounds are measured; trust them over your eyes on a sheet.
Region words (colour, position, size) come from pixel statistics: open an image before relying on them.

## 3. Look only where you need to

- **Flow sheets** (`flow_NN.jpg`, listed in the card with the time each covers): 12 frames per image,
  about one per second where the picture moves and one per still stretch, every shot included. Open
  the sheet for the time you need, not all of them.
- **Event sheets** (`ev_NN.jpg`, referenced in `pic` lines): `[where | before | after]` crops of one
  event, for small text and UI detail.
- **Zoom**: for a gesture, a face or small text at one moment, do not guess from a flow tile (about one
  small frame a second): `framesift look VIDEO 11.5-13` gives 8 frames over that span plus a strip of the sound
  under them (loudness, short sounds in orange, tags, speech). `-n 2` for a before/after pair,
  `-n 1` for one moment at full width.

Budget: a question usually needs the card plus one to three images. If two `look` calls do not show
it, answer from what you have and say what you could not confirm.

## 4. Correct a wrong sound name

Sound names come from a tagger and are sometimes wrong (a cough heard as "throat clearing"). When
the user or a frame you looked at shows what a sound really was, record it:

    framesift fix VIDEO 21.0 "cough"

Only when you know, never as a guess. Repeated agreeing fixes teach later cards
("throat clearing (usually: cough)"); `framesift feedback` lists them.

## Long videos

For talks over a few minutes the card summarises speech one line per minute and points to
`transcript.txt` (every line with times): grep it for words, then `look` at the times you find.
Picture lines are capped; quieter stretches appear as `+N smaller changes` with a time range to zoom.

## Other commands (unchanged from earlier versions)

`framesift frames VIDEO --start A --end B --fps 2 -o DIR` (raw frames), `framesift analyze` (segments
and keyframes, the old storyboard), `index` / `find` (search frames by meaning, needs llama.cpp),
`split` (cut clips).

Cite moments by time, e.g. `@ 00:11.9`.
