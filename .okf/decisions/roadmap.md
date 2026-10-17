---
type: Plan
title: Roadmap - what to build next, from the 2026-10-17 council and its stress tests
description: The ordered next steps for mediasift after the music-mood work, each with the measured evidence that put it there, the stress-test changes it must carry, and its status. Start here before picking new work.
tags: [roadmap, plan, council, integrity, captions, library]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-17T16:00:00Z"
sources:
  - id: council
    resource: .council/20261017-140250
    title: 8-lens council (fact-check desk, dubbing engineer, film music supervisor, audio describer, agent operator, evidence skeptic, user advocate, pre-mortem); develop/ and stress/ hold the four shortlisted concepts
  - id: usage
    resource: process:usage-log-review-2026-10-17
    title: Review of the local usage log and session prompts - of 737 cached cards about 7 were real use (phone reels, a YouTube clip, a screen video, long talks)
---

# Why this order

Real jobs, from session prompts since 2026-09-17: transcribe, translate and summarise talks **with
timestamps and links across videos**; fact-check a TikTok; "watch and listen" to reels; re-voice into
Vietnamese; turn a reel into a credited script; a bug-repro screen recording.[^usage] The council found that
the largest gains are failures that already happen, not new senses.[^council] Order follows the owner's cost
rules: running cost and breakage first, build effort last.

| # | step | evidence | stress-test changes to carry | status |
|---|---|---|---|---|
| 1 | URL fetch: a caption failure never costs the video; captions in the video's own language | 5 of 7 real fetches died on HTTP 429 for auto-translated captions; an auto-dubbed English talk got German captions | done without the "sensing ladder": the 429 was a request bug, and a card built from post text could raise trust in unsensed text | **done** ([URL fetch](/decisions/see-sensor.md)) |
| 2 | Vietnamese speech on QUAN even when the GPU box is up | box faster-whisper wrote "Hãy subscribe cho kênh ..." over 20 s of real speech and looped "bắt tự" ~40 times on a real reel; QUAN heard the same span | measure on the real reels first (both backends) | next |
| 3 | Card header rendered at read time; `CARD_VERSION` out of the cache key; `cache prune` | 733 of 734 cached cards print a dead `framesift look`; 2.49 GB of cards can never be served again | derive any per-track revision from source and pin hashes, not hand-bumped constants; no `--refresh` | open |
| 4 | Speech integrity, cheap half: clamp line ends to CED speech runs, collapse loops, alarm on known invented phrases | Tears of Steel lines stretched to 15.5 s; loops and stock phrases on 2 real VN cards | before any mark goes on the card, test whether agents obey it (`~x~` reads as strikethrough); sweep parakeet chunking before building re-hearing | open |
| 5 | Burned-in captions: offline bench first | 3 of 4 real reels carry captions that `_drop_captions` deletes | CapCut / TikTok captions may be ASR too (not independent); URL inputs may already use platform subtitles; band search only on short or vertical video; tone-mark-only differences stay out | open |
| 6 | The cache as a citable library (`grep`, `quote`, `align`) | a real clip (`tDmjz6HB-yw`) cut from a 47 min talk drops the interviewer's question | measure against `rg` over existing transcripts; count how often multi-video jobs recur; index lazily, never inside `see`; voxtract's `ID@start-end` handles | open, gated |
| 7 | Word-timestamp deixis zoom (`mediasift look VIDEO --say "phrase"`, council C05) | deixis bench questions ("when he says X, what is his hand doing?") inside multi-clause lines (`reel_en` `q02`, `q04`, `q06`; `tiktok_vi`) | keep `siftasr` word timestamps (`speech[i].words`, SentencePiece subwords merged); match exact or accent-stripped Vietnamese or fuzzy 1-word ASR diffs; interpolate cue span when `words` is absent; list closest 3 lines on miss; highlight matched words in yellow on the sound strip (capped at 8 word labels) | **done** |
| 8 | Adaptive & preset sensing modes (`--mode {auto,adaptive,probe,podcast,music,reel,screen,full}`, `--with` / `--without`) and incremental `sensed.json` cache upgrades | `altman_vi.mp4` spent 88% of wall time (`61.4s / 69.9s`) and 45% of card lines on picture/OCR/image work on a talking-head interview; `cs50_lecture.mp4` spent 42% of wall time on unused visual/OCR passes | keep `--mode auto` (`full`) as default so existing bench contracts stay 100% compatible; `probe` runs S2T + 32x18 grid + sound tags + 1 keyframe sheet and reports `detected: <kind>`; `adaptive` classifies from the lightweight signal and runs only relevant heavy passes; incremental `--with` upgrades run only missing passes on cached `sensed.json` | **done** |

Lower in the pool, kept for later: a self-checking dub (card of the dub vs the source; the judges split
most on it), a "futz" tag for phone or radio voices, a claims ledger (which witnesses back each number),
one voice roster across a video.

[^usage]: The usage log in the cache, the cached cards' `source.txt`, and the owner's prompts that named a video file or link.
[^council]: `harvest` in the run folder lists the develop and stress file per concept.
