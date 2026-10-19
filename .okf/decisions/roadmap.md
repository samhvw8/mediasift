---
type: Plan
title: Roadmap - what to build next, from the 2026-10-17 council and its stress tests
description: The ordered next steps for mediasift after the music-mood work, each with the measured evidence that put it there, the stress-test changes it must carry, and its status. Start here before picking new work.
tags: [roadmap, plan, council, integrity, captions, library]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-18T16:15:00Z"
sources:
  - id: council
    resource: .council/20261017-140250
    title: 8-lens council (fact-check desk, dubbing engineer, film music supervisor, audio describer, agent operator, evidence skeptic, user advocate, pre-mortem); develop/ and stress/ hold the four shortlisted concepts
  - id: usage
    resource: process:usage-log-review-2026-10-17
    title: Review of the local usage log and session prompts - of 737 cached cards about 7 were real use (phone reels, a YouTube clip, a screen video, long talks)
  - id: see14-artifact
    resource: https://mediasift-see-14.art.3cxo.work/
    title: Interactive see-14 Before & After visual proof and 18-clip Hugging Face / MotionBench report (local copy at bench/sense/real/r20/see_14_before_after.html)
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
| 9 | Codec-aligned motion-compensated P-frame residuals, stratified intra-bucket peak snapping (`flow_sheets` & `look`), `look` active-region Codec Canvas 2x zoom row (`_active_crop_row`), and compressed-domain `decode.packet_activity` (`see-14`)[^see14-artifact] | `reel_en` `q06` (`0.25 s` raised fist at `9.28 s`) fell between uniform clock midpoints in r19; stationary camera with central actor motion triggered false camera tilts while fast pans wrapped `0.5 s` FFT phase (`1/3` recall on `zai-org/MotionBench`) | outer 25% border-strip consensus shift `(dx, dy)` warps `prev` when median peak >= `0.16` (**3/3** MotionBench pan recall, **0/3** still FP, **-40.2%** pan background noise); `[0.18, 0.82]` interior peak window + `0.6 s` cut guard (**1.79x** mean peak activity gain across 18 HF clips); 4-tile 2x `ACT CROP` row in `look` (**16/16** on `real/r20`) | **done** ([picture sensing](/decisions/picture-sensing.md)) |
| 10 | Active-viewport layout classification (`events.detect_layout`), matte-trimmed border shift (`events._border_shift`), and persistent full-frame banner/headline OCR (`see.scan_persistent_text`, `see-15`)[^see14-artifact] | `_ocr_events` only inspects `before`/`after` crops of changed regions, missing persistent headlines present from `t = 0.0 s` on **9 of 10** real TikTok, YouTube Shorts, and Instagram Reels clips (`10%` recall); static letterbox/pillarbox mattes pinned 2 of 5 border strips to `0.0` | `detect_layout` checks both low temporal delta (`< 0.008`) and spatial bar uniformity (`std < 0.035`) + 1-column seam sharpness for `split-vertical` (**11/11** across 30 clips); `scan_persistent_text` samples up to 6 settled full frames, normalizes Vietnamese `đ`/`Đ` + NFD accents, drops speech-redundant captions, and surfaces `Lasting text` by `top`/`middle`/`bottom` zone (**10/10** recall) | **done** ([picture sensing](/decisions/picture-sensing.md)) |

Lower in the pool, kept for later: a self-checking dub (card of the dub vs the source; the judges split
most on it), a "futz" tag for phone or radio voices, a claims ledger (which witnesses back each number),
one voice roster across a video.

[^usage]: The usage log in the cache, the cached cards' `source.txt`, and the owner's prompts that named a video file or link.
[^council]: `harvest` in the run folder lists the develop and stress file per concept.
[^see14-artifact]: Published at `https://mediasift-see-14.art.3cxo.work/` and committed locally at `bench/sense/real/r20/see_14_before_after.html`.
