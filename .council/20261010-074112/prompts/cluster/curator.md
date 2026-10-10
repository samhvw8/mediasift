You are the curator of a creative council of 5 members. They have generated ideas and built on each other's ideas across several rounds. Your job is to turn all of it into a pool of distinct concepts, without judging which are best: independent judges score the pool after you.

=== THE BRIEF ===
# Brief: rethink framesift as a daily-use video + audio "sensor" for AI agent harnesses

## The question (the user's words, lightly edited)
"LLMs don't sense video well (a sequence of images that connect) and the audio with it. So I
built this to cherry-pick images for that sequence, also the audio with it. But the current stack
is very hard to run daily or for regular use. Make a sensor for an AI harness. Or maybe my
approach is not correct — think, then help me. We can improve, or even change everything and
build it up again."

How might we give a text-and-image agent (Claude Code and similar harnesses) a real sense of
video — motion, continuity, cause and effect across frames, and the sound that goes with it —
in a form the user will actually run every day? Is "cherry-pick frames + transcript" the right
shape at all, or is there a better one?

## Who it is for
- The consumer: an agent harness (Claude Code today). It reads text and still images. It cannot
  hear audio and cannot take video input. Every image costs tokens (a 768 px frame is roughly
  1-1.5k tokens; a 12-tile contact sheet about the same). It calls CLI tools and reads files.
  Each session already carries ~63-65k tokens of fixed overhead (instructions + tools).
- The operator: one developer (Sam) on an Apple Silicon Mac (M1 Pro 16 GB). A home GPU box
  (RTX 3060 12 GB, `ssh gpu-box`) exists but is not always on. Videos: screen recordings
  (bug repros, UI walkthroughs), tutorials, lectures, podcasts (English and Vietnamese, some
  1-2 hours long), short films/explainers, YouTube links.

## What exists today: framesift v0.2 (read these; absolute paths)
- README.md — the idea and measurements.
- .okf/architecture/pipeline.md — modules and data flow.
- .okf/decisions/ — why the heuristics are what they are, with measurements (embedding-search.md has the agent QA bench results).
- skills/framesift/SKILL.md — how an agent uses it now.
- src/framesift/ — ~2.4k lines of Python (decode, visual, audio, segment, keyframes, render, asr, transcript, embed, index, query, meaning, cli).
- research/261002-architecture.md — an earlier 5-lens architecture brainstorm (time-indexed library, card/find/read/look verbs, layer store). Mostly not built.
- research/261008-model-ratio/README.md — which small models fit each slot (ASR, embeddings, OCR).
- bench/qa/README.md and bench/qa_long/README.md — the agent QA bench.

In short: `framesift analyze VIDEO -o OUT [--asr] [--index]` decodes every frame at 128x72 for
cheap pixel statistics, reads codec packet sizes, computes 16 kHz log-mel audio novelty, fuses
them into segments, picks keyframes by drift from the last kept frame, optionally transcribes
(faster-whisper / parakeet-mlx / mlx-whisper), and writes storyboard.md, 12-tile contact sheets,
frames/*.jpg, manifest.json, signals.npz. Optional: `index` embeds frames + speech with
EmbeddingGemma 2 via llama.cpp (started per command), then `find "query"` / `look --at T` return
image grids. `meaning` calls a VLM over HTTP. `frames` and `split` extract ranges/clips.

## Measured facts (from the bench and from a run today)
- A 2:23 explainer, `analyze --asr --index`, end to end: 63 s on the M1 Pro (ASR 22 s with
  faster-whisper small at 6.6x realtime; indexing 30 s at 9 frames/s). Output: 20 segments,
  80 keyframes, 7 contact sheets, a 10 KB storyboard. Needs an output folder and 2-3 flags.
- A 1h45m podcast: 8.2 min analyse+index; 207 KB storyboard.
- Agent QA, short videos (55 questions): transcript only 49% correct; uniform contact sheets 93%;
  framesift sheets 96%; find/look 96-100%. Total session tokens: equal between arms (~72k for a
  one-question session) — search saved image tokens but its text output ate the savings.
- Long videos (one-question sessions): every arm 8/8; agents GREP the storyboard rather than read
  it; storyboard 78.7k, transcript 81.5k, find 88.0k total tokens. The visual questions were
  guessable from context, so they did not separate the arms.
- Never benchmarked: questions that need temporal understanding (what moved where, what happened
  between A and B, order of UI steps, counting repetitions, cause→effect), and non-speech audio
  (music changes, laughter, alarms, tone of voice, silence). That is exactly the gap the user names.
- Vietnamese ASR: whisper large-v3-turbo best (2.37% WER on a TTS slice).
- Known output flaws seen today: transcript cues repeat across adjacent segments; storyboard
  carries stat lines ("visual 1.51 · audio 0.63") an agent cannot act on.

## Constraints the user stated (global preferences)
- Judge a design by what it costs AFTER it ships. Highest weight: infra cost ($/month, metered
  APIs, per-turn tokens) and operational burden (what breaks, needs watching). Then maintenance
  (every copy, knob, sync step). Build effort is the lowest weight and never a reason to skip.
- Local first (Mac, optional home GPU). No new hosted service or always-on daemon by default.
  A paid cloud API is allowed only if its running cost is justified against the alternative.
- Crashing beats corrupting; never fail silently.
- The user gives full freedom: improve, or rebuild from scratch, or change the approach entirely.

## Ideas already on the table (build past them or argue against them; do not just repeat)
- From the earlier brainstorm (not built): a time-indexed library with `card / find / read / look`
  verbs, a content-addressed layer store, an extractive overview card, write-back of what the
  agent saw.
- Proposed earlier, not built: OCR on screen keyframes; semantic keyframe veto for camera footage;
  an MCP server; streaming mode.

## Rejected earlier
- Long-running model servers (vLLM-style daemons): slow start, VRAM fights, things to watch.
- A vector database server for a single video.

## A great outcome
A design (possibly a rebuild) that (1) gives the agent a genuine sense of motion, sequence and
sound per token spent — measurably better on temporal and audio questions than "keyframes +
transcript"; (2) is something the user runs every day without thinking (one command or none,
no flag tuning, fast enough, nothing to babysit); and (3) a cheap way to prove each claim.
=== END OF THE BRIEF ===

Read every one of these files in full (some may say NO OUTPUT; skip those):
- .council/20261010-074112/diverge/m01.md
- .council/20261010-074112/cross1/m01.md
- .council/20261010-074112/cross2/m01.md
- .council/20261010-074112/diverge/m02.md
- .council/20261010-074112/cross1/m02.md
- .council/20261010-074112/cross2/m02.md
- .council/20261010-074112/diverge/m03.md
- .council/20261010-074112/cross1/m03.md
- .council/20261010-074112/cross2/m03.md
- .council/20261010-074112/diverge/m04.md
- .council/20261010-074112/cross1/m04.md
- .council/20261010-074112/cross2/m04.md
- .council/20261010-074112/diverge/m05.md
- .council/20261010-074112/cross1/m05.md
- .council/20261010-074112/cross2/m05.md

How to curate:
1. Group ideas and builds that are really the same concept. A concept is a distinct direction, not a feature list.
2. For each concept, write the strongest version the material supports. Prefer the most developed build over its parents, and keep what was surprising: do not sand it down into something safe.
3. Record the lineage: the ids of every idea and build the concept draws on.
4. Keep the odd ones. A strange idea that only one member had is a concept of its own. Do not merge it into something tamer to tidy the pool.
5. At most 12 concepts. If there are more, keep the most distinct ones.
6. Orphans: pick up to 3 single ideas that fit no concept and that you would be tempted to drop because they are odd, not because they are weak. Copy each one's title and text as the member wrote it, without improving it. Number them after the concepts.
7. The members' Baseline sections are context, not ideas: do not turn a baseline into a concept unless a member developed it further.
8. Do not score, rank or recommend. Do not add ideas of your own.
9. Do not create, edit or delete any file except the one below.

Write this JSON, and nothing else, to .council/20261010-074112/pool.json:
{
  "concepts": [
    {"id": "C01", "title": "short name", "pitch": "two or three sentences: what it is, for whom, and what makes it non-obvious", "parents": ["m03.2", "m07.r1.4"], "members": ["m03", "m07"]}
  ],
  "orphans": [
    {"id": "C13", "title": "the member's title", "pitch": "the member's text, as written", "parents": ["m05.6"], "members": ["m05"]}
  ]
}
Ids run C01, C02, and so on, across concepts and orphans.

When the file is written, reply with this one line and nothing else:
POOL <number of concepts>
