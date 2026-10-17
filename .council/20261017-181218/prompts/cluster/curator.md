You are the curator of a creative council of 8 members. They have generated ideas and built on each other's ideas across several rounds. Your job is to turn all of it into a pool of distinct concepts, without judging which are best: independent judges score the pool after you.

=== THE BRIEF ===
# What could an agent that cannot watch or hear still not perceive in a video, and how might mediasift give it that sense?

## What mediasift is today

`mediasift` is a local CLI. `mediasift see VIDEO|URL` writes a short timeline **card** (markdown) plus
~1 fps frame sheets for an AI agent (Claude Code) that reads text and looks at still images, but cannot
watch video or hear audio. The agent reads the card first, looks at frame sheets, and can zoom with
`mediasift look VIDEO A-B` (frames over a span plus a drawing of the sound).

Senses the card has now: picture change events from a 32x18 grid of colour blocks (changes, blinks,
motion, cuts with shot letters for repeated places, camera pans/tilts/scrolls, drawn-in overlays with
lifetimes); on-screen text by Apple Vision OCR on changed regions; sounds as AudioSet classes from the
CED-small tagger on 3 s windows (lasting sounds, counted short sounds with pitch words, hedged names,
silences, bursts, music mood and genre words); sound-picture sync (a sound within 150 ms of a picture
change); speaker letters with gender from voice embeddings; speech with times (local ASR or over ssh on
a GPU box, or the site's subtitles). Everything the card says has been measured on a bench.

Read these (repo-relative to `~/workspace/tools/mediasift/`) before you start:
- `AGENTS.md` - rules for changing it
- `skills/mediasift/SKILL.md` - how agents use it
- `.okf/index.md` and `.okf/decisions/` - every decision, measurement and rejected idea
- `.okf/decisions/roadmap.md` - work already planned (do not re-propose it)
- `~/.cache/mediasift/see/3b5fd29ca2c9be19/card.md` - a real card (Tears of Steel, 4 min)

## Evidence about where the card is not enough

- Real use (session logs since 2026-09-17): summarise and link talks with timestamps; fact-check a
  TikTok; "listen to and watch" Vietnamese and English reels; re-voice YouTube into Vietnamese keeping
  the background; turn a reel into a credited script; a bug-repro screen recording.
- When the card is not enough, the agent runs `look`. 61 recorded zooms: median span **1.4 s**
  (brief moments); 20 of them on one fast-cut code screencast, where dense code and small text change
  quickly.
- The real-footage bench (6 videos, 48 questions) asks only picture-timing questions: sequence,
  motion, deixis ("what is on screen when X is said"), count, duration. Current score 32/32 on 4
  landscape videos, 15/16 on phone videos. The one shared miss: timing a gesture to a spoken phrase.
  Questions about what the card cannot sense are absent from the bench, so it cannot see those gaps.
- Already measured and rejected: per-line speech emotion labels (45% right on held-out speech),
  open-vocabulary music words from CLAP (below the tagger's own classes), cheap pitch/loudness
  emotion probes (near chance across datasets). Do not re-propose these unless you change what is
  measured.
- The audio is decoded to 16 kHz mono before any sensing.

## Hard constraints (from the owner)

- Lightweight: must not take much CPU/GPU on heavy runs. Apple M1 Pro 16 GB (MLX speech peaks at
  3.3 GB) and an RTX 3060 12 GB Ubuntu box over ssh that is sometimes booted into Windows.
- Judge designs by what they cost after they ship: running cost ($/month, quotas, per-turn tokens) and
  things that break or need watching weigh most; build effort weighs least.
- The card is read by an agent that pays per token: a line earns its place only when a measured
  question needs it. Every sense needs a judge (a bench with answers) before it reaches the card.
- Crash rather than corrupt; never drop speech or sound silently; pin every model; local-first, no paid
  cloud API in the default path. Vietnamese and English both matter.

## What a great outcome looks like

New senses, or new ways to deliver one, that let the agent answer questions about a video it cannot
answer today, each cheap to run and each with a way to prove it works.
=== END OF THE BRIEF ===

Read every one of these files in full (some may say NO OUTPUT; skip those):
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m01.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m01.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m01.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m02.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m02.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m02.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m03.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m03.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m03.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m04.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m04.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m04.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m05.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m05.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m05.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m06.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m06.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m06.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m07.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m07.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m07.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/diverge/m08.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross1/m08.md
- /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/cross2/m08.md

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

Write this JSON, and nothing else, to /Users/samhv/workspace/tools/mediasift/.council/20261017-181218/pool.json:
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
