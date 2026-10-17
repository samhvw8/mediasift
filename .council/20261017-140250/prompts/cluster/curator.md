You are the curator of a creative council of 8 members. They have generated ideas and built on each other's ideas across several rounds. Your job is to turn all of it into a pool of distinct concepts, without judging which are best: independent judges score the pool after you.

=== THE BRIEF ===
# How might we extend mediasift so an agent that cannot watch or hear gets what it needs from media?

## What mediasift is today

`mediasift` (renamed from framesift on 2026-10-17) is a local CLI. `mediasift see VIDEO|URL` writes a
short timeline **card** (markdown) plus ~1 fps frame sheets, for an AI agent (Claude Code) that reads
text and still images but cannot watch video or hear audio. The card holds: picture events (changes,
cuts with shot letters, camera moves, on-screen text by OCR), sounds (AudioSet classes from the CED-small
tagger, counted short sounds, hedged names), speaker letters, speech with times (local ASR on the Mac,
or over ssh on a GPU box), and since today music mood and genre words. `look VIDEO A-B` zooms into a
span, `fix` logs an agent's correction of a sound name, an older `index` / `find` does embedding
search over frames and speech. A Claude Code skill and a prompt hook tell agents to run it.

Read these (absolute paths) before you start:
- AGENTS.md - layout and rules for changing it
- skills/mediasift/SKILL.md - how agents use it
- .okf/index.md and decisions/ - every design decision, measurement and rejected idea
- .okf/decisions/music-and-voice-mood.md - today's mood work
- ~/.cache/mediasift/see/3b5fd29ca2c9be19/card.md - a real card (Tears of Steel, 4 min film clip)

## Who uses it, and for what (from real logs, not guesses)

One person (the owner) and the agents working for them. Real use since 2026-09-17, from session logs:
- YouTube talks: transcript, translate to Vietnamese, summarise **with timestamps**, then **link several
  videos together** (asked many times; a sibling tool `~/workspace/tools/anydub` does download + dub).
- A TikTok: "check the content from this, then web search to verify the information" (fact-check).
- TikToks / Facebook reels: "listen to and watch this" (short vertical social video, Vietnamese and English).
- Dubbing YouTube into Vietnamese where "the English is too quiet ... and the background sound is lost".
- A Facebook reel turned into a shared script with credit to its author.
- A screen recording of an app crash for a bug report.
- Of 737 cached cards, ~730 are bench runs; real ones are 26 s - 2:44 phone videos, a 15 s YouTube
  clip, a 2:40 screen video, and long talks (47 min) and podcasts (1h45).

## What was measured today

- Music mood from CED's own AudioSet classes: right 88% when stated (score >= 0.10, 35% of clips),
  genre 83% (>= 0.15), on 742 held-out AudioSet clips. Free: the tagger already runs.
- Open-vocabulary music words ("epic", "heroic") from CLAP or an MTG-Jamendo head scored lower than
  CED's plain words. No test set exists for "epic" on real film scores.
- Speech emotion labels: emotion2vec+ 88% on RAVDESS (in its training lineage) but 45% on newer held-out
  speech, still ~58% right when it says it is 95% sure; SenseVoice said NEUTRAL for everything; cheap
  pitch/loudness probes were near chance across datasets. **Per-line emotion labels were rejected.**
- The card already names laughter, shouting, crying, screaming and whispering as sounds.

## Ideas already on the table (build on them or avoid them)

From the owner's assistant, not yet built: (1) put vocal sounds on the speech line they overlap
(`say [laughing] "..."`); (2) speaker-relative prosody ("louder and higher than A's usual"); (3) let
`fix` take mood corrections so real use grows a test set; (4) a music build/peak finder; (5) an
on-demand audio LLM on the GPU box for one span. The owner also floated a bigger name and scope
("media-understanding"); the rename to mediasift is done.

## Hard constraints (from the owner)

- **Lightweight**: must not take much CPU/GPU on heavy runs. Hardware: Apple M1 Pro 16 GB (MLX speech
  peaks at 3.3 GB already) and an RTX 3060 12 GB Ubuntu box reachable by ssh (sometimes booted into Windows).
- **Judge designs by what they cost after they ship**: running cost ($/month, quotas, per-turn tokens)
  and things that break or need watching weigh most; build effort weighs least.
- The card is read by an agent that pays per token: add a line only when a measured question needs it.
- Crash rather than corrupt: never drop speech or sound silently. Pin every model.
- Every change is proven with a number on a bench (synthetic clips with scripted answers, real-footage
  questions, a 200-clip sound bench, a voice bench). A feature without a judge cannot earn a card line.
- Vietnamese and English both matter. Local-first; no paid cloud API in the default path.

## What a great outcome looks like

Extensions that make the owner's real jobs (verify, summarise and link with citations, re-voice,
understand short social video, bug repro) clearly better, cheap to run, each with a way to prove it works.
=== END OF THE BRIEF ===

Read every one of these files in full (some may say NO OUTPUT; skip those):
- .council/20261017-140250/diverge/m01.md
- .council/20261017-140250/cross1/m01.md
- .council/20261017-140250/cross2/m01.md
- .council/20261017-140250/diverge/m02.md
- .council/20261017-140250/cross1/m02.md
- .council/20261017-140250/cross2/m02.md
- .council/20261017-140250/diverge/m03.md
- .council/20261017-140250/cross1/m03.md
- .council/20261017-140250/cross2/m03.md
- .council/20261017-140250/diverge/m04.md
- .council/20261017-140250/cross1/m04.md
- .council/20261017-140250/cross2/m04.md
- .council/20261017-140250/diverge/m05.md
- .council/20261017-140250/cross1/m05.md
- .council/20261017-140250/cross2/m05.md
- .council/20261017-140250/diverge/m06.md
- .council/20261017-140250/cross1/m06.md
- .council/20261017-140250/cross2/m06.md
- .council/20261017-140250/diverge/m07.md
- .council/20261017-140250/cross1/m07.md
- .council/20261017-140250/cross2/m07.md
- .council/20261017-140250/diverge/m08.md
- .council/20261017-140250/cross1/m08.md
- .council/20261017-140250/cross2/m08.md

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

Write this JSON, and nothing else, to .council/20261017-140250/pool.json:
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
