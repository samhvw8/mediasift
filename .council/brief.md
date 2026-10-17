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
