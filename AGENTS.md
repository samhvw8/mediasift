# mediasift: notes for coding agents

mediasift is a CLI that senses video and audio for agents that cannot watch or hear.
`mediasift see VIDEO` writes a timeline card (picture events, sounds, speech, on-screen text) plus
frame sheets. This file is for agents that **change** mediasift. To **use** it on a video, read
[`skills/mediasift/SKILL.md`](skills/mediasift/SKILL.md) instead.

## Setup

```bash
mise install            # Python 3.14, uv, pinned llama.cpp (for index/find)
uv sync                 # project environment in .venv
mise run test           # unit + sense tests, about 1 minute
uv tool install -e .    # optional: put this checkout on PATH as `mediasift`
```

You need `ffmpeg` and `ffprobe` on PATH. `mise run samples` fetches the test videos into `samples/`
(gitignored, needs `yt-dlp`). Two tests skip without them.

## Where things are

| path | what |
|---|---|
| `src/mediasift/see.py` | `see`: the card. Runs the picture pass, the sound pass and speech, then writes `card.md` |
| `src/mediasift/events.py`, `visual.py` | picture events from a 32x18 grid of colour blocks |
| `src/mediasift/sound.py` | CED-small sound tags (Core ML or sherpa-onnx), short-sound counting, voices |
| `src/mediasift/asr.py` | speech backends behind one `transcribe()`; `pick()` chooses per machine and language; `remote()` for `--on HOST` |
| `src/mediasift/coreml.py` | Core ML through pyobjc: compile, load, predict inside an autorelease pool |
| `src/mediasift/ocr.py` | Apple Vision OCR on macOS, falling back to Indra OCR (ONNX Runtime CPU) on Linux |
| `src/mediasift/decode.py` | every ffmpeg / ffprobe call |
| `src/mediasift/polite.py` | `nice 10` and capped worker pools |
| `src/mediasift/cli.py` | argument parsing for every subcommand |
| `pipeline.py`, `segment.py`, `keyframes.py`, `index.py`, `query.py`, `embed.py` | the v0.2 `analyze` / `index` / `find` path |
| `skills/mediasift/` | the agent skill; `.claude-plugin/` and `hooks/` make the repo a Claude Code plugin |
| `bench/` | benchmarks: questions, answers and grades are kept, media is regenerated |
| `scripts/ced_coreml.py` | builds the Core ML sound tagger (`mise run ced-coreml`) |
| `.okf/` | design decisions, measurements and every rejected idea |

## Rules

- **Read `.okf/` before you change a heuristic.** Start at [`.okf/index.md`](.okf/index.md). Most
  thresholds were tuned against a bench, and the rejected alternatives are listed there. Do not
  try them again without new evidence.
- **Update `.okf/` in the same change as the code.** Record the new numbers and add a line to
  `.okf/log.md`. The bundle follows the Open Knowledge Format: every concept file needs YAML
  frontmatter with a `type`.
- **Prove a change with a number.** A speed change needs a timed run before and after. A change to
  what the card says needs the sense bench (`.okf/benchmarks/sense-bench.md` explains how to run it).
  A bug gets a failing test first.
- **Crash rather than corrupt.** A missing model or a failed backend raises or logs one clear line
  on stderr. Never drop speech or sounds silently.
- **Pin every model.** Downloads use a fixed revision or SHA (see `QUAN_REV` in `asr.py`,
  `CED_REV` in `sound.py`, `INDRA_REV` in `ocr.py`). Models go into `~/.cache/mediasift/models/` (`MEDIASIFT_CACHE`).
- **Keep platform guards.** MLX, Core ML and Apple Vision exist only on macOS (Apple Silicon for MLX
  and Core ML). Linux falls back to faster-whisper, the CPU tagger and Indra OCR. Tests for one platform
  skip on the other.
- **Mind memory.** MLX speech peaks at about 3.3 GB, so it waits for the picture pass. Core ML calls
  go through `coreml.predict`, which drains an autorelease pool. Without it, a 47-minute file grew to
  36 GB.
- **Keep the card short.** The card is read by an agent that pays per token. Add a line only when a
  bench question needs it.
- **No personal data in the repo.** Write paths relative to the repo root. Read hosts from `--on` or
  `MEDIASIFT_ON`, never from the code.
