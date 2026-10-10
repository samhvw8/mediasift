You are the curator of a creative council of 3 members. They have generated ideas and built on each other's ideas across several rounds. Your job is to turn all of it into a pool of distinct concepts, without judging which are best: independent judges score the pool after you.

=== THE BRIEF ===
# Brief: how could framesift extend its use of EmbeddingGemma 2?

## The question
What could framesift do with EmbeddingGemma 2 beyond the uses already proposed (below)?
The user's words: "we can extend usage of this embedding gemma 2 model". The end goal they
described: find the best frames, build image grids, send them back to the agent harness to
analyse, and from that analyse the video.

## What framesift is (read these; absolute paths)
- README.md — the idea: "analyse dense and cheap, emit sparse and rich".
- .okf/architecture/pipeline.md — modules and data flow.
- .okf/decisions/ — why the heuristics are what they are (keyframe-drift.md, screen-vs-camera.md, codec-as-evidence.md).
- src/framesift/ — code (visual.py, audio.py, segment.py, keyframes.py, render.py, meaning.py).
- research/261001-models/en-embeddings.md — earlier embedding-model research.
- skills/framesift/SKILL.md — how an agent uses it.

framesift is a Python CLI for coding agents (Claude Code and similar). It turns a video into
segments (where the story changes), keyframes (what to look at), `storyboard.md`, labelled
contact sheets (12 frames per JPEG), `manifest.json` and `signals.npz`. Every frame is analysed
at 128x72 with cheap pixel statistics (~50x realtime); audio as 16 kHz log-mel (~500x realtime);
the codec's packet sizes and I-frames are free evidence. Speech comes from faster-whisper /
parakeet. An optional `meaning` step sends keyframes to a VLM over an OpenAI-compatible API to
describe frames and summarize chapters; chapters use TextTiling on MiniLM sentence embeddings.
A bench exists (bench/evaluate.py, bench/prep_qa.py) with 4 sample videos (animation, lecture,
100-second explainer, live-action short film). The consumer is an agent harness that reads
markdown and images; it has a vision model but every image costs tokens.

## EmbeddingGemma 2 (released 2026-10-06, google/embeddinggemma-2)
- 740M parameters, Apache 2.0, built on Gemma 4. Modular: 270M text core, optional 170M vision
  and 300M audio encoders.
- One shared embedding space for text, code, images, video and audio.
- Matryoshka (MRL): 768 dims, truncate to 512/256/128.
- 8K-token context: up to 5.5 min audio, 29 images, 58 video frames, or interleaved mixes, in one embedding.
- Runs in sentence-transformers, transformers, MLX, llama.cpp (GGUF), Ollama, LiteRT. ~567 MB RAM quantized for full multimodal on a phone.
- Unknown: throughput on an Apple Silicon Mac, Vietnamese quality, how well it keeps layout/instance detail (earlier research: semantic embedders like SigLIP2 score ~20 on instance retrieval vs DINOv2 ~51 — they collapse meaning, not layout).

## Constraints the user stated (their global preferences)
- Judge a design by what it costs after it ships: infra cost ($/mo, metered APIs, per-turn
  tokens) and operational burden weigh highest, then maintenance (every knob, copy, sync step),
  then build effort (lowest — never a reason to skip).
- Surface added to save a cost not actually paid is a maintenance increase.
- Runs locally (macOS Apple Silicon; a home GPU server exists). No new hosted services wanted by default.
- Crashing beats corrupting; never fail silently.

## Already proposed in this session (build past these; do not just repeat them)
1. Embedding-distance veto on keyframes for camera footage (gestures keep meaning; drop them), not for screen recordings.
2. `--max-frames` budget chosen by coverage of distinct meanings (greedy k-center) instead of dropping smallest drifts.
3. `framesift find <out> "query"`: embed the text, rank stored frame vectors, write a grid of the top-k with timestamps for the harness.
4. Replace MiniLM in chapter segmentation with the EmbeddingGemma 2 text tower.
5. Store vectors in signals.npz; use numpy brute force.

## Rejected
- turbovec / a quantized vector index for a single video: a few thousand vectors fit in ~3 MB; numpy is exact and sub-millisecond. May revisit only for search across a library of many videos.

## A great outcome
Concrete ways to use this one model that make the agent understand a video better per token
spent, or that remove a running cost or a heuristic knob — each with a cheap way to prove it on
the bench.
=== END OF THE BRIEF ===

Read every one of these files in full (some may say NO OUTPUT; skip those):
- .council/20261007-210124/diverge/m01.md
- .council/20261007-210124/cross1/m01.md
- .council/20261007-210124/cross2/m01.md
- .council/20261007-210124/diverge/m02.md
- .council/20261007-210124/cross1/m02.md
- .council/20261007-210124/cross2/m02.md
- .council/20261007-210124/diverge/m03.md
- .council/20261007-210124/cross1/m03.md
- .council/20261007-210124/cross2/m03.md

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

Write this JSON, and nothing else, to .council/20261007-210124/pool.json:
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
