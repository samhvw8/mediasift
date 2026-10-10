---
type: Decision
title: Embedding search with find and look
description: EmbeddingGemma 2 puts frames and speech in one space so an agent pulls a few moments per question instead of reading every contact sheet; ranking, sizes and the llama.cpp engine are set by bench measurements.
tags: [search, embeddings, agents]
status: draft
generated:
  by: claude-code/opus-5.5
  at: "2026-10-08T16:00:00Z"
resource: src/framesift/query.py
sources:
  - id: eg2-card
    resource: https://huggingface.co/google/embeddinggemma-2
    title: EmbeddingGemma 2 model card by Google DeepMind (MRL table, prompts, precision, encoder loading)
    last_modified: 2026-10-06
  - id: eg2-blog
    resource: https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
    title: EmbeddingGemma 2 announcement (Google blog)
    last_modified: 2026-10-06
  - id: qa-bench
    resource: bench/qa/README.md
    title: Agent QA bench (writer / answerer / grader sub-agents)
  - id: search-eval
    resource: bench/search_eval.py
    title: LLM-free retrieval check against question evidence spans
---

# Decision

`framesift index OUT` embeds keyframes, a 1 fps probe (at most 600) and ~8 s
speech windows with EmbeddingGemma 2 (text + vision towers, Apache 2.0) into
`vectors.npz`. `framesift find OUT "<what is visible>"` returns the top 6
distinct moments as one grid plus the speech there; `look --at T,...` returns
exact moments. The agent reads the storyboard text first and pulls pictures
on demand instead of reading every sheet.

| setting | value | why |
|---|---|---|
| model | `google/embeddinggemma-2`, no audio tower (440M) | one space for text and images; Apache 2.0, unlike EmbeddingGemma 1[^eg2-blog] |
| vision budget | 70 soft tokens per image | M1 Pro MPS: 280 → 1.1 img/s, 140 → 2.8, 70 → 5.9; pairwise-similarity correlation 70 vs 280 = 0.955 |
| precision | float32 on MPS/CPU, bfloat16 on CUDA | float16 overflows in this model[^eg2-card]; bfloat16 was slower than float32 on MPS (1130 vs 873 ms/img) |
| dimension | 768 stored and used | 256 lost hit@1 on the bench (0.59 vs 0.65 visual-only) and 3 MB per video saves nothing; vendor table: 128 d loses 13 points on MMEB v2[^eg2-card] |
| prompts | `SearchQuery` for queries, `Document` for speech, none for images | model card[^eg2-card] |
| score | cos(frame, q) + 0.5 · max cos(speech within 1 s, q) | hit@3 0.80 → 0.87 over 46 visual questions, never worse on a video; weights 0.5 and 1.0 tie at hit@3, 0.25 lower, 1.0 loses hit@6[^search-eval] |
| distinct moments | skip results < 2 s apart or cosine > 0.95 to one already chosen | k moments, not one moment k times |
| `look` tiles | one moment at full width; several at 384 px, 3 per row | full width for 2-4 moments let agents double their image tokens with no accuracy gain (rounds 2-3) |

# The embedding engine: llama.cpp, one server per command

The first version ran PyTorch in a warm local server, because a cold load took
8-11 s (5 s of it importing torch). On 2026-10-08 it moved to llama.cpp
(`llama-server`, `ggml-org/embeddinggemma-2-GGUF` Q8_0, release b11501+), which
removed torch (~2 GB) and the daemon:

| | measured |
|---|---|
| `find` (text model only, `--no-mmproj`) | ready in 1.7-2.1 s, 2.0-2.6 s per call end to end, 0.46 GB RSS |
| `find --like` | 0.23 s: compares stored frame vectors, starts no model |
| `index` (text + vision projector) | ready in 2-4 s; 7.1-7.9 frames/s on an M1 Pro (Metal) |
| `index --on gpu-box` (RTX 3060, over ssh) | ready in 3.2 s; 19 frames/s; cs50 (396 frames) in 28 s vs 59 s locally |
| retrieval, 46 visual questions, llama.cpp vectors | hit@1/3/6 0.72 / 0.87 / 0.91 (PyTorch: 0.74 / 0.87 / 0.91) |
| llama.cpp vs PyTorch, same input | speech/queries cosine ≥ 0.9997; frames median 0.96-0.99 (image resize differs, not quantization) |
| Metal vs CUDA index of the same video | frames median 0.9997 (min 0.9964), speech ≥ 0.9996 |

How it runs (`embed.py`): llama.cpp is pinned in framesift's own `mise.toml`, not installed
globally, so framesift asks `mise ls --installed` for that install and runs it from any folder
(`FRAMESIFT_LLAMA_SERVER` overrides). Each command starts `llama-server` on a private Unix
socket under a small `sh` wrapper that waits on its stdin. When framesift is
done, or dies (tested with `kill -9`), stdin closes and the wrapper removes the
socket and kills the server; when the server fails to start, the wrapper exits
too and framesift prints the log tail at once instead of waiting. Over ssh the
same wrapper runs on the host and the socket is forwarded (`ssh -L`), so no
server is left on either machine. The wrapper uses no signal traps because the
box's login shell (zsh) skips EXIT traps when killed. Starts try `--offline`
first (no network) and download only when the cache misses; the failed
offline attempt leaves its socket behind, so the wrapper removes it before
retrying.

Vectors record model, GGUF, dimension, image budget, engine build and host.
`find` checks only model and dimension: queries are text, and text vectors
agree across engines, so an index built by PyTorch or on another machine stays
searchable. Frame vectors of one index all come from one `index` run, so they
never mix engines. Details: `research/261008-model-ratio/engine-test.md`
(including a zml benchmark, dropped: ~4x slower prefill on the 3060).

# Evidence

Retrieval, no LLM: 46 visible-answer questions, hit = a returned moment
inside the evidence span (± 1 s)[^search-eval].

| ranking | hit@1 | hit@3 | hit@6 |
|---|---|---|---|
| shipped (frame + speech) | 0.74 | 0.87 | 0.91 |
| frame only | 0.65 | 0.80 | 0.89 |
| random distinct frames | 0.07 | 0.19 | 0.34 |

Agent QA: 4 videos x 14 questions written from dense oracle frames, answered
by separate sub-agents per arm, graded against the key[^qa-bench]; 55 valid
questions after two key corrections.

| arm | correct | image tokens |
|---|---|---|
| A transcript only | 49% | 0 |
| B uniform sheets | 93% | 25.5k |
| C framesift sheets (2 rounds) | 96%, 96% | 37.0k, 28.4k |
| F find/look (4 rounds) | 98%, 98%, 96%, 100% | 24.7k, 49.8k, 43.8k, 37.7k |

With 14 questions in one session, C reads its sheets once and spreads the
cost; F pays per question, so the two cost about the same. With **one
question per session** (7 sampled visual questions), both were 7/7 and F used
0.8k image tokens per question against 2.3k for C.

**Total session tokens** (from the sub-agent usage reports) erase that gain on
short videos: one-question sessions averaged 71.6k (F) vs 71.7k (C), and
14-question sessions 85-87k (F) vs 79.5k (C), because `find` prints the
speech at every hit and each call adds tool overhead. On 2-6 minute videos
search costs the same as reading the sheets. Its case rests on long videos,
where the storyboard alone is ~60k tokens.

**Long videos** (1h45m English podcast, 47 min Vietnamese dub; one-question
sessions, 4 per video): every arm was 8/8, and total session tokens were
storyboard 78.7k, transcript 81.5k, `find` 88.0k. Agents grep the storyboard
instead of reading it, so a long text file does not cost what its size
suggests, and `find`'s verbose output makes it the dearest arm. Details:
`bench/qa_long/README.md`.

# Consequences

- On short and long videos alike, search buys accuracy parity and moments no
  keyframe kept, not lower cost. Grep over the storyboard is the bar to beat;
  `find` should print less (hits without the full speech) before it can. For a full summary, the sheets remain the cheaper read.
- Search has no ceiling. One answerer hunting a slide that did not exist read
  29 images (26k tokens); the skill now says to stop after two `find` and one
  `look` and report what could not be confirmed.
- The z-score of the top hit does not tell a real match from an off-topic
  query (medians 3.7 vs 2.9, heavy overlap), so no "weak match" warning ships.
- Not measured: Vietnamese speech, screen recordings beyond one montage,
  videos longer than 6 minutes, and the keyframe veto proposed in
  [keyframe drift](/decisions/keyframe-drift.md).

[^eg2-card]: EmbeddingGemma 2 model card, Hugging Face.
[^eg2-blog]: Google blog, EmbeddingGemma 2 announcement.
[^qa-bench]: bench/qa/README.md and bench/qa_report.py.
[^search-eval]: bench/search_eval.py over bench/qa/*/key.json.
