# Engine test: llama.cpp vs PyTorch for EmbeddingGemma 2 (2026-10-08)

Setup: llama.cpp commit 08246a2 (includes PR #30054, EG2 text+vision+audio), built from source on the
Mac (Metal, against MacOSX26.5.sdk because the CLT linker rejects the 27.0 SDK) and on the home box (CUDA,
SM86, in `~/llama.cpp-eg2`, separate from the running Qwen build). Model `ggml-org/embeddinggemma-2-GGUF`
Q8_0 + mmproj Q8_0 (also BF16 for one control run). `llama-server --embeddings --image-max-tokens 70
--image-min-tokens 70`; image request shape `{"input":[{"content":[{"type":"image_url",...}]}]}`;
query / document prefixes added by hand. Script: `bench/engine_eval.py` (removed with the PyTorch backend on 2026-10-08; it compared a llama-server against the stored PyTorch vectors).

| check | PyTorch (MPS, current) | llama.cpp Metal (M1 Pro) | llama.cpp CUDA (RTX 3060) |
|---|---|---|---|
| cosine to PyTorch: queries / speech | — | 0.9999 / 0.9998 (min) | 0.9998 / 0.9990 (min) |
| cosine to PyTorch: frames | — | median 0.972, min 0.890 | median 0.972, min 0.890 |
| cosine Metal ↔ CUDA | — | frames median 0.9998 (min 0.9987), queries 0.9999 | same |
| hit@1 / @3 / @6, 46 visual questions | 0.74 / 0.87 / 0.91 | 0.72 / 0.87 / 0.91 | 0.70 / 0.87 / 0.91 |
| frames/s | 6-8 | 8.7 | 14.9 (through a ZeroTier SSH tunnel) |
| cold start | 8-11 s (5 s torch import) + 3 s first-call warm-up | 2.7 s warm cache (8.1 s first ever) | 3.2 s |
| text-only start + 1 query | — | 1.25 s, 0.5 GB RSS | — |
| VRAM | — | — | 3.1 GB at -ub 8192 |

Findings:
- Text is identical across engines. Image vectors differ by preprocessing (resize), not quantization: BF16
  gave the same 0.972; budget sweep 70/140/280 never lines up exactly (best 280↔280 at 0.988).
- Retrieval quality is unchanged (hit@3 identical; hit@1 differs by 1-2 of 46 questions).
- llama.cpp on Metal and on CUDA agree with each other to 0.9998, so an index built on the box can be
  searched from the Mac. A text query from either engine can search an index from either engine; only frame
  vectors must all come from one engine.
- Decision input: llama.cpp removes torch (~2 GB), cuts cold start 3-4x, and lets `find` start a text-only
  server per call (1.25 s) instead of keeping a warm daemon.

## zml (Zig + OpenXLA), 2026-10-08

zml has no EmbeddingGemma 2 port (it ships Llama 3.1, Qwen 3.5, LFM 2.5), so the test ran the same model on both
engines: Qwen3.5-0.8B BF16 (safetensors for zml, unsloth BF16 GGUF for llama.cpp), zml commit 728eedf with a local
timing patch (prefill timed 3x; zml always prefills the full `--seqlen`, so seqlen = prompt size).

| RTX 3060 | llama.cpp | zml (cuda_fa2) |
|---|---|---|
| prefill 512 tokens | 59 ms (8,614 tok/s, llama-bench) | 230 ms warm, 250 ms first |
| prefill 2048 tokens | 226 ms (9,077 tok/s) | 830 ms warm, 890 ms first |
| decode | 143-149 tok/s | 62-64 tok/s |
| one-shot run, start to exit | 3.0-4.0 s | 22-29 s (XLA compile 11-15 s every run, load 1.6-2.4 s) |
| peak RSS | 1.7 GB | 3.8 GB |
| build | cmake, minutes | Bazel, 15m47s on the box (XLA/LLVM from source), 29 GB cache |

zml skipped FlashInfer CUTLASS MoE on the 3060 (`UnsupportedArchitecture`); irrelevant for a dense model.
The Metal build was stopped at 7,219/7,251 actions when the decision was made, so there is no Mac number.
Verdict: llama.cpp is ~4x faster at prefill, ~2.3x at decode, starts ~6x sooner, and runs EG2 today. zml is dropped.
