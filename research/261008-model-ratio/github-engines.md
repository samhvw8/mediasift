# github-engines — inference engines and runtimes (gh search, 2026-10-08)

Question: alternatives to framesift's current stack (PyTorch + sentence-transformers for EmbeddingGemma 2, mlx-whisper / parakeet-mlx for ASR) for running LLM / VLM / embedding / tensor models on an M1 Pro and an RTX 3060.

Name check: no inference engine called `llmz` exists. `zllm` = two tiny projects (jiacai2050/zllm, Zig + Metal, macOS only, 6 stars; zllm-lab/zllm, 10 stars). The serious Zig engine is **zml/zml** (4.1k stars, Zig + OpenXLA/MLIR + Bazel, Apache-2.0).

## Engines (stars, last push, license)
| group | repo | stars | push | license | fit for framesift |
|---|---|---|---|---|---|
| C/C++ core | ggml-org/llama.cpp | 130.6k | 10-08 | MIT | **EmbeddingGemma 2 GGUF ships `mmproj` (vision)** → text + image embeddings without torch; Metal + CUDA; mmap load |
| | ggml-org/whisper.cpp | 54.2k | 10-06 | MIT | ASR, same ggml base |
| | ikawrakow/ik_llama.cpp | 3.3k | 10-07 | MIT | faster quants fork |
| Apple | ml-explore/mlx (28.7k), mlx-lm (7.2k), Blaizzy/mlx-vlm (5.6k), mlx-audio (8.0k), mlx-embeddings (449, last push 05-13) | | | MIT | already used for ASR; EG2 MLX ports in flight (mlx-swift-lm PR #684 "full support", Obscyra EmbeddingGemma2Swift 8-bit) |
| | trymirai/uzu | 1.8k | 10-08 | MIT | Apple-silicon engine |
| | argmaxinc/argmax-oss-swift (WhisperKit) | 6.4k | 10-08 | MIT | Swift ASR |
| Google | google-ai-edge/LiteRT-LM | 6.6k | 10-08 | Apache-2.0 | named by Google for EG2 on-device |
| Servers (GPU) | vllm 93.4k, sglang 36.9k (EG2 PR #42782 open), TensorRT-LLM 14.8k, lmdeploy 8.1k, exllamav3 1.6k | | | | 3060 serving; heavy for our use |
| Rust | huggingface/candle 21.1k, tracel-ai/burn 16.0k, EricLBuehler/mistral.rs 7.7k, sonos/tract 3.1k | | | Apache/MIT | no-Python runtimes |
| Portable | microsoft/onnxruntime 22.0k (+ genai 1.1k), openvino 11.0k, pytorch/executorch 5.1k, apple/coremltools 5.4k, tinygrad 33.7k, mlc-llm 23.2k | | | | ONNX/Core ML export paths |
| Speech / OCR | k2-fsa/sherpa-onnx 15.2k (Zipformer/Gipformer, hotwords), RapidAI/RapidOCR 8.1k (PP-OCRv6 ONNX) | | | Apache-2.0 | candidate OCR/ASR runtimes |
| Embedding servers | text-embeddings-inference 5.1k, fastembed 3.2k, infinity 3.0k (last push 03-24) | | | | text-only focus |
| New / small | ashhart/TensorFold 1.1k (Metal/CUDA/Vulkan), QuixiAI/embeddinggemma.c 27 (C11 + Metal server, EmbeddingGemma 1 text only) | | | | watch |

## What matters for framesift
- The warm embedding server exists because torch + transformers take ~5 s to import and ~9 s to load. **llama.cpp with `ggml-org/embeddinggemma-2-GGUF` (Q8_0 + mmproj)** could embed both text and images with a fast mmap load. That might remove both torch and the daemon, which the architecture plan (2026-10-02) asked for ("no long-running model servers").
- Unverified: whether llama.cpp's image-embedding path for EG2 gives the same vectors as the PyTorch reference (cosine ≈ 1), its images/s on M1 Metal, and its cold-start time. Vectors from two runtimes must not be mixed in one index (`check_identity` would need the runtime in the identity).
- EG2 support landing elsewhere (all opened 10-06..10-08): mlx-swift-lm #684, sglang #42782, CoreML-LLM #172, ailia-models #1947, MTEB results PRs #745 / #747 (independent evaluation).
