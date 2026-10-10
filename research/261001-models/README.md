# Model alternatives for framesift: synthesis (2026-10-01)

Six gatherers: EN vision models (`en-vlm.md`), ZH vision models (`zh-vlm.md`), ZH-TW vision models (`zh-tw-vlm.md`, thin), EN video models (`en-video-models.md`), EN embeddings (`en-embeddings.md`), VI speech recognition (`vi-asr.md`). Hardware targets: RTX 3060 12 GB (vLLM) and Apple Silicon (MLX). Every number below is either published by the model's authors or comes from one practitioner. None has been measured on framesift's own frames or audio yet.

## Recommendation per role

| Role | Keep / change | Model | Why | Confidence |
|---|---|---|---|---|
| Keyframe description | **Keep** | `RedHatAI/Qwen3.5-4B-FP8-dynamic` (3060), `mlx-community/Qwen3.5-4B-MLX-4bit` (Mac) | Ranked first of 23 local vision models in an independent test on real images with a task close to framesift's (75.5, ahead of Qwen3.5-9B 73.1 and Qwen3-VL-8B 74.4; Gemma 4 E4B 58.8) ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1ubx4rw/best_local_model_for_vision_2nd_benchmark_update/)). No newer small Qwen exists: Qwen3.6 and Qwen3.8 start at 27B (HF API). | medium |
| Fallback describer | Test | `Qwen/Qwen3-VL-4B-Instruct-FP8` | No thinking mode. Plain attention, so it avoids the hybrid-cache vLLM bugs. Two Chinese hands-on reports prefer it for recognition, while vendor benchmarks favour 3.5 (OCRBench 80.8 vs 85.0). | low |
| Text-heavy frames | Add, optional | `Qwen/Qwen3.5-9B` 4-bit (`QuantTrio/Qwen3.5-9B-AWQ`) or an OCR specialist (`zai-org/GLM-OCR`, `PaddlePaddle/PaddleOCR-VL-1.6`) | 9B OCRBench 89.2 vs 85.0. Chinese practitioners favour a separate OCR model for verbatim text, because general vision models "correct" the source. Use it only on frames with a lot of text. | medium |
| Clips for motion | Add | same Qwen3.5-4B, video input | Qwen3.5-4B is video-native (self-reported Video-MME 76.9), and a 10 s clip at 2 fps / 640×360 costs about 2.2k tokens, the same as 10 keyframes. Use clips only on high-motion segments. | medium |
| "Find when X happens" | Add later, on demand | `MCG-NJU/TimeLens2-4B` | Charades-TL mIoU 57.7, against Qwen3-VL-4B 49.4 and Gemini 2.5 Pro 52.8 ([paper](https://arxiv.org/html/2607.17423)). Run it only on the top-k segments found through transcript and description search. | medium |
| Speech, English | **Keep** | faster-whisper large-v3-turbo, batched (CUDA); parakeet-mlx (Mac) | Measured: 55–64× realtime on the 3060. | high (measured) |
| Speech, Vietnamese | **Test** | `khanhld/chunkformer-rnnt-large-vie`, `nvidia/parakeet-ctc-0.6b-Vietnamese` (3060); Zipformer-30M via sherpa-onnx (Mac) | All beat whisper-large-v3 on the authors' own Vietnamese tables. ChunkFormer handles long audio natively. Parakeet-vi adds punctuation, casing and word timestamps. No head-to-head against turbo, and no podcast audio, in any source. | low until measured |
| Text embeddings (chapters, search) | **Change** | `Alibaba-NLP/gte-multilingual-base` (Vietnamese-heavy) or `intfloat/multilingual-e5-small` (CPU) | The current MiniLM scores MMTEB 48.8; e5-small 55.5 at the same size; gte-multilingual VN-MTEB 65.22 ([VN-MTEB](https://arxiv.org/html/2507.21500v1)). Vietnamese-specific fine-tunes did not beat their parents. | medium |
| Image embeddings (semantic dedupe) | Add | `google/siglip2-base-patch16-224`, plus DINOv2-small as a second check | SigLIP2 places frames with the same meaning close together and can also search by text. DINOv2 is strong where SigLIP2 is weak, on "same instance" (Oxford-Hard 51 vs 20). Drop a frame only when both agree. | medium |

## Cheap wins before any model swap

1. `enable_thinking=false`, already set in framesift. Every source says thinking hurts Qwen3.5 small models: it lowers scores, causes timeouts, and once thought for 40 s on "Hi".
2. Upgrade vLLM 0.21 → about 0.29 (the latest), which fixes Qwen3.5 multimodal prefix-cache issues ([#43587](https://github.com/vllm-project/vllm/issues/43587)).
3. Cut image tokens: throughput is prefill-bound (OvisOCR2 vLLM report). Keep 768 px or smaller, with dimensions divisible by 32 for Qwen.
4. `--mm-processor-cache-gb 0` for unique frames. Disable prefix caching if requests hang under concurrent load ([#37729](https://github.com/vllm-project/vllm/issues/37729)).
5. Swap in a better text-embedding model. It's a one-line change and helps chapters immediately.

## Where sources disagree

- **Qwen3.5-4B vs Qwen3-VL-4B.** Vendor benchmarks and the r/LocalLLaMA test favour 3.5. Two Chinese anecdotes (bottled drinks, ID-card names) favour Qwen3-VL. Only an A/B on our own frames settles it.
- **Gemma 4.** Most tests put it behind Qwen on OCR and description, with invented details. One thread says it wins at bounding boxes and on European content, and its default vision budget of 280 tokens handicaps it. If tested, raise that budget to 560–1120.
- **9B vs 4B.** 9B is better on documents and OCR. In the r/LocalLLaMA test it was not better at plain description, apparently because it overthinks.
- **Why the first FP8 run broke.** The EN agent assumed the RedHat config was the broken one. In fact the broken run was the original `Qwen/Qwen3.5-4B` with vLLM's on-the-fly `--quantization fp8`. The RedHat build, which keeps the vision tower in bf16, passed the self-test. Rule: use a checkpoint whose ignore list keeps `model.visual.*` in bf16. Never quantize a vision model on the fly.

## Licence notes

Fine for any use: Qwen3.5 and Qwen3-VL (Apache-2.0), SigLIP2, DINOv2, e5, gte-multilingual, bge-m3, TimeLens2, ChunkFormer RNNT (CC-BY-4.0).
Restricted: MobileCLIP2 (research only), jina v3 and clip-v2 (non-commercial), DINOv3 (custom), EmbeddingGemma (Gemma terms), Zipformer-30M (CC-BY-NC-ND), Parakeet-vi (NVIDIA Open Model License, commercial OK, gated).

## Next measurements (Ubuntu box)

1. Vision model A/B on 200 framesift keyframes, with the same image budget and thinking off: Qwen3.5-4B FP8 (baseline) vs Qwen3-VL-4B FP8 vs Qwen3.5-9B AWQ. Measure frames/s and question-answering accuracy.
2. Vietnamese speech: a 20-minute hand-checked sample from the anydub podcasts, scored with one normaliser, comparing turbo vs ChunkFormer vs Parakeet-vi vs PhoWhisper-large.
3. Text embeddings: chapter quality with gte-multilingual vs e5-small vs MiniLM on two transcripts (one EN, one VI).
4. Semantic dedupe: calibrate the SigLIP2 and DINOv2 thresholds on the six "rocket igniting" frames and the lecture gesture frames.

## Coverage gaps

Taiwanese forums had nothing measured. Vietnamese forums had nothing measured. Several Zhihu bake-offs returned 403 (listed in `zh-vlm.md`). The MTEB leaderboard returned 429. No source measured any candidate on an RTX 3060 in frames per second, so that has to be our own benchmark.
