# en-vlm: open VLMs vs Qwen3.5-4B for keyframe description (RTX 3060 12GB / vLLM, Apple Silicon / mlx-vlm)
Date: 2026-10-01 (all HF/GH data pulled live). Lang: EN. Tags: GEM / MEH / NOISE.

## 0. TL;DR
- Landscape moved: Qwen3.5 (Feb 26), Gemma 4 (Mar 26), Qwen3.6 (27B/35B-A3B only), Qwen3.8 (27B, 125B-A6B+ only), MiniCPM-V 4.6 (1.2B), Moondream 3.1 (9B-A2B), LFM2.5-VL-3B, Granite Vision 4.1 4B, Mage-VL 4B. NO Qwen3.6/3.8 small (<=9B) VLM exists. Qwen3.5 0.8/2/4/9B is still the newest Qwen small tier.
- Nothing clearly beats Qwen3.5-4B at same size. Real upgrade path = Qwen3.5-9B (+4 OCRBench, +3 CC-OCR) but needs 4-bit on 12GB; independent bench says 9B NOT better than 4B for describe-style tasks (overthinking). Qwen3-VL-8B-Instruct is the strongest *non-thinking* alternative (independent: best OCR/mixed-script).
- Gemma 4 E4B/12B: worse than Qwen for OCR/description in every independent test found. NOISE for this use case (unless vision budget set to 1120, still behind).
- Biggest lever may be config not model: thinking OFF (enable_thinking=false), and for Gemma the vision token budget.

## 1. Comparison table (verified = from model card/API unless "est.")
| Model | Params (HF safetensors) | License | Released | OCRBench | CC-OCR | OmniDocBench1.5 | MMMU | MMBench-EN | VideoMME(w/o sub) | vLLM | mlx-vlm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B (baseline) | ~4.5B incl. ViT, hybrid GDN | Apache-2.0 | 2026-02-27 | 85.0 | 76.7 | 86.2 | 77.6 | 89.4 | 76.9 | yes (needs recent; README says nightly at launch) | yes, mlx-community/Qwen3.5-4B-MLX-4bit |
| Qwen3.5-9B | ~9-10B (est.) | Apache-2.0 | 2026-02-27 | 89.2 | 79.3 | 87.7 | 78.4 | 90.1 | 78.4 | yes | yes |
| Qwen3.5-2B | ~2B | Apache-2.0 | 2026-02-28 | 84.5/85.4 (think/non) | 72.9/75.8 | 79.8/80.9 | 64.2 | 83.3 | 69.0 | yes | yes |
| Qwen3.5-0.8B | ~0.8B | Apache-2.0 | 2026-02-28 | 74.5/79.1 | 63.2/66.7 | 61.0/70.6 | 49 | 69.9 | 57.7 | yes | yes |
| Qwen3-VL-4B-Instruct | 4.44B (verified) | Apache-2.0 | 2025-10-11 | 80.8 | 73.8 | 80.0 | 70.8 | 86.7 | 68.9 | yes, official -FP8 | yes |
| Qwen3-VL-8B-Instruct | 8.77B (verified) | Apache-2.0 | 2025-10-11 | not extracted (card has only images); indep. rated best OCR | ? | ? | ? | ? | ? | yes, official -FP8 (modules_to_not_convert=model.visual) | yes (lmstudio-community 4-8bit) |
| Qwen3-VL-2B-Instruct | ~2B | Apache-2.0 | 2025-10-19 | 79.2 | 68.3 | 65.9 | 61.4 | 81.9 | 62.1 | yes | yes |
| Gemma 4 E4B | 4.5B eff / 8B w/ PLE embeddings | Apache-2.0 | 2026-03-02 | n/a | n/a | edit-dist 0.181 (lower=better; Qwen3.5-4B 86.2 higher=better, scales differ, E4B clearly worse than 12B 0.164) | n/a (MMMU-Pro 52.6 vs Qwen3.5-4B 66.3) | n/a | n/a | yes (needs current transformers; open bugs below) | yes (Gemma4 supported; issues open) |
| Gemma 4 12B "Unified" | 11.95B | Apache-2.0 | 2026-05-23 | n/a | n/a | 0.164 | MMMU-Pro 69.1 | n/a | n/a | yes; RedHatAI FP8-Dynamic, google qat-w4a16-ct | yes (mlx-community/gemma-4-12B-it-qat-4bit) |
| Gemma 4 26B-A4B | 25.2B / 3.8B active | Apache-2.0 | 2026-03-11 | n/a | n/a | 0.149 | MMMU-Pro 73.8 | | | yes | yes; too big for 12GB |
| GLM-4.6V-Flash | 9B | MIT | 2025-12-07 | n/a in card | | | | | | yes (vllm>=0.12) | yes (lmstudio MLX 4/6/8bit) |
| MiniCPM-V 4.5 | 8.7B (verified) | Apache-2.0 | 2025-08-24 | "leading OCRBench", OpenCompass 77.0 | | SOTA among general MLLM on OmniDoc (claim) | | | | yes (>=0.10.2); -AWQ official | llama.cpp/ollama; mlx-vlm lists MiniCPM-o and V4.6 only |
| MiniCPM-V 4.6 | ~1.2B (SigLIP2-400M + Qwen3.5-0.8B) | Apache-2.0 | 2026-04-13 | ~Qwen3.5-2B level (claim) | | | | | | yes BUT official GPTQ/AWQ fail to load (#58943, vision tower built quantized) | yes (listed) |
| InternVL3.5-8B | 8.53B (verified) | Apache-2.0 | 2025-08-25 | n/a extracted | | | | | | yes | not in README list |
| LFM2.5-VL-3B | ~3B | LFM1.0 (custom, check revenue cap) | 2026-08-11 | OCRBenchv2-en 47.5 (Qwen3-VL-4B? col unlabeled) | | | MMMU-Pro 30.5 | | | yes | yes (LiquidAI official MLX 8bit) |
| Granite Vision 4.1 4B | 4.0B (verified) | Apache-2.0 | 2026-04-16 | doc/KIE focused | | | | | | native since vLLM 0.21.0 | yes (README) |
| Mage-VL (microsoft) | 4.74B (verified), Qwen3-4B + from-scratch codec ViT | Apache-2.0 | 2026-07-25 | "on par with Qwen3-VL-4B on static images" | | | | | | custom codec processor, unclear | MLX port exists (mlx-community/Mage-VL-OptiQ-4bit, load bug #2227) |
| Moondream 3.1 | 9B / 2B active MoE | custom moondream-model-license-1.0 | 2026-06-30 | | | | | | | NO (own Photon engine) | listed (moondream3) |
| Phi-4-reasoning-vision-15B | 15B | MIT? (unverified) | 2026-01-23 | | | | | | | RedHatAI quant exists; too big for 12GB bf16 | yes |
| Ministral-3-3B / 8B Instruct | 3B/8B | Apache-2.0 (unverified) | 2025-10-31 | | | | | | | | |
| Molmo2-4B / 8B (Ai2) | 4B/8B | ? | 2025-12-14 | | | | | | | | MolmoPoint listed |
| Kimi-VL-A3B | 16B total / 3B active (est.) | | 2025-04 | stale | | | | | | | |
| SmolVLM2-2.2B, Florence-2, Phi-4-MM, Ovis2.5 | stale 2025 or specialist | | | below Qwen3.5-2B | | | | | | | |
Sources: Qwen3.5 card table https://huggingface.co/Qwen/Qwen3.5-4B (4B/9B/2B/0.8B rows; Qwen3-VL-4B/2B rows come from the Qwen3.5-2B card https://huggingface.co/Qwen/Qwen3.5-2B); Gemma https://huggingface.co/google/gemma-4-12B-it and /gemma-4-E4B-it; params via https://huggingface.co/api/models/<id>; mlx-vlm list https://github.com/Blaizzy/mlx-vlm; LFM https://huggingface.co/LiquidAI/LFM2.5-VL-3B; MiniCPM https://huggingface.co/openbmb/MiniCPM-V-4.6 and /MiniCPM-V-4_5.
DocVQA/ChartQA/InfoVQA: NOT in Qwen3.5 / Gemma 4 cards (they moved to OmniDocBench/CC-OCR/CharXiv). Qwen3.5 CharXiv(RQ) 4B 70.8 / 9B 73.0; AI2D 89.6 / 90.2. LFM2.5-VL-3B ChartQA 81.3.

## 2. VRAM guide (weights only; add ~1.5-3 GB for ViT activations + KV for 8 concurrent 768px; est. = params x bytes, not measured)
| Model | bf16 | FP8 | 4-bit | Notes |
|---|---|---|---|---|
| Qwen3.5-4B | ~9 GB | ~5 GB | ~3.2 GB (GGUF Q4, measured in [Reddit bench](https://www.reddit.com/r/LocalLLaMA/comments/1ubx4rw/best_local_model_for_vision_2nd_benchmark_update/)) | hybrid GDN = small KV |
| Qwen3.5-9B | ~19 GB | ~10 GB (too tight on 12GB with ViT+KV) | ~6-7 GB: QuantTrio/Qwen3.5-9B-AWQ, cyankiwi/Qwen3.5-9B-AWQ-4bit, RedHatAI/Qwen3.5-9B-quantized.w4a16 | fits 12GB |
| Qwen3-VL-8B | ~17.5 GB | ~9.5 GB (Qwen/Qwen3-VL-8B-Instruct-FP8) tight | cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit ~6 GB; GGUF Q8 8.1GB | full attention => larger KV than Qwen3.5 |
| Qwen3-VL-4B | ~9 GB | ~5.5 GB (Qwen/Qwen3-VL-4B-Instruct-FP8) | AWQ-4bit | |
| Gemma 4 E4B | ~16 GB (8B incl. PLE embeddings) | ~9 GB | google/gemma-4-E4B-it-qat-w4a16-ct | PLE tables count toward VRAM in practice |
| Gemma 4 12B | ~24 GB | RedHatAI/gemma-4-12B-it-FP8-Dynamic ~12+ GB (does not fit) | google/gemma-4-12B-it-qat-w4a16-ct (10.3 GB on disk) | |
HF usedStorage numbers in API are inflated by multiple revisions; do not trust for size.

## 3. Findings
### F1 [GEM] Qwen3.5-9B / 4B official benchmark table
9B beats 4B: OCRBench 89.2 vs 85.0, CC-OCR 79.3 vs 76.7, OmniDoc 87.7 vs 86.2, MMMU 78.4 vs 77.6. Both beat Gemini-2.5-Flash-Lite and GPT-5-Nano on OCRBench (82.5 / 75.3). Qwen3.5-2B (84.5 OCRBench) is within 0.5 of 4B on OCRBench but -7 on OmniDoc/CharXiv. https://huggingface.co/Qwen/Qwen3.5-4B
### F2 [GEM] Independent real-image benchmark (30 images, 23 variants, llama.cpp, M2 Max): Qwen3.5-4B nothink Q4 = 75.5 (best pts/GB); Qwen3-VL-8B Q8 = 74.4; Qwen3.5-9B nothink Q4 = 73.1 (WORSE than 4B); Qwen3-VL-4B 65.9 ("degenerates", high variance); Gemma 4 12B Q4 64.1, E4B Q4 58.8/Q8 63.9 ("wrong counts", hallucinations); GLM-4.6V-Flash-9B Q4 75.1 (best Chinese OCR). Thinking ON lowers every Qwen hybrid score and causes timeouts; Qwen3.5-9B Q8 85% failures. Prompt = "describe, visible text verbatim, objects, people..." which is framesift's task. https://www.reddit.com/r/LocalLLaMA/comments/1ubx4rw/best_local_model_for_vision_2nd_benchmark_update/ . Caveat: single author, LLM judge 60% + probes 40%, llama.cpp not vLLM, Q4 GGUF not FP8.
### F3 [GEM] Earlier round (Ollama, 20 images): Qwen3-VL-8B best OCR (only one reading Chinese+Latin banner); Gemma 4 E4B invented details (6 people vs 5); InternVL3.5-8B called QR code a maze, terse; GLM-4.6V-Flash best on multi-panel. https://www.reddit.com/r/LocalLLaMA/comments/1u5oydc/which_is_the_best_local_vlm_benchmark_results/
### F4 [GEM] Gemma 4 vision token budget: default 280 cripples OCR; must raise to 560-1120 (vLLM max_soft_tokens; llama.cpp --image-min-tokens 560 --image-max-tokens 2240). Even then: HN user says Qwen3.5-9B beats Gemma 4 31B at scene description; jwr filament-box OCR test Qwen near perfect, Gemma 26B-A4B useless (before budget fix). https://news.ycombinator.com/item?id=48182721 ; https://www.reddit.com/r/LocalLLaMA/comments/1t1te8y/qwen_36_wins_the_benchmarks_but_gemma_4_wins/ (counter: this thread claims Gemma wins "reality" for bbox/European content; many commenters disagree: Qwen far better on forms/handwriting/charts).
### F5 [MEH] Gemma 4 E4B OCR anecdote: loses most lines on rotated Polish text vs Gemma 3 4B and Qwen3 4B; early llama.cpp bugs possible. https://www.reddit.com/r/LocalLLaMA/comments/1sbc8jb/gemma_4_4b_vs_gemma_3_4b_qwen_3_4b_in_ocr/
### F6 [GEM] Document-AI leaderboard (9000 docs): Qwen3.5 overall 0.8B 58.0 / 2B 63.2 / 4B 73.1 / 9B 77.0; 4B OlmOCR 77.2, KIE 86.0, VQA 72.4, table GrITS 76.7, handwriting 64.7; 9B VQA 79.5. 4B->9B gain mostly VQA, not raw OCR. https://www.reddit.com/r/LocalLLaMA/comments/1rv98wo/qwen359b_on_document_benchmarks_where_it_beats/ (leaderboard idp-leaderboard.org, not fetched).
### F7 [GEM] Vision-tower quantization handling (matches your breakage):
- Qwen official Qwen3-VL FP8: modules_to_not_convert = ["lm_head","model.visual"] (ViT stays bf16). https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8/raw/main/config.json
- RedHatAI Qwen3.5-4B/9B FP8-dynamic and w4a16, cyankiwi AWQ: compressed-tensors `ignore` lists every model.visual.blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2} (ViT NOT quantized). So your RedHatAI repo already keeps the ViT in bf16 by config; if images break, it is a vLLM loading/version issue not quantizing the tower. Check that ignore list actually matches module names in vLLM 0.21 + transformers 4.57 (Qwen3.5 landed after 4.57; README required vLLM nightly at launch).
- Gemma 4 qat/FP8 ignore embed_vision.* and vision_embedder.patch_dense only (small); cyankiwi E4B AWQ ignores all vision_tower layers.
- vLLM #58943 (open 2026-09-27): official MiniCPM-V-4.6 GPTQ/AWQ fail to load because vision tower is built quantized, reporter on RTX 3060. https://github.com/vllm-project/vllm/issues/58943
- llm-compressor #2842 "AWQ model can not deploy" (closed). https://github.com/vllm-project/llm-compressor/issues/2842
- Reddit comment: Qwen3.5 raw vLLM/transformers load gotcha, weights carry model.language_model.* prefix and leftover mrope_section in config causes silent shape errors (single comment, unverified) in thread above (u/laul_pogan).
### F8 [GEM] vLLM issues relevant to your baseline (Qwen3.5 hybrid):
- #37729 (open, updated 2026-09-26): engine deadlocks under concurrent load with fp8 + prefix caching + Qwen3.5; trigger is tile misalignment seq_len%128 (GDN prefill kernel, mostly Blackwell/flashinfer reports, one Triton/FLA GDN report). You run 8 concurrent: if you see hangs, disable prefix caching. https://github.com/vllm-project/vllm/issues/37729
- #43587 (open): prefix caching fails for incremental multimodal requests on Qwen3.5 hybrid; partly fixed on 0.29 per comment. Qwen3-VL (pure attention) unaffected. https://github.com/vllm-project/vllm/issues/43587
- #38696 (open): response_format json_schema => runaway whitespace on Qwen3.5/Gemma 4/Qwen3.8 (checkpoint dependent; official FP8 fine, third-party NVFP4 bad). Relevant: framesift asks JSON. Mitigation: use guided decoding off + prompt JSON + parse, or test your checkpoint. https://github.com/vllm-project/vllm/issues/38696
- Latest vLLM is ~0.29 (comment dated 2026); you are on 0.21. Granite Vision 4.1 needs >=0.21.
- Gemma 4 vLLM bugs open: #49717 (streaming content empty, reasoning holds output), #39468 (tool call format), #40994. https://github.com/vllm-project/vllm/issues/49717
### F9 [MEH] mlx-vlm status: very active (0.7.3, daily diagnostic sweeps of ~50 models #2374/#2368). Qwen3.5 first-class in README (thinking budget, DFlash/MTP spec decode, continuous batching in server). Gemma 4 supported but open bugs: prefill 3x slower in single-chunk path (#2155 closed), can't build without vision tower (#2386), image processor edge cases (#2346). Qwen3-VL supported; qwen3_vl_moe video broken (closed). Granite 4.0 Vision, MiniCPM-V 4.6, GLM-OCR, PaddleOCR-VL, DeepSeek-OCR, Moondream3, Phi-4 reasoning vision listed. InternVL3.5 and GLM-4.6V-Flash not in README list (GLM-4.6V-Flash has lmstudio MLX quants, so it runs somewhere). https://github.com/Blaizzy/mlx-vlm ; https://github.com/Blaizzy/mlx-vlm/issues/2374
### F10 [GEM] MLX repos that exist: mlx-community/Qwen3.5-4B-MLX-4bit (35.8k dl), mlx-community/Qwen3.5-9B-MLX-4bit, lmstudio-community/Qwen3.5-9B-MLX-4bit/8bit (600k/560k dl), mlx-community/Qwen3.5-4B-OptiQ-4bit, lmstudio-community/Qwen3-VL-4B/8B-Instruct-MLX-{4,5,6,8}bit (100k dl each), mlx-community/gemma-4-12B-it-qat-4bit, lmstudio-community/gemma-4-E4B-it-MLX-4bit, LiquidAI/LFM2.5-VL-3B-MLX-8bit.
### F11 [MEH] Throughput: NO 3060 or vLLM VLM numbers found for any candidate. Only data: M2 Max Ollama Q4 tok/s Qwen3-VL-4B 61, 8B 43, Gemma E4B 41, GLM-9B 37, Gemma 12B 21 (F3 link); llama.cpp median s/img: Qwen3.5-4B 20s, Qwen3-VL-8B Q8 26s (long verbose outputs, F2 link). LFM2.5-VL-3B claims ~11K tok/s on H100 vLLM at high concurrency, highest of tested (card). Must benchmark yourself.
### F12 [MEH] MiniCPM-V 4.5 (8.7B) older but strong OCR claims, 96x video token compression, official AWQ/int4/GGUF; ecosystem for mlx-vlm weak. 4.6 (1.2B) = edge model, below Qwen3.5-2B-ish. https://huggingface.co/openbmb/MiniCPM-V-4_5
### F13 [MEH] Moondream 3.1: caption/query/detect/point, 9B-A2B MoE, own Photon engine (NVIDIA Ampere+ and Apple Silicon), custom license, not vLLM. Skip for pipeline. https://huggingface.co/moondream/moondream3.1-9B-A2B
### F14 [MEH] LFM2.5-VL-3B: OCR with layout annotation, high throughput, MLX export, vLLM ok; scores low on MMMU-Pro 30.5 and OCRBenchv2-en 47.5; license lfm1.0 (non-Apache). Worth a throughput A/B only. https://huggingface.co/LiquidAI/LFM2.5-VL-3B
### F15 [MEH] OCR specialists for two-stage design (VLM for scene/people, specialist for on-screen text): PaddleOCR-VL-1.6 (+GGUF, MLX-8bit), LightOnOCR-2-1B ("keeps winning" per r/LocalLLaMA commenter), GLM-OCR, DeepSeek-OCR, dots.ocr, HunyuanOCR, NuExtract3 (4B, built on Qwen3.5-4B, Apache-2.0, FP8/W4A16/W8A8 variants; image->markdown/structured extraction). https://www.reddit.com/r/LocalLLaMA/comments/1tn8utn/nuextract3_released_openweight_4b_vlm_for/ . Florence-2 (2024) not researched in depth; superseded by these for OCR.
### F16 [NOISE] Qwen3.6/3.8 (27B, 35B-A3B, 125B+): too big for 12GB (35B-A3B possible on 3060 via llama.cpp CPU-offload per r/LocalLLaMA but not vLLM/throughput-friendly). Gemma-4-26B/31B: too big. InternVL3.5 (poor on facts per F3, no mlx-vlm). Phi-4-MM, Kimi-VL-A3B, SmolVLM2, Ovis2.5, Florence-2: stale/below Qwen3.5-2B. Mage-VL: video-codec model, ~Qwen3-VL-4B on images, no gain for stills.
### F17 [MEH] Hacker News: only elite hit with signal = jwr/tredre3 thread (F4); site:news.ycombinator.com search otherwise returned 2024-25 Qwen2-VL items (stale). Reddit r/LocalLLaMA threads are the main signal.

## 4. Ranked shortlist
### (a) RTX 3060 12GB via vLLM
1. KEEP baseline, but A/B config first: RedHatAI/Qwen3.5-4B-FP8-dynamic with chat_template_kwargs enable_thinking=false (every independent test: thinking hurts vision), bump vLLM to latest 0.29 if possible, disable prefix caching if hangs. Free win.
2. Qwen/Qwen3-VL-4B-Instruct-FP8 (official, ViT kept bf16 by config, pure attention so no hybrid-cache bugs, non-thinking by design). Official OCRBench 80.8 < 85.0, and independent bench 65.9 (unstable) -> likely a regression in quality, but the safest vLLM path. Test as fallback if Qwen3.5 hybrid bugs bite.
3. Qwen3.5-9B 4-bit: QuantTrio/Qwen3.5-9B-AWQ or RedHatAI/Qwen3.5-9B-quantized.w4a16 (ViT ignored in quant config). Best official OCR (89.2). Expect ~2x slower than 4B; independent bench says no gain over 4B for describing. Run only if 4B OCR misses text.
4. Qwen/Qwen3-VL-8B-Instruct-FP8 (9.5 GB weights, probably OOM/tight with 8 concurrent) or cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit: best independent OCR/mixed-script scorer among non-thinking models. Try AWQ first.
5. Throughput option: Qwen/Qwen3.5-2B (OCRBench within 0.5 of 4B) if 2 kf/s is too slow, accept weaker people/scene description.
Skip: Gemma 4 (E4B weak, 12B doesn't fit FP8, needs max_soft_tokens=1120 and newer vLLM), MiniCPM-V 4.6 AWQ (load bug on 3060), InternVL3.5, Moondream.
Specialist add-on: PaddlePaddle/PaddleOCR-VL-1.6 or lightonai/LightOnOCR-2-1B for the verbatim-text field only, if OCR recall is the actual weakness.

### (b) Apple Silicon via mlx-vlm
1. mlx-community/Qwen3.5-4B-MLX-4bit (or 8bit variant for accuracy; independent bench: Q8 of 4B scored LOWER due to thinking timeouts, so run nothink). Same model family => same prompts/JSON schema as the 3060 path.
2. lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit (or 4bit): strongest OCR in independent tests, no thinking mode to disable; needs ~16GB+ Mac for 8bit.
3. mlx-community/Qwen3.5-9B-4bit / lmstudio-community/Qwen3.5-9B-MLX-4bit on >=24GB Macs.
4. lmstudio-community/Qwen3-VL-4B-Instruct-MLX-8bit as cheap fast option (61 tok/s M2 Max Ollama) but quality unstable.
5. Maybe: LiquidAI/LFM2.5-VL-3B-MLX-8bit for throughput A/B; mlx-community/gemma-4-12B-it-qat-4bit only if you want Gemma (slowest of the set, 21 tok/s).
mlx-vlm has continuous batching in server, so concurrency tests are possible.

## 5. Unresolved
- Why exactly did FP8 vision-tower quantization break? RedHatAI config already ignores the visual blocks; need your repro (own quantization via llm-compressor? ignore regex not matching Qwen3.5 module names? vLLM 0.21 predates/lags Qwen3.5 fix?). Not found as a filed issue.
- Qwen3-VL-8B/4B official OCRBench/DocVQA/ChartQA numbers (card only shows images; tech report table not extracted). Qwen3-VL-8B Qwen3.5-comparison numbers are from memory-level knowledge, left blank on purpose.
- Throughput on 3060 for every candidate: no public data. Needs own benchmark (kf/s at 8 concurrency, 768px, JSON out).
- Does enable_thinking=false for Qwen3.5-4B in vLLM improve your JSON quality/speed? Only llama.cpp evidence.
- Gemma 4 E4B with max_soft_tokens=1120 on vLLM never measured against Qwen3.5-4B on OCR-heavy keyframes.
- Chapter-summary-from-transcript text task: no text-only comparison done (Qwen3.5-9B/Qwen3-VL-8B text scores MMLU-Pro 71.6 for Qwen3-VL-8B, Qwen3-VL-4B 67.1 per tech report https://arxiv.org/html/2511.21631; Qwen3.5 text table not extracted).
- Licenses unverified: Phi-4-reasoning-vision, Ministral-3, Molmo2, Kimi-VL.
- HN coverage thin; idp-leaderboard.org and OpenCompass/OCRBench v2 leaderboards not fetched.
