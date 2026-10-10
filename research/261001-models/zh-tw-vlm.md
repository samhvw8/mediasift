# zh-TW community: local VLM (description + OCR) on consumer GPU / Mac

Date: 2026-10-01. Lang: Traditional Chinese queries. Tools: parallax web_search/fetch, WebSearch/WebFetch, HF API.

## Verdict (plain)
TW coverage is THIN again. No PTT / iThome 鐵人賽 / Mobile01 post found with real measurements of local VLM OCR (zh-TW or Vietnamese), VRAM, tok/s, or quantization effect on vision. Parallax site: searches on ptt.cc, ithelp, mobile01 returned 0 results (only 2/10 engines answered). WebSearch returned only generic PTT board rules and an old Soft_Job OCR beginner thread. Everything "TW" below is vendor/blog/HF-card material, not forum benchmarks. Don't treat TW sources as evidence; use the English/ZH benchmarks from sibling agents.

## Findings

### GEM
- xCloudinfo (TW company) zh-TW OCR fine-tunes on Gemma 4 E4B: only source with actual CER numbers for Traditional Chinese. https://huggingface.co/xCloudinfo/Gemma-4-E4B-xReadable-zhTW-OCR-v1
  - Printed CER: unseen fonts Songti 0.00%, STHeiti 0.00%; held-out 8-font set (200 imgs) 5.08%; vs nemotron-ocr-v2 2.78% / 9.72%. Own synthetic-font eval, vendor-reported, not independent.
  - Handwriting CER ~40% clean, 64-88% messy ("experimental").
  - LoRA on language side only, vision tower frozen (= base Gemma 4 vision). Ships Q4_K_M GGUF ~5GB + mmproj-f16 ~1GB, runs on llama.cpp/Ollama/LM Studio. Card: "量化 GGUF 精度略低於 bf16 safetensors，關鍵用途建議用 safetensors". 8.0B total params (E4B counts embeddings).
  - Relevance: synthetic font renders, not video frames; no Vietnamese. But evidence that Gemma 4 E4B base is a viable zh-TW OCR base. Try base google/gemma-4-E4B-it vs Qwen3.5-4B on own frames.
- HF landscape (API, 2026-10-01): Qwen3.5 small VLMs are the volume leaders (9B 9.3M dl, 4B 7.5M, 2B 4.9M); Gemma 4 (E4B/12B/26B-A4B/31B) vision models; AWQ 9B `QuantTrio/Qwen3.5-9B-AWQ` 685k dl; MLX `lmstudio-community/Qwen3.5-9B-MLX-4bit/8bit` ~600k dl each. https://huggingface.co/api/models?search=qwen3.5&pipeline_tag=image-text-to-text&sort=downloads

### MEH
- Breeze2 (MediaTek Research), Llama-Breeze2 3B/8B: zh-TW-first VLM, InternViT-300M-448px + Llama 3.2. Paper: MMMU 3B 38.2 / 8B 43.2 (Llama-3.2-11B 41.7); TMMBench 41.7 / 45.2 (11B: 42.4). OCR data 2.45M samples, "InternVL pretrained model demonstrates a higher level of comprehension for Chinese characters". No OCR benchmark, no VRAM for vision config. Jan 2025 vintage, Llama 3.2 base; behind Qwen3-VL/Qwen3.5/Gemma 4 for OCR almost certainly. HF downloads ~1k/mo (dead). Only quants: mradermacher text-only GGUFs, MXLouis MLX (3B 4/8bit, 8B text-only). No working vision GGUF for the community. https://arxiv.org/html/2501.13921v3 ; https://huggingface.co/MediaTek-Research/Llama-Breeze2-8B-Instruct ; https://www.ithome.com.tw/news/167459 (iThome news on architecture, LLaVA-style, no tests)
- TAIDE Gemma-3-TAIDE-12b-Chat-2602: card says "由於沒有進行圖片資料的訓練，故建議僅以文字輸入" = TEXT ONLY, not a VLM. Gated. Skip. https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602 . TAIDE-on-Gemma-4-26B-A4B by xCloudinfo (image-text-to-text, GGUF) is a community finetune, no vision eval seen.
- Generic TW blogs recommending Qwen2.5-VL-7B (好中文, OCR 強) for single RTX 3090/4090 or lower; outdated, no numbers. https://pccicblog.wordpress.com/2026/01/03/%E8%A6%96%E8%A6%BA%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B-vlm/
- HackMD (YungHuiHsu, TW): dots.ocr beats DeepSeek-OCR on OmniDocBench in many cases, DeepSeek-OCR far less memory; personal note: expert VLMs beat Docling/Unstructured pipelines on A4 docs. Documents, not video frames. https://hackmd.io/@YungHuiHsu/ryNy1-IClg
- tenten.co (TW): Gemma 4 26B-A4B ~16GB Q4, E4B download ~9.6GB, ollama gemma4:e4b / gemma4:26b; mlx-vlm TurboQuant ~4x memory cut. Marketing-ish, no OCR or tok/s measurements. https://tenten.co/learning/gemma-4-26b/
- Search snippet (not fetched, English aggregator sites): Qwen3-VL-8B Q4_K_M ~6.1GB GGUF, fits 12GB; "3-5% OCR accuracy drop at Q4_K_M" claim (codersera / insiderllm, unverified). Gemma-3 27B OCR 42.9% (snippet, source unclear). Mac mini M4 32GB: Qwen3.5 9B ~22-35 tok/s MLX/LM Studio (willitrunai.com), text decode not VLM prefill.

### NOISE
- codelove.tw reposts of Qwen3.5 small-model guides (likely simplified-origin translations, no vision/VRAM numbers): https://codelove.tw/@tony/post/qeWD13
- ithelp Day 4 Gemma 3 4B Ollama install tutorial (install only): https://ithelp.ithome.com.tw/articles/10382097
- Aliyun / Alibaba Cloud qwen-vl-ocr docs (cloud API, not local, though Qwen-VL-OCR is built on Qwen3-VL architecture): https://www.alibabacloud.com/help/tc/model-studio/qwen-vl-ocr
- 0xPabloLI/inside-china-ai#264: planned VLM benchmark, no results; only baseline Qwen3-VL-2B ~3.5 s/image vs GLM-4.1V-9B ~28.5 s/image (hardware unstated). https://github.com/0xPabloLI/inside-china-ai/issues/264
- Chinese social (Tieba/Juejin/Hupu): "qwen3-vl-30b-a3b 视觉识别效果很好拿来做ocr十分不错" https://tieba.baidu.com/p/10181123474 ; PaddleOCR-VL-1.6 claims #1 OCR https://bbs.hupu.com/639674445.html . Not TW, anecdotal (sibling ZH agent's turf).
- English reddit surfaced (not TW, pointers for sibling EN agent): r/LocalLLaMA "Qwen 3.5 2B is an OCR beast" https://www.reddit.com/r/LocalLLaMA/comments/1rivzcl/qwen_35_2b_is_an_ocr_beast/ ; "Qwen3-2B-VL for OCR ... --max-model-len 16384 errors on 10GB 3080, use ~8000" https://www.reddit.com/r/LocalLLaMA/comments/1p4hkaf/qwen32bvl_for_ocr_is_actually_insane_dockerized/

## Takeaways for framesift
1. No TW evidence that anything beats Qwen3.5-4B for zh-TW OCR. Breeze2 and TAIDE are not competitive candidates (old / text-only).
2. Only candidate worth an A/B from this sweep: Gemma 4 E4B (base) and the xCloudinfo zh-TW OCR finetune; also Qwen3.5-9B (AWQ on 12GB tight; Q4 GGUF/MLX 4bit fine). Must test on Vietnamese diacritics yourself: no community data in zh-TW.
3. Quantization effect on vision: only vendor caveat (GGUF slightly worse than bf16, mmproj stays f16). No TW measurement.

## Unresolved
- Any PTT/Dcard/Mobile01/iThome post with real VLM OCR measurements? Search engines may not index them (ptt.cc site: gave 0). Could try bsk on ptt.cc/Mobile01 native search (not done).
- Gemma 4 E4B vs Qwen3.5-4B on zh-TW + Vietnamese video frames: unmeasured.
- Does the xCloudinfo OCR finetune hurt scene description (LoRA on language side)?
- Qwen3.5 VLM quality under AWQ/FP8/GGUF vs bf16: no source.
- VLM prefill speed on Apple Silicon (mlx-vlm) with image tokens: only text decode numbers found.

## Sources
https://huggingface.co/xCloudinfo/Gemma-4-E4B-xReadable-zhTW-OCR-v1 ; https://arxiv.org/html/2501.13921v3 ; https://huggingface.co/MediaTek-Research/Llama-Breeze2-8B-Instruct ; https://www.ithome.com.tw/news/167459 ; https://huggingface.co/taide/Gemma-3-TAIDE-12b-Chat-2602 ; https://hackmd.io/@YungHuiHsu/ryNy1-IClg ; https://pccicblog.wordpress.com/2026/01/03/%E8%A6%96%E8%A6%BA%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B-vlm/ ; https://tenten.co/learning/gemma-4-26b/ ; https://huggingface.co/api/models (searches: breeze, taide, qwen3.5, gemma-4, zh-tw) ; others inline above.
