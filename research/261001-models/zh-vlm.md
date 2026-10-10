# ZH community: small VLMs for keyframe caption + OCR (RTX 3060 12GB / Apple Silicon)
Gatherer: ZH-CN only. Date: 2026-10-01. Tags: GEM = measured/first-hand, usable; MEH = partial/indirect; NOISE = farm/estimate/off-topic.

## TL;DR (brutal)
- Hard ZH measured numbers for 3060-12GB + vLLM + keyframe workload: NONE found. Nobody in ZH forums publishes img/s on a 3060. Your 2 frames/s baseline has no ZH comparator.
- ZH consensus for Chinese OCR + scene description at <=9B: Qwen family (Qwen3.5 small, Qwen3-VL) beat Gemma 4 / Molmo / InternVL on Chinese text. For pure OCR the ZH crowd says: use an OCR-specialist (GLM-OCR 1.3B, PaddleOCR-VL 0.96B, HunyuanOCR 1.1B, OvisOCR2 0.8B), not a general VLM.
- Contested: Qwen3.5-small vs Qwen3-VL-4B. Benchmarks favour 3.5; two hands-on ZH reports (object ID, ID-card OCR) say 3-VL as good or better. Needs your own A/B on 100-300 frames.
- Qwen has NO newer small VL than Qwen3.5 0.8/2/4/9B (checked HF API 2026-10-01). Qwen3.6 = 27B + 35B-A3B only; Qwen3.8 = 27B, Flash-Next (180B). Newest small non-Qwen contenders: MiniCPM-V 4.6 (1.3B), Gemma 4 E4B/12B (Google, ZH users test it), GLM-4.6V-Flash (10.3B).
- Overthinking in Qwen3.5 small models is the #1 ZH complaint; must pass enable_thinking=false.

## Candidate table (exact HF ids; params = safetensors total incl. vision tower)
| Model | HF id | Params | ZH verdict |
|---|---|---|---|
| Qwen3.5-4B (baseline) | Qwen/Qwen3.5-4B | 4.66B | OK OCR, weaker understanding than 9B; overthinks |
| Qwen3.5-9B | Qwen/Qwen3.5-9B | 9.65B | ZH favourite "small steel cannon"; Q4 ~6GB; best small for Chinese vision |
| Qwen3.5-2B | Qwen/Qwen3.5-2B | 2.27B | "OCR beast"; fine for short text, weak on formulas/blur |
| Qwen3-VL-4B | Qwen/Qwen3-VL-4B-Instruct (+ -FP8) | 4.44B | Still preferred by some over 3.5-4B; no think-mode trap (Instruct) |
| Qwen3-VL-8B | Qwen/Qwen3-VL-8B-Instruct | ~8.8B | ZH forum OCR winner Oct 2025; API-validated |
| MiniCPM-V 4.6 | openbmb/MiniCPM-V-4.6 (-AWQ, -GPTQ, -BNB, -gguf) | 1.3B (SigLIP2-400M + Qwen3.5-0.8B) | fastest/smallest; accuracy ~Qwen3.5-2B claimed; Chinese-board: "很多遗漏，错字" on hard OCR |
| MiniCPM-V 4.5 | openbmb/MiniCPM-V-4_5 | 8.7B | superseded; little 2026 ZH measurement found |
| InternVL3.5-4B/8B | OpenGVLab/InternVL3_5-4B, -8B (+ -Flash variants) | 4.7B/8.5B | ZH: charts/math screenshots good; general/CN OCR behind Qwen; no new 2026 buzz |
| GLM-4.6V-Flash | zai-org/GLM-4.6V-Flash | 10.3B | tutorials only, no measured data; too big for 3060 at BF16 |
| GLM-OCR | zai-org/GLM-OCR | 1.33B | ZH "神中神" for printed text; no JSON-scene-description |
| HunyuanOCR | tencent/HunyuanOCR | 1.12B | OCR only; llama.cpp hang unless pre-resized |
| PaddleOCR-VL-1.6 | PaddlePaddle/PaddleOCR-VL-1.6 | 0.96B | doc parser; V2EX: rare subtle char hallucination |
| OvisOCR2 | ATH-MaaS/OvisOCR2 | 0.8B (Qwen3.5-0.8B base) | doc-page parser, OmniDocBench top |
| Ovis2.5-9B / 2B | ATH-MaaS/Ovis2.5-9B (old AIDC-AI org moved to ATH-MaaS) | 9.2B | no ZH measurement beyond launch posts |
| Keye-VL | Kwai-Keye/Keye-VL-1_5-8B ; Keye-VL-2.0-30B-A3B | 8.7B / 30B | ZH real-world feedback: none found |
| Kimi-VL-A3B | moonshotai/Kimi-VL-A3B-Instruct | 16.4B total (3B active) | stale (Apr 2025); no 2026 ZH feedback; MoE total weights too big for 12GB w/o heavy quant |
| Youtu-VL-4B | tencent/Youtu-VL-4B-Instruct | 5.3B | new (Jan 2026), custom_code; no ZH practitioner data |
| Gemma 4 E4B / 12B (non-CN, ZH users test) | google/gemma-4-E4B-it, gemma-4-12B-it | 8.0B (incl. per-layer embeds) / 12.0B | See below |
| Hunyuan "小" general VLM | none exists as open small general VLM; only HunyuanOCR | - | gap |

## Findings by question

### 1. Qwen3.5 small vs Qwen3-VL
- GEM: GitHub Discussion (ZH) object ID on bottled drinks via ollama + llama.cpp: "qwen3.5_4b/8b ... 表现均远不如qwen3_vl_4b_guff". Anecdote, no screenshots, author admits llama.cpp novice. https://github.com/QwenLM/Qwen3.8/discussions/133
- GEM: linux.do ID-card/credential OCR, A100 BF16 vLLM: Qwen3.5-27B misreads names (许晨->许展) where Qwen3-VL-8B was right; same on Bailian API with 3.5-plus. Replies: 35B-A3B "有概率识别对，但是错误率也挺高"; 9B "感觉还行". Poster later also notes thread veteran: need 100-300 samples for statistical meaning. https://linux.do/t/topic/1703503
- MEH: Reddit (EN) thread notes Qwen3.5-4B beats Qwen3-VL-30B-A3B on most vendor benchmarks, but vendor numbers only. https://www.reddit.com/r/LocalLLaMA/comments/1s1vp9i/
- MEH: Zhihu summary repeating vendor benchmarks (MathVision 4B 74.6 etc.). https://zhuanlan.zhihu.com/p/2011914556159582443 (zhaode.wang copy: https://zhaode.wang/blog/2026/qwen35-eval/)
- Vision tower sizes (zhaode.wang): 0.8B 12 layers/768d; 2B and 4B both 24 layers/1024d; 9B 27 layers/1152d. 4B shares the 2B's ViT size - OCR gap 2B->4B comes from LLM, not vision.
- Interpretation: for caption + verbatim text, 3.5-4B vs 3-VL-4B is unresolved. Keep both in A/B. Qwen3-VL-4B-Instruct has no thinking mode -> sidesteps overthink.

### 2. OCR quality, Chinese + English
- GEM (linux.do, 2B MLX 4bit + 2B Q6 GGUF): "一般 OCR（十多个字）足够，数学公式提取能力不足...稍微模糊就容易错误"; another user, 2B Q6 GGUF full-page OCR of scanned standards, 12 ID cards in one image -> table "几乎没有错误"; 0.8B "适合给图片打标" not OCR; "4B 能识别信息，但理解力差一点；9B 惊喜". https://linux.do/t/topic/1682421
- GEM (linux.do Oct 2025): Qwen3-VL vs Gemini 2.5 on canteen menu: Qwen3-VL "一点毛病没有，全对", Gemini wrong x2; company experimental project: "只有qwen的ocr识别正确率能到90%左右". Also resolution matters: "大模型处理识别前会统一缩放...字体太小就识别错误". https://linux.do/t/topic/1045861
- GEM (cnblogs ocean1ee, Intel Ultra 7 165U, llama.cpp+SYCL, Q4/Q8 GGUF, 2026-05): printed doc OCR basically fine on Qwen3VL-2B, Gemma4-E2B, HunyuanOCR, GLM-OCR; Gemma4 "中文识别出错率似乎高一点"; handwriting+print mixed -> Qwen3VL/Hunyuan/GLM "持续重复输出" (repetition loop). https://www.cnblogs.com/ocean1ee/p/19983196
- GEM (same author): Qwen3.5-0.8B vs MiniCPM-V-4.6: both fine on plain text; local-field JSON extraction from receipt: MiniCPM "常有识别错误"; mixed hand/print: Qwen "偶尔持续重复输出", MiniCPM "很多遗漏，错字"; vertical classical Chinese: Qwen "有遗漏，错字", MiniCPM "质量较差". Speed 18/s vs 20-22/s decode (Intel iGPU). https://www.cnblogs.com/ocean1ee/p/20112824
- GEM (V2EX): PaddleOCR-VL has hard-to-notice hallucination (one char more/less, sentence still reads fluent). Relevant to "verbatim" requirement. https://www.v2ex.com/t/1224765
- MEH (linux.do): "Gemma E4B just can't translate Chinese text from images, while Qwen does" (Reddit, LM Studio, possibly template issue). https://www.reddit.com/r/LocalLLaMA/comments/1sbec70/
- MEH (Zhihu Q&A, search snippet; page 403): for Chinese visual recognition "还是Qwen3.5系列比较好用，Qwen3.5-9B不开思考模式就已经表现非常优秀"; 4B also "非常不错"; Gemma-4 vs Qwen-3.5 close on foreign exhibits. https://www.zhihu.com/question/2023493220097598697
- Consensus on forums: general VLM OCR hallucinates "corrects" erroneous source text (grading use-case) ; specialist OCR (PaddleOCR / GLM-OCR / MinerU) recommended for pure text; use a VLM for understanding. linux.do #28 (AlexChu1996): "OCR场景直接上那几个OCR特化模型吧...跑的更快更省显存" https://linux.do/t/topic/1703503 ; https://linux.do/t/topic/2901249 ("打印用glm-ocr, 手写用qwen3.7-flash").
- Implication: your single-pass JSON (scene + verbatim text) is exactly the case ZH users split in two. Option: caption model + GLM-OCR/PaddleOCR-VL pass only on frames flagged text-heavy.

### 3. Speed / VRAM (almost all non-3060, non-vLLM)
- GEM: OvisOCR2 (0.8B, Qwen3.5 arch) on RTX 5090, vLLM 0.22.1: 590 pages/min @ cc=32 (~2.3k image tok + ~600 out/page); single-stream ~570 tok/s decode; KV cache <11%; VRAM irrelevant; saturated throughput prefill-bound. Reddit EN but best vLLM-with-Qwen3.5-arch data. https://www.reddit.com/r/LocalLLaMA/comments/1ux3nps/
- GEM (zicode.com, RTX 4060 Laptop 8GB, Ollama): gemma4:e4b 54-56 tok/s, 3.1GB hot; gemma4:12b 7-9 tok/s, 5.8GB; qwen3:4b 38-43 tok/s. e4b image describe ok in Chinese with think:false; 12b >40s per image. https://zicode.com/blog/local-open-model-benchmark-qwen-gemma/
- GEM (V2EX, RTX 3080 10G, Q4): Gemma4-12B 85-105 tok/s vs Qwen3.5-9B+MTP 75 tok/s. https://www.v2ex.com/t/1219170 (snippet only)
- GEM (V2EX): Qwen3.5-4B on a 6G P106 + 8G CMP40HX junk-GPU box "40+ T/s", think+answer 2-3s. https://global.v2ex.com/t/1195703
- GEM (linux.do Orange Pi/Jetson Orin Nano 7.4GB): MiniCPM-V 4.6 Q4 42 tok/s vs Gemma4-E2B 6 tok/s. https://linux.do/t/topic/2241870
- GEM (zhaode.wang, M3 Pro CPU 4-thread MNN HQQ-4bit): Qwen3.5 decode 0.8B 140, 2B 70, 4B 60, 9B 50 tok/s; TTFT 0.5/0.9/1.1/1.5s; 4-bit size 0.53/1.37/2.59/4.85 GB. https://zhaode.wang/blog/2026/qwen35-eval/
- MEH (linux.do): Qwen3.5-9B Q4 6.1GB VRAM, >=8GB cards OK. https://linux.do/t/topic/1690082
- MEH (MiniCPM official, ZH news): MiniCPM-V 4.6 TTFT 75.7 ms at 3136^2 (2.2x faster than Qwen3.5-0.8B), ~1.5x token throughput vs Qwen3.5-0.8B, 4x/16x mixed visual-token compression, 6GB RAM phone. Vendor claim. https://zhuanlan.zhihu.com/p/2038188070856561070 ; HF card https://huggingface.co/openbmb/MiniCPM-V-4.6
- MEH (Qwen3.8/3.6 MoE, 5090 Q5, linux.do): 35B-A3B >230 tok/s; "最好用" for private vision-inspection set vs Qwen2.5-VL-7B-finetuned and Gemma4-26B-A4B. Does not fit 12GB w/o CPU offload. https://linux.do/t/topic/2382261
- NOISE: yomxxx.com "本地VLM横评 2026-06" - aggregator of alleged Reddit users ("throwaway_eval"), claims Qwen3.6 27B uses an "外接视觉编码器" and 35B-A3B needs 8-12GB VRAM (wrong for Q4 weights ~20GB). Do not cite. https://yomxxx.com/posts/2026-06-26-local-vlm-benchmark-june-2026-qwen3-vl-tools
- NOISE: xmsumi.com Mac mini M4 tok/s are self-labelled "预估"; CSDN/devpress 星图 posts (Qwen3.5-9B "12GB", "95% quality kept") have no method.
- Gap: no ZH source gives images/s or per-frame latency for 3060 12GB or for Apple Silicon VLM *prefill* (vision tower) - all numbers are decode tok/s. For frame captioning, prefill+ViT matters more.

### 4. Quantization vs vision tower
- GEM (linux.do 2382261): "APEX量化...更省显存更快，但是视觉能力我测试起来好像差一点"; GGUF vs AWQ(vLLM) "能力差不太多"; GGUF loads much faster. https://linux.do/t/topic/2382261
- GEM (OvisOCR2 notes): disable MM processor cache for unique frames (`--mm-processor-cache-gb 0`); fake 1000+ p/min otherwise. Also needs `ninja` installed or `--gdn-prefill-backend triton` crashes at startup (FileNotFoundError). https://www.reddit.com/r/LocalLLaMA/comments/1ux3nps/
- GEM (vLLM docs): `--language-model-only` skips vision encoder (text only, not for you); multimodal flags `--mm-encoder-tp-mode data`, `--mm-processor-cache-type shm` (multi-GPU oriented). https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html
- MEH: Ampere has no native FP8; vLLM runs FP8 as weight-only W8A16 via Marlin (3060 baseline = W8A16, saves VRAM but not compute vs BF16 of a 4B model). https://github.com/vllm-project/vllm/pull/5975
- MEH: mmproj precision: Qwen3-VL vision tower natively BF16; ship mmproj F16/BF16 or Q8_0 (official Qwen3-VL GGUF offers mmproj FP16 & Q8_0). https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct-GGUF ; https://www.reddit.com/r/LocalLLaMA/comments/1okz6ko/ . A bf16 mmproj failed in ik_llama.cpp for Qwen3.5: https://github.com/ikawrakow/ik_llama.cpp/issues/1554
- NOISE: vision-tower-damage claims in CSDN ("保持视觉部分FP16精度...95%以上") - no data. Direct measured "AWQ broke ViT" ZH evidence: NOT FOUND. Reddit EN thread says Qwen3.5 27B nvfp4 "produced nonsense", AWQ hangs in tool calls: https://www.reddit.com/r/LocalLLaMA/comments/1s9iyrw/
- Practical: FP8 of 4B on 3060 is fine for VRAM (weights ~4.7GB), but check whether your FP8 ckpt keeps ViT BF16 (Qwen official FP8 ckpts typically leave visual in BF16 - verify `quantization_config.ignore`).

### 5. Deployment pitfalls (ZH)
- Thinking: Qwen3.5 small models default to thinking. V2EX: qwen3.5:4b "说Hi思考了40秒"; "关闭思考之后好像也不比上一代qwen3强"; another: 9B/4B "无限思考，10分钟没响应", reverted to qwen3-4b; setting max-think-tokens "没用". https://global.v2ex.com/t/1195703 ; https://www.v2ex.com/t/1203064
  Fix for vLLM: `--default-chat-template-kwargs '{"enable_thinking": false}'` (Zhihu https://zhuanlan.zhihu.com/p/2016192121036637589) or per-request `chat_template_kwargs`. llama.cpp: `--chat-template-kwargs '{"enable_thinking":false}'` + recommend non-thinking sampling (temp 0.7, top-p 0.8, top-k 20, presence 1.5). Note: Qwen3.6 default differs (QwenLM/Qwen3.6 discussion #90). vllm-ascend issue #7094 reports accuracy anomaly when disabling thinking via "thinking":false key (wrong key; use enable_thinking).
- Repetition/loops (linux.do 2382261 #7): "qwen从3.5开始经常会反复思考跳不出来"; small models + long context. cnblogs ocean1ee: handwritten+print images cause endless repeat output on Qwen3VL/GLM-OCR/Hunyuan. Mitigation: set max_tokens, repetition/presence penalty, short context per frame.
- llama.cpp image preprocess (cnblogs, SYCL backend): Qwen3-VL/GLM hang at "encoding image slice..." for tall/odd aspect images unless `--image-max-tokens` set or you pre-resize: long side <=1000px (tall receipts <=1400); keep W,H divisible by patch_size*n_merge (Qwen3-VL 32, GLM-OCR 28, Gemma4 48). MiniCPM-V 4.6: slice-based (overview + <=9 tiles); avoid 1:1 and dims that are multiples of 448; best 4:3 (1512x1176). Your 768px frames: 768 = 24*32 so fine for Qwen3-VL. https://www.cnblogs.com/ocean1ee/p/19983196 , https://www.cnblogs.com/ocean1ee/p/20112824
- vLLM version: early Mar 2026 posts say vLLM needs nightly for Qwen3.5 (linux.do 1680593/1683389); now mainline (v0.17+ per CSDN guides). Related issue: vLLM #44985 (0.22.0 Qwen3.5-9B offline fails); vLLM #38411/#37992 vision-encoder crashes on SM100/ROCm (not your GPU).
- mlx-vlm: needs >=0.3.12 for Qwen3.5 (Reddit EN https://www.reddit.com/r/LocalLLaMA/comments/1rezq19/); mlx-vlm issue #832 vision tower re-runs on every turn with Qwen3.5 - irrelevant for one-shot frames but mention `--vision-cache-size`. https://github.com/Blaizzy/mlx-vlm/issues/832 ; https://txtmix.com/posts/tech/mlx-vlm-apple-silicon-vlm-inference-guide/
- Ollama vs MLX on Mac: MLX ~2x Ollama decode for Qwen3.5 (EN sources: https://insiderllm.com/guides/qwen35-mac-mlx-vs-ollama/). V2EX user: on M1/M2 no bf16 HW -> use fp16 not bf16; DFlash/MTP "净收益为负" on M1. https://www.v2ex.com/t/1203064
- mlx-vlm-kit repo (Mac, Qwen3-VL-4B MLX, local caption service, ZH): https://github.com/webkubor/mlx-vlm-kit

## 2026 additions worth testing
1. MiniCPM-V 4.6 (1.3B): best speed/size; AWQ/GPTQ/BNB/gguf official; accuracy below Qwen3.5-0.8B on tough OCR per one hands-on, above per vendor. Candidate as cheap pass 1.
2. Gemma 4 E4B (Google): 56 tok/s on 8GB laptop, good Chinese description with think:false, but weaker Chinese OCR. Not a ZH lab; ZH says Qwen better for Chinese text.
3. OvisOCR2 0.8B / GLM-OCR 1.3B / PaddleOCR-VL-1.6 0.96B: dedicated OCR; only worth it as second pass.
4. Qwen3.5-9B Q4 (AWQ/GGUF) fits 12GB (~6GB weights) and is ZH's recommended quality step up; likely ~half the img/s of 4B.
5. Not found: any new <10B VL from Qwen after 3.5, InternVL4, Ovis3, Hunyuan small general VLM, Keye small 2.0, Kimi-VL 2.

## What this adds / contradicts vs Qwen3.5-4B baseline
- Adds: Qwen3.5-4B is still the ZH default small model for Chinese vision, but ZH hands-on says (a) 9B is clearly better in understanding, (b) Qwen3-VL-4B-Instruct may match/beat it on recognition and has no thinking trap, (c) 2B is surprisingly OK at short-text OCR.
- Adds: thinking must be off; confirm your vLLM call sets enable_thinking=false or you are paying token latency (2 frames/s suggests you already do).
- Adds: unique frames -> `--mm-processor-cache-gb 0` for honest throughput; install `ninja`; consider throughput prefill-bound (image tokens dominate) -> lowering image tokens (768px -> fewer patches) is the biggest speed lever, not model swap.
- Adds: repetition loops are the failure mode on hand+print mixed text; set max_tokens + penalties.
- Contradicts: yomxxx-style claims that 35B-A3B fits 12GB (NOISE). Contradicts vendor claim "3.5-4B > Qwen3-VL-4B everywhere" only weakly (2 anecdotes).
- Nothing in ZH sources shows a model that is both faster than 3.5-4B on a 3060 AND equal in Chinese OCR, except possibly Qwen3-VL-4B (similar speed) and MiniCPM-V 4.6 (faster, lower accuracy).

## Exact repo ids to evaluate
Qwen/Qwen3.5-4B, Qwen/Qwen3.5-4B-FP8 (check exists), cyankiwi/Qwen3.5-4B-AWQ-4bit, QuantTrio/Qwen3.5-4B-AWQ, unsloth/Qwen3.5-4B-GGUF, Qwen/Qwen3.5-9B, cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit, Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-4B-Instruct-FP8, Qwen/Qwen3-VL-4B-Instruct-GGUF, lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit, lmstudio-community/Qwen3.5-9B-MLX-4bit, mlx-community/Qwen3.5-9B-MLX-4bit, openbmb/MiniCPM-V-4.6, openbmb/MiniCPM-V-4.6-AWQ, openbmb/MiniCPM-V-4.6-gguf, google/gemma-4-E4B-it, zai-org/GLM-OCR, PaddlePaddle/PaddleOCR-VL-1.6, tencent/HunyuanOCR, ATH-MaaS/OvisOCR2, zai-org/GLM-4.6V-Flash (+ unsloth/GLM-4.6V-Flash-GGUF, lmstudio-community/GLM-4.6V-Flash-MLX-4bit), OpenGVLab/InternVL3_5-4B.

## Unresolved
1. No ZH number for img/s on 3060 12GB with vLLM; need own benchmark (same 100-300 frames, same image-token budget, thinking off).
2. Qwen3.5-4B vs Qwen3-VL-4B Chinese OCR head-to-head with real screenshots: only anecdotes.
3. Does FP8 / AWQ of Qwen3.5-4B degrade ViT? No ZH measurement; check each ckpt's ignore list; compare BF16 vs FP8 on text-heavy frames.
4. Apple Silicon: no ZH prefill/ViT latency for 4B VLMs (only decode tok/s); mlx-vlm 4B per-image seconds unknown.
5. Keye-VL, Kimi-VL, Ovis2.5, InternVL3.5: no ZH practitioner measurements surfaced (search engines weak; Zhihu pages 403 to fetchers).
6. Zhihu question 2023493220097598697 and zhuanlan 2052765345404998716 / 2050250537350836806 / 2042044379725043649 (llama.cpp --mmproj four-VLM OCR bake-off) were not fetchable; worth reading by browser (bsk) - last one likely has OCR bake-off.
7. linux.do/t/topic/1563022 (GLM OCR vs DeepSeek OCR 2) and /671030 (中文OCR哪家强 series) not read.
8. Not covered: ModelScope model pages (only via search snippets); bilibili video tests (BV1HwUzBNEWa HunyuanOCR vs MinerU; BV13JPezGEEB Qwen3.5 small VRAM/OCR) not transcribed.

## Sources
GitHub: https://github.com/QwenLM/Qwen3.8/discussions/133 ; https://github.com/vllm-project/vllm/pull/5975 ; https://github.com/ikawrakow/ik_llama.cpp/issues/1554 ; https://github.com/Blaizzy/mlx-vlm/issues/832 ; https://github.com/vllm-project/vllm-ascend/issues/7094 ; https://github.com/vllm-project/vllm/issues/44985 ; https://github.com/webkubor/mlx-vlm-kit ; https://github.com/datawhalechina/self-llm (Qwen3.5-4B vLLM doc)
Forums: https://linux.do/t/topic/1703503 ; /1682421 ; /1045861 ; /2901249 ; /2382261 ; /2241870 ; /2826416 ; /1690082 ; https://www.v2ex.com/t/1203064 ; https://global.v2ex.com/t/1195703 ; https://www.v2ex.com/t/1224765 ; https://www.v2ex.com/t/1219170 ; https://www.v2ex.com/t/1131060
Blogs: https://www.cnblogs.com/ocean1ee/p/19983196 ; https://www.cnblogs.com/ocean1ee/p/20112824 ; https://zicode.com/blog/local-open-model-benchmark-qwen-gemma/ ; https://zhaode.wang/blog/2026/qwen35-eval/ ; https://zhuanlan.zhihu.com/p/2038188070856561070 ; https://zhuanlan.zhihu.com/p/2016192121036637589 ; https://www.zhihu.com/question/2023493220097598697
Official: https://huggingface.co/openbmb/MiniCPM-V-4.6 ; https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html ; HF API (param counts, createdAt) queried 2026-10-01
English cross-ref: https://www.reddit.com/r/LocalLLaMA/comments/1ux3nps/ ; /1s1vp9i/ ; /1s9iyrw/ ; /1sbec70/ ; /1okz6ko/
Rejected as NOISE: yomxxx.com, xmsumi.com, CSDN/devpress 星图 posts
