# en-frame-extract: cheap non-VLM frame -> text (OCR / tags / captions / UI parse)
Date: 2026-10-08. Lang: EN. Targets: Apple M1 Pro 16GB, RTX 3060 12GB. Tags: GEM / MEH / NOISE.
Prior VLM notes (not repeated): research/261001-models/en-vlm.md
"Measured" = I ran it on this M1 Pro (macOS 27.0.1) today with synthetic Vietnamese text images (6 lines, 30pt, 1600px wide). Small n, synthetic; treat as smoke test, not benchmark.

## 0. TL;DR
1. **Apple Vision `VNRecognizeTextRequest` .accurate = GEM for M1 Pro.** Measured: lists `vi-VT` (accurate only; .fast = 6 Western langs). Near-perfect Vietnamese diacritics. ~75-120 ms/image warm on full 1600x400 image. Free, zero deps, ANE/GPU. Caveats: macOS only (no 3060), 66 s first-ever-call model compile (second process: 294 ms), code punctuation errors (`->` read as `>`, `l`->`1`).
2. **PP-OCRv6 (Jun 2026) = GEM for speed/size, but Vietnamese is BROKEN.** Documented as 50 langs incl. Vietnamese; dict lacks all 88 tone-marked vowels U+1EA0-1EF9 (tiny also lacks ơ/ư). Measured via RapidOCR+PP-OCRv6 small: "Tiếng Việt" -> "Ting Vit". Maintainers: fix needs retraining, future model. Fine for English/code/CJK screens; unusable for Vietnamese until a fine-tuned rec model is used.
3. **Doc-OCR VLMs (0.8-1.2B) top OmniDocBench v1.6 but are page parsers, not scene-text/screen-recording tools.** Best new: Xiaomi-OCR-0 (0.8B, 2026-09-29), TeleOCR (1.2B), OvisOCR2 (0.8B), PaddleOCR-VL-1.6 (0.9B, MLX + GGUF exist). Use only as escalation tier for dense slides/docs, not per-frame.
4. **Tagging (b) is a dead field**: RAM++ weights from 2023, no successor; SigLIP2 zero-shot with a fixed vocab is the cheap path. NOISE for new work.
5. **Captioners (c)**: nothing small beats "OCR text + a 450M-1.6B VLM"; LFM2.5-VL-450M is the only new sub-500M with grounding but self-declared "not for fine-grained OCR". Florence-2/SmolVLM are 2024-25 stale.
6. **UI parsers (d)**: OmniParser v2 still the reference (YOLO + Florence-2 caption); repo got YOLOv9-E detector option Jul 2026. Holo3.1 (0.8B/4B) are agent/grounding models, not parsers.
7. Since 2026-10-01 almost nothing new: only TeleOCR quants (Oct 2-6), PepperOCR-VL (Oct 4, AGPL, 4B), Xiaomi-OCR-0 quants (Oct 5-7), PP-OCRv6 small ONNX re-uploads (Oct 7), RapidOCR denormal issue (Oct 2). Real news is Jun-Sep 2026.

## 1. Recommended stack for framesift (ranked by accuracy per resource)
| Hardware | Tier 1 (every kf) | Tier 2 (escalate: dense text/slides) | Tier 3 (meaning) |
|---|---|---|---|
| M1 Pro 16GB | Apple Vision accurate, langs `["vi-VT","en-US"]` (+zh-Hans etc.) | PaddleOCR-VL-1.6 MLX (0.9B) or Xiaomi-OCR-0 (0.8B, MLX not verified) | existing Qwen3.5-4B, only on kf where OCR text is thin |
| RTX 3060 12GB | PP-OCRv6 small/medium ONNX (en/zh/code); Vietnamese needs a fine-tuned rec model (see F3) | PaddleOCR-VL-1.6 / GLM-OCR / OvisOCR2 via vLLM (~2-3GB bf16) | existing Qwen3.5-4B |
Cost note: Tier 1 on Mac = $0, no model download, no new service. Needs a long-lived helper process (Swift CLI/daemon) because of the 66 s first-use compile.

## 2. (a) OCR
| Model | Params | License | Date | Metric (source) | Vietnamese | Runtime | Verdict |
|---|---|---|---|---|---|---|---|
| Apple Vision accurate | n/a (OS) | OS | OS-bundled, request rev 3 | not published; measured 5/6 Vietnamese lines exact at 30pt, 1 diacritic slip ("điều khiển"->"điều khiền") | YES `vi-VT` (measured list) | CoreML/ANE native, Swift/pyobjc | GEM (Mac) |
| PP-OCRv6 tiny/small/medium | 1.5M / 5.2M(rec) / 34.5M total | Apache-2.0 | 2026-06-08..10 | W-Avg rec acc: tiny 73.5, small 81.3, medium 83.2 vs PP-OCRv5_server 78.1, Qwen3-VL-235B 74.9 (screen subset: medium 82.5, small 79.7) [HF card](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec) | NO tone marks (dict bug) | Paddle, ONNX (official `_onnx` repos), RapidOCR, OpenVINO, GGUF/MLX community ports | GEM (non-VN) / MEH (VN) |
| RapidOCR 3.9.x (PP-OCRv6 ONNX wrapper) | same | Apache-2.0 | rel 2026-07-21 | measured 0.61-0.77 s/image full pipeline small det+small rec, onnxruntime CPU, M1 Pro | NO (measured) | pip, onnxruntime | GEM for Linux/3060 non-VN |
| PaddleOCR-VL-1.6 | 0.9B | Apache-2.0 | 2026-05-27 | OmniDocBench v1.6 96.34 ([Docsumo](https://www.docsumo.com/blog/best-ocr-models)) | claims 111 langs, unmeasured; VN handwriting SFT test CER 5.40% ([repo](https://github.com/khanhlinh-innomation/vlm-handwriting-ocr)) | PaddleOCR SDK, vLLM, GGUF, MLX (OpenGryd/PaddleOCR-VL-1.6-MLX-16bit), mlx-vlm | GEM (escalation) |
| Xiaomi-OCR-0 | 0.8B (Qwen3.5-0.8B base) | Apache-2.0 | 2026-09-29 | OmniDocBench v1.6 96.83, Real5 95.24, Wild 87.94; OCRBench 84.6, DocVQA 93.1 ([card](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0)) | unknown | transformers/vLLM; HF shows 3.9k dl, MLX unverified | MEH->GEM (new, unvetted; very young) |
| TeleOCR | 1.2B | Apache-2.0 | 2026-08-14 | OmniDocBench v1.6 96.87, Wild 88.53 ([arXiv 2608.12898](https://arxiv.org/abs/2608.12898)) | unknown (VN handwriting bench includes it) | HF, GGUF, MLX (groxaxo/TeleOCR-oQ4-MLX), ONNX (stefanj0) | MEH (page parser) |
| OvisOCR2 | 0.8B | Apache-2.0 | 2026-07 | OmniDocBench v1.6 96.47; vLLM 0.22.1 only | unknown | vLLM | MEH |
| GLM-OCR | 0.9B | MIT | 2026-01-30 | 95.22; 1.8M dl; vLLM/SGLang/Ollama/MLX | unknown | many | GEM (broadest runtime) |
| MinerU2.5-Pro-2605 | 1.2B | Apache-2.0 weights + toolkit terms | 2026-05 | 95.75; 231k dl | unknown | MinerU toolkit | MEH (PDF-oriented) |
| LightOnOCR-2-1B | 1B | Apache-2.0 | 2026-01 | 239k dl | unknown | HF/vLLM | MEH |
| Surya OCR 2 | 650M VLM now (no longer CNN) | code Apache; weights modified OpenRAIL-M (free <$5M) | 2026-05-14 | olmOCR-bench 83.3; claims 91 langs, 87.2 internal | claimed, unmeasured | vLLM / llama.cpp | MEH (license, heavier) |
| PepperOCR-VL | 4B (Qwen3.5-4B ft) | AGPL-3.0 | 2026-10-04 | card lists `vi` | listed | vLLM | MEH (same size as current VLM, AGPL) |
| DeepSeek-OCR-2 | 3B | Apache-2.0 | 2026-01-27 | OmniDocBench v1.6 90.25 | n/a | HF | NOISE (bigger, lower) |
| HunyuanOCR | 1.1B | custom (no EU/UK/KR rights) | 2025-11 | 89.95 | n/a | | NOISE (license) |
| Florence-2 base/large, EasyOCR, Tesseract 5.5.3, docTR | 0.23-0.77B / classic | MIT / Apache | 2024 / 2026-07 | superseded in every comparison found | Tesseract/EasyOCR have vi | | NOISE (fallback only) |
| olmOCR 2 (7B), Chandra OCR 2 (5.3B), dots.mocr (3B) | 3-7B | mixed | | olmOCR-Bench 82.4 / 85.8 / 83.9 | | | NOISE (too big, not cheap) |

### Findings
- **F1 [GEM] Apple Vision supports Vietnamese, measured.** `supportedRecognitionLanguages` (accurate, rev 3) on macOS 27.0.1 M1 Pro: en-US fr-FR it-IT de-DE es-ES pt-BR zh-Hans zh-Hant yue-* ko-KR ja-JP ru-RU uk-UA th-TH **vi-VT** ar-SA tr-TR id-ID cs-CZ da-DK nl-NL no/nn/nb ms-MY pl-PL ro-RO sv-SE fi-FI hi-IN mr-IN. `.fast` = en fr it de es pt only. Test: 4/6 lines exact, remaining diffs: em-dash -> hyphen, `ể`->`ề` once. 74-119 ms/image warm; first call in a fresh install 66-72 s (model compile), then 294 ms first call in new process. Language correction on. Use `recognitionLanguages=["vi-VT","en-US"]` (order matters per [WWDC21](https://developer.apple.com/videos/play/wwdc2021/10041/)). Code: `->` -> `>`, `Bool`->`Boo1`, `đầuVào`->`dầuVào` at 14pt mono font: weak on code; use `usesLanguageCorrection=false` for code frames (untested).
- **F2 [GEM] PP-OCRv6 tech report ([arXiv 2606.13108](https://arxiv.org/html/2606.13108)).** Latency (s/image, end-to-end det+rec on 200 images): Apple M4 ONNX Runtime medium 5.55 / small 1.29 / tiny 0.35; A100 Paddle 0.29 / 0.25 / 0.13; V100 ONNX 0.67 / 0.53 / 0.29; Xeon OpenVINO 1.40 / 0.59 / 0.20. Mine on M1 Pro (RapidOCR, small, CPU ORT, 1600x420 img): 0.61-0.77 s, consistent with M4 1.29 s on larger images. Apple Vision is ~6-8x faster than that on the same Mac and GPU/ANE-offloaded. Tiny 49 langs, small/medium 50 (46 Latin + zh, zh-TW, en, ja).
- **F3 [GEM, negative] PP-OCRv6 Vietnamese broken.** [PaddleOCR#18254](https://github.com/PaddlePaddle/PaddleOCR/issues/18254) (2026-07-10, closed w/ "future model"): `ppocrv6_dict.txt` (18,708 entries) misses 88 Vietnamese letters, tiny dict misses 92. Reproduced: RapidOCR PP-OCRv6 small turned "Hướng dẫn cài đặt phần mềm" into "Hưng dn cài đt phn mm". Workaround exists: community fine-tuned recs [tieubaoca/pp-ocrv6-medium-rec-vietnamese](https://huggingface.co/tieubaoca/pp-ocrv6-medium-rec-vietnamese) (trained on 102,875 handwriting/scan/uppercase lines, no metrics, not screen text), ANZ-Innovation/PP-OCRv6-Vietnamese-Rec (gated), [duchoa1297/vietocr-onnx](https://github.com/duchoa1297/vietocr-onnx) (VietOCR + PP-OCRv6 det, ONNX). All unvetted on screen text. Earlier same bug class: #15189, #16339 (v5).
- **F4 [GEM] RapidOCR issues to know** ([#743](https://github.com/RapidAI/RapidOCR/issues/743), open, updated 2026-10-02): PP-OCRv6_rec_medium and det_small ONNX carry ~200k float32 denormals; zeroing them gives 11x (rec) / 4x (det) faster CPU inference, identical output (reporter, i7-8550U). Fix not shipped. Small rec has only 241 denormals, tiny 0. [#723](https://github.com/RapidAI/RapidOCR/issues/723) (closed): v6 small rec drops inter-word spaces on Latin languages vs v5 mobile; medium "does not have the issue but too slow on commodity hardware". Implication for Vietnamese/English text: v6-small word spacing is a search-quality risk. [#749](https://github.com/RapidAI/RapidOCR/issues/749) open: word-box scores wrong.
- **F5 [GEM] Doc-OCR leaderboard shift.** OmniDocBench v1.6 (updated 2026-09-11): TeleOCR 96.91, OvisOCR2 96.47, PaddleOCR-VL-1.6 96.34, MinerU2.5-Pro 95.75, GLM-OCR 95.22, DeepSeek-OCR-2 90.25, Qwen3-VL-235B 89.78, Gemini 3 Pro 92.91 ([Docsumo](https://www.docsumo.com/blog/best-ocr-models), [OmniDocBench repo](https://github.com/opendatalab/OmniDocBench)). Caveat: page-parsing benchmark on clean+photographed documents; transfers poorly to screen recordings (no UI chrome, video compression, tiny anti-aliased fonts). Xiaomi-OCR-0 0.8B also hits OCRBench 84.6 / DocVQA 93.1 vs Qwen3.5-0.8B 77.9/88.5 and ~Qwen3.5-2B (85.9/92.4): a 0.8B model matching your 4B baseline on OCRBench would be a fine Tier 2. Self-reported, 9 days old.
- **F6 [MEH] Compute reality.** Doc VLMs are autoregressive: MLX/vLLM latency scales with output tokens (dense slide = hundreds of tokens). I did NOT measure; no M1 Pro / 3060 numbers found for PaddleOCR-VL-1.6, only M5-Max blog numbers of low trust (see NOISE list). PaddleOCR-VL Apple guide exists ([docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html)); mlx-vlm PaddleOCR-VL issues: #1283 (rope_deltas crash, closed 2026-10-06), #1415, #2399 (LoRA, open).
- **F7 [MEH] Vietnamese VLM-OCR evidence is handwriting-only.** [khanhlinh-innomation/vlm-handwriting-ocr](https://github.com/khanhlinh-innomation/vlm-handwriting-ocr): after full SFT on UIT-HWDB lines, PaddleOCR-VL test CER 5.40%, WER 12.77%, 0.374 s/line; zero-shot base numbers not extracted. Another: [ViExam](https://arxiv.org/html/2508.13680v1) 6% CER/9% WER Vietnamese for 5 frontier VLMs.

## 3. (b) Tagging / object & scene labels
- **F8 [NOISE-ish] RAM++/Tag2Text**: xinyu1205/recognize-anything-plus-model weights from 2023-10, 0 recent downloads on that repo, no successor found; GitHub topic search for "image tagging" shows only tiny apps. https://huggingface.co/xinyu1205/recognize-anything-plus-model
- **F9 [GEM, pragmatic] SigLIP2 zero-shot with a fixed vocabulary** (Apache-2.0): base-224 375M, so400m-384 1.14B, released 2025-02, 1.6M/1.0M dl; mlx-community/siglip2-so400m-patch16-384 exists (2026-06). A user project is currently benchmarking SigLIP2 vs TinyCLIP as "lightweight RAM++ alternatives" ([PhotoLibrarian#66](https://github.com/dunclaw/PhotoLibrarian/issues/66), no result yet). One embedding per kf also gives free semantic search/dedup. No accuracy-per-resource data found against tags needed for "screen / slide / terminal / face / whiteboard / diagram" classes; this is a classification job, a 20-label prompt set is enough.
- **F10 [MEH] Open-vocab detection**: OWLv2 (2023-10, 2.07M dl), Grounding-DINO-tiny (2023-09), YOLOE (2025-03, AGPL). All stale; none new since Oct 1. LFM2.5-VL-450M adds bbox prediction (below).
- Anime taggers (pixai-tagger-v1.0 2026-09-15, wd-eva02-tagger) NOISE for this use.

## 4. (c) Small captioners / dense captioning
| Model | Params | License | Date | Signal | Verdict |
|---|---|---|---|---|---|
| LFM2.5-VL-450M | 450M (SigLIP2-86M + LFM2.5-350M) | LFM1.0 custom | 2026-04-08 | OCRBench 684 (/1000), POPE 86.93, MMBench-dev 60.91, CountBench 73.31, InfoVQA 43.0; Jetson Orin 233-242 ms/image Q4_0, Ryzen AI Max 637-944 ms; MLX 4/5/6/8bit + ONNX + GGUF official; "not well-suited for fine-grained OCR" per card. Languages: en ar zh fr de ja ko pt es (no vi) | MEH (only new sub-500M; pair with OCR) |
| LFM2.5-VL-1.6B | 1.6B | LFM1.0 | 2026-01-05 | between 450M and 3B; not benchmarked here | MEH |
| SmolVLM2-256M/500M | 0.26/0.51B | Apache-2.0 | 2025-02 | 0.8/1.2GB VRAM; stale, below Qwen3.5-0.8B per prior notes | NOISE |
| Florence-2-base/large | 0.23/0.77B | MIT | 2024-06 | 389k dl; OCR + caption + OD tasks; still the caption head inside OmniParser v2 | MEH (stale, English only) |
| FastVLM-0.5B | 0.76B | apple-amlr (research) | 2025-08 | license blocks | NOISE |
| Moondream 2 / 3.1 | 1.9B / 9B-A2B | apache / custom | 2024 / 2026-06 | prior notes: skip | NOISE |
Sources: https://huggingface.co/LiquidAI/LFM2.5-VL-450M ; https://www.marktechpost.com/2026/04/11/liquid-ai-releases-lfm2-5-vl-450m-a-450m-parameter-vision-language-model-with-bounding-box-prediction-multilingual-support-and-sub-250ms-edge-inference/
- **F11**: no independent head-to-head of sub-1B captioners vs Qwen3.5-0.8B/2B found. Qwen3.5-0.8B (Apache-2.0, OCRBench 74.5/79.1) is the better-documented small option and shares the stack with your 4B (prior notes). Xiaomi-OCR-0 is literally Qwen3.5-0.8B-based, so a Qwen3.5-0.8B captioner + Xiaomi-OCR-0 OCR share a runtime.

## 5. (d) Screen / UI parsers
- **F12 [GEM, reference] OmniParser v2** (MIT; icon-detect YOLOv8 + icon-caption Florence-2-base; HF last modified 2026-09-23, repo 25.5k stars, last commit 2026-07-20 adds Hugging Face-backed YOLOv9-E detector). Output = bboxes + icon captions + OCR; designed for GUI agents (clickable regions), not for "what was on screen" summaries. Adds a Florence-2 pass per icon -> heavy relative to benefit for keyframes. No issues about slowness/mac surfaced in a repo search. https://github.com/microsoft/OmniParser ; https://huggingface.co/microsoft/OmniParser-v2.0
- **F13 [MEH] Holo3.1-0.8B/4B/9B** (H Company, Apache-2.0, 2026-05-29): Qwen3.5-based computer-use / UI-grounding agents, not parsers. Useful only if you need "where is the button"; irrelevant to text-for-search. https://huggingface.co/Hcompany/Holo-3.1-0.8B
- **F14 [MEH] ScreenParse** ([site](https://saidgurbuz.github.io/screenparse/)): a YOLO detector trained on its own dataset vs OmniParser v2 in a fixed Qwen3-VL-8B Set-of-Mark pipeline; only snippet read. Not verified.
- Others seen, unvetted: Interfaze-1-lite (2026-10-03, 52 likes, custom license, multi-task incl. OCR/GUI grounding/detection; "mixture-of-architectures"), GroundAnything (2026-09-30, diffusion-LM grounding, Kimi-K3 upstream license). NOISE until someone benchmarks them.
- Take: for coding-agent keyframes (IDE/terminal/browser), OCR text + bboxes from Apple Vision/PP-OCRv6 already gives most of "UI parsing" value; adding a UI parser = surface for little search gain.

## 6. Forum signal (elite, as requested)
- HN: [Show HN: Open-weight OCR got so cheap (item 49041317)](https://news.ycombinator.com/item?id=49041317) about PaddleOCR-VL-1.6 serving cost; page body mostly not retrievable. [item 45607613](https://news.ycombinator.com/item?id=45607613) PaddleOCR-VL launch (Oct 2025, "Baidu claims SOTA on their own benchmark"). Older PaddleOCR threads (2024) say it beats Tesseract, misreads `l` vs `1`. site:news.ycombinator.com search otherwise stale; "PP-OCRv6" HN item 48563911 returns "No such item".
- r/LocalLLaMA [PP-OCRv6 is officially released](https://www.reddit.com/r/LocalLLaMA/comments/1u3p2at/ppocrv6_is_officially_released/) (~Jun 2026): u/StanPlayZ804 swapped v5 mobile for v6 tiny det + small rec in a pipeline: "quality generally very good", rec slower than v5 mobile rec, det misses end-quote marks. Commenters note it is not a VLM; complaint that Baidu compared against Gemini/GPT/Qwen rather than its own PaddleOCR-VL-1.6; handwriting test failed (unfair, size). One commenter on supply-chain risk of Paddle binaries.
- r/computervision "Best free/open-source AI model for understanding screenshots?" (~2026-09-09): answers are Qwen3.8-27B, Mage-VL, Gemma 4 only; nobody recommends specialist OCR+parser. Signal: community defaults to VLMs.
- Reddit r/LocalLLaMA OvisOCR2 post (827 medical scans) exists: https://www.reddit.com/r/LocalLLaMA/comments/1ux3nps/ (not read).

## 7. NOISE list
- contracollective.com "Surya vs docTR vs PaddleOCR on M5 Max (2026)": benchmarks PP-OCRv4 and Qwen2.5-VL 7B, looks like generated content; do not cite numbers.
- Roboflow / Docsumo / Instavar leaderboards: useful for lists but vendor blogs; Docsumo promotes its own product. Scores are OmniDocBench (documents).
- DeepSeek-OCR-2, HunyuanOCR, olmOCR 2, Chandra 2, dots.mocr: size/license wrong for "cheap".
- RAM++ / Tag2Text as a 2026 pick.

## 8. Unresolved / next measurement (cheap, would settle the ranking)
1. Measure on real framesift keyframes (screen recording, 1080p, compressed): Apple Vision vs PP-OCRv6 small vs PaddleOCR-VL-1.6 MLX vs Xiaomi-OCR-0 on CER + latency, with Vietnamese slides and English code. My test was 30pt synthetic.
2. Cold-start: is the 66 s Apple Vision first-call compile once per OS install, per language set, or per app signature? Observed once (and again in a different binary at 72 s), then 294 ms. Test with a fresh signed binary after reboot.
3. Apple Vision on code at 12-14px: needs `usesLanguageCorrection=false` + upscale test.
4. RTX 3060 Vietnamese: no vetted option found. Candidates to test: PaddleOCR-VL-1.6 (vLLM), Xiaomi-OCR-0, EasyOCR `vi`, tieubaoca VN rec + PP-OCRv6 det. Surya 2 license may block.
5. Xiaomi-OCR-0 on MLX/GGUF? Not verified (TeleOCR has MLX quants; Xiaomi has "-Ninfer" and GGUF re-uploads).
6. RapidOCR denormal fix (#743) not shipped; zeroing denormals yourself (script in issue) is a 10-line one-off for medium/det_small.
7. No sub-1B captioner vs Qwen3.5-0.8B comparison exists publicly. LFM2.5-VL license (LFM1.0 revenue cap) unchecked.
8. HN/Reddit stat counts unavailable (scraper). idp-leaderboard.org, OCRBench v2 not fetched.

## Sources
- https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec ; https://arxiv.org/html/2606.13108v1 ; https://github.com/PaddlePaddle/PaddleOCR/issues/18254
- https://github.com/RapidAI/RapidOCR/issues/743 ; /issues/723 ; /issues/749
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6 ; https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0 ; https://arxiv.org/abs/2608.12898 ; https://huggingface.co/XingChen-AGI/TeleOCR
- https://www.docsumo.com/blog/best-ocr-models ; https://github.com/opendatalab/OmniDocBench ; https://roboflow.com/blog/best-open-source-ocr-models
- https://huggingface.co/datalab-to/surya-ocr-2 ; https://huggingface.co/sionic-ai/PepperOCR-VL ; https://huggingface.co/Hcompany/Holo-3.1-0.8B ; https://huggingface.co/interfaze-ai/interfaze-1-lite
- https://developer.apple.com/documentation/vision/vnrecognizetextrequest ; https://developer.apple.com/videos/play/wwdc2021/10041/ (plus local runs 2026-10-08)
- https://github.com/khanhlinh-innomation/vlm-handwriting-ocr ; https://huggingface.co/tieubaoca/pp-ocrv6-medium-rec-vietnamese ; https://github.com/duchoa1297/vietocr-onnx
- https://huggingface.co/LiquidAI/LFM2.5-VL-450M ; https://www.marktechpost.com/2026/04/11/liquid-ai-releases-lfm2-5-vl-450m-a-450m-parameter-vision-language-model-with-bounding-box-prediction-multilingual-support-and-sub-250ms-edge-inference/
- https://github.com/microsoft/OmniParser ; https://huggingface.co/microsoft/OmniParser-v2.0 ; https://github.com/dunclaw/PhotoLibrarian/issues/66
- https://www.reddit.com/r/LocalLLaMA/comments/1u3p2at/ppocrv6_is_officially_released/ ; https://www.reddit.com/r/computervision/comments/1wadsoh/best_freeopensource_ai_model_for_understanding/ ; https://news.ycombinator.com/item?id=49041317 ; https://news.ycombinator.com/item?id=45607613
- https://github.com/Blaizzy/mlx-vlm/issues/1283 ; https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html
