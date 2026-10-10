# zh-frame-extract — OCR / tagging / small captioners, ZH sources (gathered 2026-10-08)

Saved by the lead from the gatherer's returned findings (condensed). Vendor numbers unless marked.

## Bottom line
1. **Screen/slide text: PP-OCRv6 det+rec, not a VLM.** tiny/small/medium = 1.5M / 7.7M / 34.5M params; rec "screen" category 71.2 / 79.7 / 82.5 vs v5_mobile 57.6 and Qwen3-VL-235B 73.8 (Baidu in-house set). End-to-end s/image on **Apple M4 ONNX Runtime: 0.35 / 1.29 / 5.55** [docs](https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html), [arXiv 2606.13108](https://arxiv.org/html/2606.13108v1). GEM.
2. **Vietnamese is broken in v6**: dictionaries lack all 88 composed tone-marked vowels (U+1EA0–1EF9); tiny also lacks ơ/ư; maintainers confirmed, closed 2026-07-23 without fix [#18254](https://github.com/PaddlePaddle/PaddleOCR/issues/18254).
3. **Vietnamese text → PaddleOCR-VL-1.6** (0.96B, Apache-2.0, 109 langs, OmniDocBench v1.6 96.33; mlx-vlm, llama.cpp GGUF, vLLM) on the subset of frames with low-confidence Latin text. No independent vi accuracy number. Mac full page 16-22 s (llama-server / mlx-vlm) [cnblogs](https://www.cnblogs.com/rolandhe/p/19911302) — too slow for every frame.
4. **Tagging**: RAM/RAM++ stale (last push 2025-02), English open-set tags, no successor. Derive tags from OCR text or a small VLM.
5. **Captions**: no specialised small captioner; floor is Qwen3.5-0.8B / MiniCPM-V 4.6 (1.3B), ~1/5 of the current 4B.
6. Nothing new since 2026-10-01 (big releases were June-July 2026).

## Ranked by accuracy per resource
| model | params | license | numbers | notes | tag |
|---|---|---|---|---|---|
| PP-OCRv6 tiny/small/medium | 1.5M-34.5M | Apache-2.0 | det 80.6-86.2, rec 73.5-83.2 | ONNX/OpenVINO/RapidOCR/WASM; **ONNX denormal trap**: zero ~200k denormal weights or CPU is 4-11x slower ([RapidOCR #743](https://github.com/RapidAI/RapidOCR/issues/743)) | GEM (non-vi) |
| PaddleOCR-VL-1.6 | 0.96B | Apache-2.0 | OmniDocBench v1.6 96.33 | weak on degraded scans; vi fine-tune exists (unproven) | GEM (vi, subset) |
| MonkeyOCRv2-S/B | 0.6B/0.7B | Apache-2.0 | MDPBench 82.5/83.3 > dots.mocr 3B | no speed numbers; vi unconfirmed | MEH |
| HunyuanOCR-1.5 | 1B | Tencent community license | v1.6 94.74 (unverified) | lists video-subtitle and screenshot tasks | MEH (license) |
| GLM-OCR | 0.9B | MIT | v1.6 95.22 | 8 langs, **no vi** | MEH |
| MinerU2.5, dots.ocr, DeepSeek-OCR-2 | 1.2-3.4B | various | — | page pipelines, bigger | NOISE for frames |
| Qwen3.5-0.8B / MiniCPM-V 4.6 | 0.8B / 1.3B | — | anecdotal: screenshots "basically fine" | caption + OCR fallback | MEH |

## Signals
V2EX: PaddleOCR / PaddleOCR-VL consensus for screenshot OCR; VL OCR beats classic OCR on camera-shot screens; one complaint that PaddleOCR-VL is worse than Qwen3-VL on receipts. linux.do: PaddleOCR-VL Paddle path has no ARM CPU support → use mlx-vlm/llama.cpp on Mac. (linux.do pages carry a hidden instruction aimed at AI readers; ignored.)

## Unresolved
Vietnamese accuracy of PaddleOCR-VL-1.6 and of the older v5 `vi`/`latin` recognizers; ms/frame on M1 Pro and RTX 3060.
