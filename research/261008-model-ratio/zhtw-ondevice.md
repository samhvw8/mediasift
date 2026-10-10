# zhtw-ondevice — on-device practice, Traditional Chinese sources (gathered 2026-10-08)

Saved by the lead from the gatherer's returned findings (condensed). Coverage thin: search engines degraded; ~40 NOISE results.

## ASR
- Whisper turbo on **RTX 3060 12GB**: 7-min audio in 40-50 s (~10x RT) [jt-live-whisper](https://github.com/jasoncheng7115/jt-live-whisper) — GEM, the only 3060 number found.
- Qwen3-ASR-0.6B in the same tool: real Chinese meeting CER 28.78 → 15.75 (vs likely turbo), 37-min meeting in ~1.5 min on M5, ~3.2 GB RAM (Mac) / ~5 GB VRAM — GEM.
- mlx-whisper turbo on M5 Pro: RTF ~0.29 on a 4.5 s clip [iThome](https://ithelp.ithome.com.tw/articles/10402637) — MEH.
- Breeze-ASR-25 (MediaTek, Taiwan Mandarin): newsroom used it as a "second opinion", Whisper-1 best overall [BI Taiwan](https://www.businessinsider.tw/article/4360); with a glossary CER 2.34% on one 55 s clip; Qwen3-1.7B 68% there from over-transcription [taiwan-asr-toolkit](https://github.com/thc1006/taiwan-asr-toolkit) — MEH (Taiwan-specific).

## OCR
- **PaddleOCR-VL 0.9B on RTX 3060 12GB**: layout model on CPU + VLM on GPU via vLLM leaves ≥8 GB free; table TEDS PaddleOCR-VL 0.92/0.95 > MinerU2.5 0.90/0.95 > dots.ocr 0.82/0.84 [civilgroup](https://civilgroup.org/2026/07/21/) (snippet only, page 403) — GEM, unverified.
- PaddleOCR-VL-1.6 (2026-05-28): OmniDocBench v1.6 96.3%, 109 languages (vendor) [docs](https://www.paddleocr.ai/latest/en/index.html).
- **PP-OCRv6**: medium 83.2% weighted accuracy (+5.1 over v5); a 1.1M-param tiny variant claimed to beat 4 of 5 VLMs (vendor) [docs](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) — MEH, worth testing.
- Apple Vision OCR: only hearsay that it beats Windows OCR / Tesseract — no numbers.

## Embeddings
- EmbeddingGemma 2: zh-TW coverage repeats Google's numbers; no independent zh-TW, Mac or 3060 measurements yet. MLX Swift 8-bit port: [EmbeddingGemma2Swift](https://github.com/Obscyra-app/EmbeddingGemma2Swift). Qdrant ran text-only quantization tests [blog](https://qdrant.tech/blog/embeddinggemma-2/).
- SigLIP/CLIP: no practitioner numbers in zh-TW sources.

## Gaps
No Vietnamese/SEA measurements from this community; no independent Apple Vision accuracy numbers.
