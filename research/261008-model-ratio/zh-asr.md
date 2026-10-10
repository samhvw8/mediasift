# zh-asr — Chinese-lab multilingual ASR (gathered 2026-10-08, ZH sources)

Saved by the lead from the gatherer's returned findings (the gatherer could not write files).

## Verdict
Only real contender vs whisper-large-v3-turbo for vi+en: **Qwen3-ASR 0.6B / 1.7B** (Apache-2.0). Gap: no Vietnamese timestamps from the official aligner. Nothing new since 2026-10-01 found.

| model | size | vi? | vi accuracy | en accuracy | timestamps | Mac / 3060 | tag |
|---|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | 1.7B | yes (30 core langs) | FLEURS-vi 4.91 vs turbo 8.10 (Whisper Notes, M5 Air, ~150 clips) [whispernotes](https://whispernotes.app/zh/blog/qwen3-asr-vs-whisper); report aggregate Fleurs 4.90 vs Whisper-v3 5.27 [card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | LS 1.63/3.38 | ForcedAligner-0.6B: 11 langs, **no vi**; zero-duration spans bug [#197](https://github.com/QwenLM/Qwen3-ASR/issues/197) | mlx-qwen3-asr: RTF 0.018-0.029 on M4 Pro (35-55x RT), 1.2 GB (0.6B) / 3.4 GB (1.7B), own MLX aligner [repo](https://github.com/moona3k/mlx-qwen3-asr); 9.6x RT for 1.7B on M5 Air | GEM |
| Qwen3-ASR-0.6B | 0.6B | yes | aggregate Fleurs 7.57 (worse than Whisper-v3) | LS 2.11/4.55 | as above | 4-bit 517 MB | MEH (test vi yourself) |
| Dolphin base/small | 140M / 372M | yes (40 Eastern langs) | small 16.3 / 13.8 vs Whisper-v3 14.8 / 8.9 [arXiv 2503.20212](https://arxiv.org/abs/2503.20212) | — | word ts added 2026-05-09 | no ONNX/MLX | MEH |
| Fun-ASR-MLT-Nano | 0.8B | yes (31) | open Nano: Fleurs 10.03 (Qwen table) | — | VAD-segment only | — | MEH |
| SenseVoice-Small | small | **no** | — | CV 14.71 vs 9.39 | yes | 52-162x RT on M4/M5 | NOISE for vi |
| GLM-ASR-Nano-2512 | 1.5B | unconfirmed | Fleurs 16.08 (Qwen table) | — | — | — | NOISE |
| FireRedASR2S | — | no (zh/en) | — | — | yes | FireRedVAD (100+ langs) useful for segmenting | NOISE for ASR |

Unresolved: Qwen3-ASR vi WER on our audio; a vi CTC aligner to pair with it; 3060 throughput; GLM-ASR language list. Forums (V2EX, linux.do, cnblogs, SegmentFault): no vi-specific numbers; CSDN posts skipped as marketing.
