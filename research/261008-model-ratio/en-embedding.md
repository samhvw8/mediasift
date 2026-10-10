# en-embedding — text+image embedding per resource (gathered 2026-10-08, EN)

Saved by the lead from the gatherer's returned findings (condensed; full table in the session log).

## Verdict
1. Nothing new since 2026-10-01 beats EmbeddingGemma 2 (EG2) on accuracy per resource: no sub-1B, Apache-2.0, multilingual text+image model is better.
2. Accuracy upgrade: **Qwen3-VL-Embedding-2B** (2.1B, Apache-2.0, MLX mxfp8 port) — MMEB v2 73.2 (Image 75.0, Video 61.9, VisDoc 79.2) vs EG2 59.01 (57.28 / 50.67 / 67.84) [card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).
3. Speed alternative: **SigLIP2** So400m (COCO T→I 55.8, XM3600 48.4) or B/16 (52.1 / 40.3), Apache-2.0, ONNX/MLX ports [paper](https://arxiv.org/pdf/2502.14786); CLIP-class scores do not map onto MMEB.
4. No public Vietnamese image-retrieval number for EG2 or Qwen3-VL-Embedding; only SEA-CLIP-Tiny, MobileCLIP2 (vi R@1 3.3), MetaCLIP2 (vi 57.1, NC) [SEA-CLIP](https://arxiv.org/html/2609.30739v1). Needs our own bench.
5. EG2 scores are vendor-reported: MTEB maintainers have not reproduced them ([mteb PR 5588](https://github.com/embeddings-benchmark/mteb/pull/5588)).

## Accuracy per parameter
| model | loaded params | MMEB v2 Image | points per B params |
|---|---|---|---|
| EG2 text+vision | 0.44B | 57.28 | 130 |
| Qwen3-VL-Embedding-2B | 2.1B | 75.0 | 36 |
| WeMM-Embedding-2B | 2.1B (2.7B actual) | 79.6 (self-reported, custom code) | 38 |

## Other candidates
| model | note | tag |
|---|---|---|
| Ovis-Embedding-VL-2B | MMEB 77.46 in paper, weights "will release" [arXiv 2609.25165](https://arxiv.org/pdf/2609.25165) | MEH |
| jina-embeddings-v5-omni nano/small | CC-BY-NC; MIEB-lite 44.2 / 55.0 | NOISE / MEH |
| Omni-Embed-Mini 0.9B (Oct 1) | COCO T2I R@1 29.4 | NOISE |
| PE-Core B/L | Apache-2.0, COCO T→I 50.9 / 57.1, 32-token text, no multilingual claim | MEH |
| MobileCLIP2, MetaCLIP2, jina-clip-v2 | non-commercial | NOISE |
| ViCLIP-OT / ViSigLIP-OT | Vietnamese-only | MEH |

## Signals
- HN on EG2: one photo-search builder calls it a drop-in upgrade over EmbeddingGemma 1 / HQ-CLIP ("better quality, less RAM, faster"); open question "how vs SigLIP2" unanswered [HN](https://news.ycombinator.com/item?id=49980487). EG2 on M3 Pro ~4 img/s (consistent with our 6-8 on M1 Pro at 70 tokens).
- No published images/s for Qwen3-VL-Embedding-2B on MPS or a 3060.

## Open
Local bench: EG2 vs Qwen3-VL-Embedding-2B vs SigLIP2 So400m on our frames with English and Vietnamese queries; EG2 at higher vision budgets (140/280).
