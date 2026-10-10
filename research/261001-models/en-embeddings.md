# en-embeddings — text + image embedding models (gathered 2026-10-01)

Saved by the lead from the gatherer's returned report (the gatherer could not write files). Condensed; sources kept.

## Picks

| Job | Pick | Repo | Why |
|---|---|---|---|
| Text, quality (GPU/MPS) | EmbeddingGemma-300m | `google/embeddinggemma-300m` | MMTEB 61.15 vs 48.8 for the current MiniLM; 768d, MRL to 128; in fastembed 0.8.1; ONNX/GGUF/MLX. Gemma terms (not OSI). |
| Text, CPU default | multilingual-e5-small / base | `intfloat/multilingual-e5-small`, `-base` (ONNX `Xenova/...`) | MMTEB 55.5 / 57.0, VN-MTEB 60.66 / 62.42; same size as baseline; MIT. Not built into fastembed. |
| Text, Vietnamese-heavy | gte-multilingual-base | `Alibaba-NLP/gte-multilingual-base` (ONNX `onnx-community/...`) | VN-MTEB 65.22 (best per FLOP), MMTEB 58.24, 305M, Apache-2.0. |
| Image, semantic dedupe + text search | SigLIP2 base | `google/siglip2-base-patch16-224` | Apache-2.0, in fastembed (image + text towers), IN-1k 78.2, COCO T→I 52.1. |
| Image, second signal | DINOv2 base/small | `facebook/dinov2-base` | Apache-2.0; strong at "same instance" where SigLIP2 is weak. |

## Key numbers

- Baseline `paraphrase-multilingual-MiniLM-L12-v2` is the weakest multilingual option: MMTEB 48.8. [GEM] https://arxiv.org/html/2502.13595
- Qwen3-Embedding-0.6B MMTEB 64.33 but slowest on CPU (3.2 sent/s) [GEM] https://huggingface.co/Qwen/Qwen3-Embedding-0.6B, https://medium.com/@janschachtschabel/multilingual-embedding-models-in-2026-what-actually-works-on-cpu-d4bf3fe3d335
- VN-MTEB: gte-multilingual-base 65.22, bge-m3 64.90, e5-large-instruct 67.99, Vietnamese_Embedding (bge-m3 fine-tune) 63.34 — Vietnamese-specific fine-tunes did not beat their parent. [GEM] https://arxiv.org/html/2507.21500v1
- Instance retrieval (Oxford-Hard, base size): DINOv3-B 58.5, DINOv2-B 51.0, SigLIP2-B 20.2, PE-Core-B 20.2 — semantic models collapse meaning, not layout. [GEM] https://arxiv.org/html/2508.10104
- fastembed issue #689: some Qdrant ONNX builds (including the MiniLM `-onnx-Q`) truncate silently at 128 tokens. [GEM] https://github.com/qdrant/fastembed/issues/689
- GPU indexing (A6000, English): bge-m3 433 chunks/s, EmbeddingGemma 315, Qwen3-0.6B 170; VRAM ≤ 2.5 GiB. [MEH] https://atomic.chat/blog/guides/best-embedding-models-for-rag

## Licence traps

MobileCLIP2 (Apple research-only), jina-embeddings-v3 / jina-clip-v2 (CC-BY-NC), jina-v4 (Qwen research), DINOv3 (custom Meta), EmbeddingGemma (Gemma terms). Clean: e5, bge-m3, gte-multilingual, Qwen3-Embedding, SigLIP2, PE-Core, DINOv2, CLIP B/32.

## Dedupe design note (inference)

Drop a keyframe only if SigLIP2 cosine to the last kept frame is above a threshold AND a cheap structural check agrees (DINOv2-small cosine or the existing pixel distance). Calibrate on own videos; no published threshold.

## Unresolved

1. No Vietnamese scores for EmbeddingGemma / Qwen3-Embedding.
2. No benchmark of embeddings for transcript topic segmentation — needs a small Pk/WindowDiff eval.
3. Whether framesift is commercial/redistributed (decides EmbeddingGemma, MobileCLIP2, DINOv3).
4. fastembed custom-model loading for e5 / gte; check 128-token truncation.
5. CPU/MPS throughput not published; bench locally.
