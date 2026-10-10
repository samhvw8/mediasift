# ZH gather: Chinese-lab multimodal embeddings (text->image/video-frame), as of 2026-10-08

Lang: zh-Hans. Baseline: EmbeddingGemma 2 (440M text+vision), M1 Pro 6-8 img/s, RTX 3060.
Scope: embeddings only. Confidence tags: [V]=verified on primary page, [S]=secondary/vendor-reported, [E]=my inference.

## TL;DR
- Only real contender found: **WeMM-Embedding-2B** (Tencent WeChat, 2026-08-25). GEM on accuracy, MEH on speed/size vs baseline. Everything else is older, API-only, non-commercial, or paper-only.
- Accuracy gap is large on paper (MMEB-v2 77.9 vs ~59), but cost is ~6x params, ~8GB resident, <=2 img/s on Apple Silicon (measured by others on M4/M5 Max) [E for M1 Pro]. Accuracy-per-resource likely favours EmbeddingGemma 2 for bulk frame indexing on M1; WeMM wins if recall quality matters (Chinese queries, text-in-image).
- Since 2026-10-01: only EmbeddingGemma 2 itself (2026-10-06). No new Chinese-lab multimodal embedding found in Oct 1-8.
- Caveat on baseline score: cldnavi lists EG2 MMEB-v2 image 57.28 / visdoc 67.84 / video 50.67 for full 740M config (768d); I could not verify the 59.01 / 440M figure. Not comparable apples-to-apples with WeMM overall.

## Ranked table (accuracy per resource, text->image/video-frame)

| # | Model | Org / date | Params (actual) | License | MMEB-v2 (all / img / vid / visdoc) | Speed / mem | Ports | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **WeMM-Embedding-2B** | Tencent WeChat, 2026-08-25 | 2.72B BF16, 5.44GB, 2048d, MRL 64-2048 | Code Apache-2.0 (GitHub LICENSE); HF tag `license:other`, LICENSE file says Apache terms apply to "code" only. Base Qwen3.5-2B. Weights license ambiguous -> check before commercial | 77.9 / 79.6 / 70.8 / 80.7 | M5 Max MPS fp16: ~127ms/img window @360p, ~900ms per video doc. M4 Max 64G: 2180 imgs in ~20 min (~1.8 img/s, <=1024px, batch 2), 8.3GB resident | MLX 4-bit (Swift) community, ONNX community, vLLM, SGLang, sentence-transformers. No GGUF found | **GEM** |
| 2 | WeMM-Embedding-4B | same | ~5B (10.3GB) | same | 79.2 / 80.8 / 72.1 / 82.0 | 14GB resident, +80% encode time vs 2B; no measurable gain vs 2B on 120 Chinese queries | same | MEH (cost not worth) |
| 3 | WeMM-Embedding-9B | same | 18.8GB BF16 | same | 80.6 / 81.9 / 74.3 / 83.3 | too big for 3060 12GB fp16 [E]; ~21GB resident est. | same | NOISE for us |
| 4 | Qwen3-VL-Embedding-2B | Alibaba Qwen, 2026-01-07 | 2.13B BF16 | Apache-2.0 [V] | 73.2 / 75.0 / 61.9 / 79.2 | no independent speed data found | vLLM/transformers; no MLX/GGUF found | **GEM-lite**: safest license, 4.7 pts below WeMM-2B |
| 5 | Qwen3-VL-Embedding-8B | same | 8B | Apache-2.0 | 77.8 / 80.1 / 67.1 / 82.4 | heavy | - | MEH |
| 6 | Jina-embeddings-v5-omni-nano / small | Jina AI (now Elastic), 2026-04-01, upd 2026-08-27 | nano ~0.95-1.04B (768d); small 1.57B (1024d) | **CC BY-NC 4.0** [V] (commercial via sales) | MIEB-Lite/MMEB-V in arXiv 2605.08384; no numbers retrieved | smallest of the "real" MLLM-class models; frozen aligned towers, 0.35% trained | transformers, ST, vLLM | MEH: license blocks; accuracy unknown |
| 7 | DME (Douyin Multimodal Embedding) 2B/9B | ByteDance Douyin, arXiv 2608.02148 (2026-08-03) | 2B / 9B | none stated | 74.8 / 78.4 | - | **no weights found** | NOISE (paper only) |
| 8 | MoEMB-A3B | arXiv 2609.08663 (2026-09-08) | ~3.1B active | none | 70.4 | claims >4x fewer active params than peers, token pruning -50% compute | **no weights found** | NOISE (watch) |
| 9 | Ops-MM-embedding-v1-2B / 7B | Alibaba Cloud OpenSearch, 2025-07 | 2.21B (2B) | Apache-2.0 [V] | 63.44 / 69.03 / 47.56 / 66.96 | - | - | MEH: superseded |
| 10 | Seed-1.6-Embedding (doubao-embedding-vision) | ByteDance, 2025-06-24 | undisclosed | API only (Volcano Ark) | MMEB-v2 image 77.78 (vendor) | n/a | none | NOISE: no local run, API metering |
| 11 | GME-Qwen2-VL-2B / 7B | Alibaba Tongyi, 2024-12 | 2B / 7B | check (HF) | older; below Qwen3-VL | 1024 visual-token cap | ModelScope | MEH: superseded; one CSDN source says training data English-only (low-tier source) |
| 12 | Chinese-CLIP, M2-Encoder (Ant) | OFA-Sys / Ant | CLIP-size to 10B | open | no MMEB | cheap, fast | ONNX-able | MEH: only if tiny CLIP-class + Chinese queries needed; no 2026 refresh found |
| - | BGE-VL (BAAI) | - | - | - | - | - | - | No 2026 release found in ZH searches; skip |

## Practitioner signal (ranked by trust)

1. **ryangu00/apple-silicon-wemm-embedding** (GitHub, updated 2026-10-04): real ZH text->image eval on a 2,180-image personal KB, 120 Chinese queries. jina-clip-v2 hit@1 0.217 -> WeMM-2B@1024d 0.592 (recall@5 0.825, MRR 0.698). WeMM-4B 0.575/0.842 = no detectable gain (n=120, SE ~4.5pt). Biggest win: text-inside-image 0.167 -> 0.833. Hazards: image processor barely downscales; original-res batch 8 on 64GB M4 Max hit 22GB + 18GB swap. Fix = cap 1024px, batch 2. Author's own caveat: single-user, n=120. https://github.com/ryangu00/apple-silicon-wemm-embedding [V]
2. **Tencent/WeMM-Embedding issue #6** (closed 2026-09-10): user on M5 Max MPS fp16 measured video-doc vs image-doc for text->moment search on Big Buck Bunny. Image doc: ~127ms/window, 1/3 queries correct. Video doc: ~900ms, 3/3 correct. Cosine between the two encodings of same content only 0.73 mean. Maintainers: encode directly, omit "Represent this image." instruction, use video for video. Relevant to framesift: single-keyframe indexing is 7x cheaper but loses recall; embeddings from image and video paths are NOT interchangeable. https://github.com/Tencent/WeMM-Embedding/issues/6 [V]
3. **知乎 question 2076324817746256714** (10.9k views): skeptic "Freezy": new embedding scores rise but real use shows little gain, early Qwen embedding adopters "叫苦连天", "效果不见得比BGE好上多少"; MMEB-v2 pools tiny, video only 8 uniform frames, v3 has bias (some videos synthesized from images), crude weighting. Fan answer (恋猫) cites Gemini Embedding 2 comparison (WeMM-2B avg 79.8 vs Gemini Emb 2 79.5; text->image 81.5 vs 83.8, text->video 70.4 vs 63.1). https://www.zhihu.com/question/2076324817746256714 [V]
4. **V2EX t/1240341** (2026-09-08, 0 replies): hype post "当世最强嵌入模型", no data. https://www.v2ex.com/t/1240341 [V] -> NOISE
5. **V2EX t/1236302** (2026-08-21): Magpie local launcher uses **SigLIP 2 + ONNX** for pure-local text->image search. Evidence SigLIP2 remains the default light local choice in the ZH dev crowd. https://www.v2ex.com/t/1236302 [V]
6. linux.do: site down for scraper (403 / timeout). One query hit earlier (topic 2891724, embedding free-tier request, 2026-09-11, off-topic). Gap, not absence.
7. Zhihu "含金量" analysis (p/2076056590386468118): 2B 77.9 vs Qwen3-VL-8B 77.8 is 0.1pt, "not significant"; 14 online A/B tests reported but unreported per-test metrics; audio unsupported (v3 11 audio tasks scored 0 inflate nothing but cap V3 score). https://zhuanlan.zhihu.com/p/2076056590386468118 [V]

No ZH practitioner head-to-head of WeMM / Qwen3-VL-Embedding vs SigLIP2 or EmbeddingGemma 2 found. EmbeddingGemma 2 is 2 days old; only launch coverage exists (yingzheng, cldnavi, both low-tier summaries).

## Known issues / risks (GitHub)
- WeMM repo: 1,727 stars, last push 2026-09-20; open issue #13 "vLLM inference example" (no offline example). Issues #6 (video/image gap), #7 "评测代码不完整" (closed), #9 reproducibility of PeerQA/DeepPlanning scope (closed). https://github.com/Tencent/WeMM-Embedding
- Pins: transformers==5.2.0 (upstream warns newer preprocess differs; ryangu00 measured identical text/image vectors on 5.15.0, video unverified), qwen-vl-utils 0.0.14 without decord on macOS ARM, torchvision needed. [V]
- GitHub license field = NOASSERTION; HF = `other`. Treat as Apache-2.0 code + unclear weights; Qwen3.5 base is Apache-2.0 [E].
- Audio unsupported. Not relevant for us.
- Community ports exist but are tiny/unvetted: metaswang/voxstudio-wemm-embedding-mlx-4bit (Swift MLX 4-bit, 2026-08-30), Choise-ieee/WeMM-Embedding-ONNX (peak ~8GB per branch, sim deviation ~0.02 vs fp32), weifding/wemm-medical-embedding (pruned 5.44->3.31GB, medical domain, lossless R@1 claim, unverified).
- Stat fix: "2B" is 2.72B in HF safetensors (includes vision tower); vs EG2 440M that is ~6x params.

## Fit for framesift (E = my inference)
- M1 Pro: WeMM-2B BF16 5.4GB weights, 8.3GB resident (MPS). M4/M5 Max throughput ~2-8 img/s depending on resolution; M1 Pro is slower [E: probably 1-2 img/s at 1024px, maybe 3-5 at 360p]. Baseline 6-8 img/s => expect 3-6x slower index time. Run a 200-frame bench before committing.
- RTX 3060 12GB: 2B fits in bf16/fp16 with room for batch [E]; vLLM `--runner pooling` supported.
- Cost lever: MRL to 256d keeps 98.7% of full-dim MMEB-v2 perf (vendor-reported) => smaller index; encode cost unchanged.
- Cost lever: downscale frames to <=360-512px before encode (issue #6, ryangu00 gate 1).
- Decision rule I would use: if existing EG2 R@k on framesift's frame set is acceptable, stay. If Chinese queries or text-in-frame (slides, subtitles, UI) matter, WeMM-2B is the only candidate worth benchmarking; Qwen3-VL-Embedding-2B is the clean-license fallback.

## Self-check
- Elite forum: V2EX yes (via sov2ex API), Zhihu yes, cnblogs searched (1 useful: Qwen3-VL paper reading note). linux.do blocked. site: operator via meta-search was unreliable (site:v2ex.com returned noise), so used sov2ex API.
- GitHub repos: yes. Issues: yes (#6, #13 etc). Ran only 3 issue/PR style searches.
- Not covered: ModelScope pages (not fetched directly; HF used), C-MTEB (text-only, out of scope), MMEB leaderboard page itself (not fetched; relied on cards/papers), BGE-VL (no hit).

## Unresolved
1. Real WeMM weights license (Apache-2.0 or other)? Read HF LICENSE body, only got code header.
2. EG2 440M MMEB-v2 = 59.01 per baseline vs published full-config 57.28/67.84/50.67; need per-size table from Google card.
3. WeMM-2B vs EG2 on identical text->frame benchmark; none exists.
4. M1 Pro actual WeMM-2B throughput (MPS vs MLX 4-bit). MLX 4-bit quality loss unmeasured.
5. Jina v5-omni numeric MIEB/MMEB-V scores.
6. DME / MoEMB weights release.

## Sources
- WeMM HF card: https://huggingface.co/tencent/WeMM-Embedding-2B ; GitHub: https://github.com/Tencent/WeMM-Embedding ; LICENSE raw: https://raw.githubusercontent.com/Tencent/WeMM-Embedding/main/LICENSE ; paper: https://arxiv.org/abs/2608.24053
- Zhihu WeMM: https://zhuanlan.zhihu.com/p/2076596701901726512 , https://zhuanlan.zhihu.com/p/2077712428910100994 , https://zhuanlan.zhihu.com/p/2076056590386468118 , https://www.zhihu.com/question/2076324817746256714
- Blogs: https://blog.xlap.top/post/tech/2026-08-28/wemm-embedding/ , https://zuphp.com/5390.html (secondary, one claims Apache 2.0 for all)
- Practitioner: https://github.com/ryangu00/apple-silicon-wemm-embedding , https://github.com/Tencent/WeMM-Embedding/issues/6 , https://github.com/metaswang/voxstudio-wemm-embedding-mlx-4bit , https://github.com/Choise-ieee/WeMM-Embedding-ONNX
- V2EX: https://www.v2ex.com/t/1240341 , https://www.v2ex.com/t/1236302
- Qwen3-VL-Embedding: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B , https://arxiv.org/abs/2601.04720
- Jina v5-omni: https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano , https://arxiv.org/abs/2605.08384 , https://sotasync.com/reader/2026-05-14-jina-embeddings-v5-omni/
- DME: https://arxiv.org/abs/2608.02148 ; MoEMB: https://arxiv.org/html/2609.08663
- Ops-MM: https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B ; Seed-1.6-Embedding: https://www.geekpark.net/news/350830
- EmbeddingGemma 2: https://cldnavi.com/zh/blog/embeddinggemma-2-guide-2026/ , https://www.winzheng.com/article/embeddinggemma-2-multimodal-on-device-embeddings-october-202
- Roundup: https://zhuanlan.zhihu.com/p/2021754924443149771 ; GME: https://modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct
