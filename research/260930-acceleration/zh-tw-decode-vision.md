# zh-TW: decode / keyframe / embeddings / OCR acceleration

**Verdict: coverage is THIN.** TW community (PTT, iThome, Mobile01, Medium-TW) has almost nothing with real numbers for this stack. ~14 searches run. Most "zh" hits are simplified-Chinese (Zhihu/CSDN/Tencent) -- reported here as ZH-CN overflow, tagged. Nothing TW-native on Apple Silicon video decode, scale_vt, torchcodec/PyNvVideoCodec, SigLIP on MPS, ocrmac. Do not treat absence as evidence of no problems.

Search-quality note: parallax engines degraded (duckduckgo blocked, 5 engines no hits). `site:ptt.cc` / `site:ithelp...` returned 1 and 0 hits. iThome: 0 relevant. Mobile01: 0. PTT Soft_Job/Python/MacShop: 0. Medium-TW: 0.

## Findings

### 1. NVDEC + scale_cuda all-GPU pipeline, real numbers -- GEM (TW, PTT)
- PTT AVEncode, 2024-10, author YouCantGetME, HW 12700k + RTX 4090, source 4K 55Mbps 10-bit HEVC -> 1080p 8-bit AVC.
- Xmedia Recode (CPU decode + NVENC): 1x / 25 fps, CPU 100%, NVDEC 0%: "很明顯瓶頸在解碼".
- `ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i in -vf "scale_cuda=-1:1080:format=yuv420p" -c:v h264_nvenc`: ~10x / 255 fps, CPU 5%, NVDEC 100%, NVENC 30%.
- NVDEC/NVENC are separate silicon: "4050與4090解碼編碼速度跟內容會完全一樣" (same decode speed across GPU tiers; 4070+ has 2 NVENC but a single file uses only one).
- Pitfall quote: CUDA-decoded frames live in GPU memory and "除了CUDA功能...外是不能直接使用的, 需要從GPU記憶體搬回系統記憶體才能再套其他傳統濾鏡" -> hwdownload + format needed; "Impossible to convert between the formats supported by the filter" is the classic error. Open-source tools rarely do HW decode because of these branches.
- Relevance: framesift needs 128x72 RGB on CPU anyway -> scale_cuda then hwdownload (tiny frames = cheap PCIe) is the pattern. Caveat: transcode benchmark, not raw-pipe extraction.
- https://www.ptt.cc/bbs/AVEncode/M.1728622825.A.795.html

### 2. RapidOCR on Mac, TW blog test -- MEH (useful facts, CPU-only)
- TechMoon (TW), macOS Apple Silicon, rapidocr 3.9.2, onnxruntime 1.29.0, CPU only: 0.13-0.23 s/image, engine init 0.17 s.
- Default model read Traditional Chinese screenshot (presidential office site, 18 lines) correctly. Switching to the dedicated "繁中" rec model WORSENED it: 健康台灣 -> 健康合灣, stray 口, "20:30" -> "20-30". Only 2 images tested; author admits GPU engines, batching not measured.
- Takeaway: for RapidOCR on CUDA, keep default models; do not assume the ch_tra model is better. Single-image evidence, weak.
- https://techmoon.xyz/rapidocr/

### 3. Seek/keyframe extraction tips (TW blogs) -- MEH (old, well-known)
- miniasp (The Will Will Web, 2022): seek lands on keyframe only; asking 00:28:16 returned 0:28:11.9. -> `-ss` before `-i` fast seek is keyframe-snapped unless decoded forward. https://blog.miniasp.com/post/2022/10/08/Useful-tool-FFmpeg
- lnpcd / chuiwenchiu blogs: `-ss` before `-i` = fast keyframe seek; after = slow. Old (2012/2013). https://lnpcd.blogspot.com/2012/09/ffmpeg.html , https://chuiwenchiu.wordpress.com/2013/05/05/
- Relevance: framesift's 80 single-process keyframe extractions: input-side `-ss` is already the right design; nothing new.

### 4. ZH-CN overflow (not TW, but concrete) 
- Tencent Ads GPU frame extraction: T4 has 2 NVDEC; NVDEC does not benefit from external multithreading; use n+1 workers per NVDEC count; single instance caps utilisation <50%; ~2x decode capacity with CPU/CUDA load unaffected; fallback to CPU decode for unsupported formats; 720p ~26 s clips. GEM for CUDA design. https://cloud.tencent.com/developer/article/2013062 (NVIDIA blog mirror: https://developer.nvidia.com/zh-cn/blog/nvidia-tencent-gpu-accelerate/)
- CSDN: `-skip_frame nokey` decode-only-keyframes claims "10x" (unverified numbers, content-farm). MEH. Relevant idea: keyframe-only decode for the 80 extractions. https://blog.csdn.net/s1t2u3/article/details/152399476
- decord/NVDEC/pipe breakdown (CSDN, 2026): recommends decord GPU bridge for random access. MEH, unverified. https://blog.csdn.net/lzhcoder/article/details/161510512
- Zhihu: PyTorch MPS on M1 MNIST epoch 19.6 s vs CPU 33.4 s (1.7x); P100 CUDA 20.4 s vs CPU 34.2 s. Toy model, 2022. MEH. https://zhuanlan.zhihu.com/p/517699916
- MLX vs MPS vs CUDA (Zhihu/Aliyun translation of Medium bench): MLX 2.34x faster than MPS on M1 Pro, +24% on M2 Ultra, no real gain on M3 Pro; CUDA still faster overall; MLX conv ops slow. GEM-ish for "MLX vs MPS on M1 Pro" -- conv-heavy ViT embedders may not benefit. https://developer.aliyun.com/article/1400914 , https://developer.volcengine.com/articles/7387624901676367891
- PaddleOCR GPU discussion (CN): multithread use_gpu only ~2x, GIL limits, per-thread instance wastes VRAM. MEH. https://github.com/PaddlePaddle/PaddleOCR/discussions/14267
- Zhihu 4-local-OCR review: RapidOCR 1-2 s/image (CPU?), accuracy ~PaddleOCR. NOISE-ish. https://zhuanlan.zhihu.com/p/2023055658082117490

### 5. ONNX Runtime CoreML EP failure modes (EN GitHub/blogs surfaced by zh query) -- GEM
- FP16 8x slower than FP32 on M1 Pro CoreML EP: https://github.com/microsoft/onnxruntime/issues/17033
- CoreML EP silently converts to FP16 (prediction drift vs MPS/CPU): https://ym2132.github.io/ONNX_MLProgram_NN_exploration
- Nodes silently fall to CPU ("VerifyEachNodeIsAssignedToAnEp"): https://github.com/microsoft/onnxruntime/issues/16934
- Crash on macOS 15 with CoreML: https://github.com/microsoft/onnxruntime/issues/22275
- MLProgram + static-shape options; dynamic rank pitfalls: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html , https://macgpu.com/en/blog/2026-0420-mac-onnx-runtime-coreml-ep-vs-cpu-dynamic-shapes-remote.html (vendor blog, MEH)
- torchcodec vs decord: issue #426 "slower than decord" (CPU codec); torchcodec docs recommend "beta" CUDA interface. https://github.com/meta-pytorch/torchcodec/issues/426 , https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/performance_tips.html
- PyTorch MPS slower than CPU on small workloads (M1): https://github.com/pytorch/pytorch/issues/77799

### 6. NOISE
- llmmac.com/tw CLIP/SigLIP CoreML "matrix" page: SEO for remote-Mac rental, fetch yielded no numbers. https://llmmac.com/tw/blog/articles/2026-mac-multimodal-clip-siglip-coreml-embedding-matrix.html
- SemIf issue #33 (TW author, Mac mini M4 MLX vs MPS): page fetch returned no matched numbers; unverified. https://github.com/EndeavorYen/SemIf/issues/33
- compute-market Mac mini M4 (EN, LLM focus), yre.jp / originell VideoToolbox *encode* posts (wrong direction, encode not decode).

## Coverage table
| Sub-topic | TW coverage |
|---|---|
| ffmpeg cuda decode speed/pitfall | 1 GEM (PTT) |
| videotoolbox decode, scale_vt, hwdownload | none TW |
| torchcodec / PyNvVideoCodec / decord / PyAV | none TW (CN only) |
| MPS vs CUDA inference | none TW (CN toy benchmarks) |
| MLX | none TW for vision (LLM only) |
| ORT CoreML EP | none TW (EN issues) |
| Apple Vision / ocrmac | none TW with numbers |
| RapidOCR/PaddleOCR | 1 TW blog (CPU Mac) |

## Unresolved
- videotoolbox decode speed for 1080p h264 -> 128x72 on M1 Pro vs software (76 s baseline)? No TW source; needs own benchmark or EN sources.
- Does scale_vt + hwdownload beat sw decode at tiny output? Unknown.
- ocrmac/Vision accuracy + latency for Traditional Chinese slides? Not found.
- SigLIP/MobileCLIP img/s on M1 Pro MPS vs CoreML vs RTX? Not found.
- PyNvVideoCodec TW experience: none.
- Gaps may be search-engine degradation (only 1-2 of 10 engines answered), so a retry with WebSearch (US-only) or bsk on ptt.cc search could find more.
