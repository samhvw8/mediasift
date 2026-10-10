# en-decode-backends: GPU decode + cross-backend inference (framesift)
Date 2026-09-30. Lang EN. Machine for own measurements: M1 Pro, macOS, ffmpeg 8.0.1, **load avg 8-22 during runs (other agents) -> absolute numbers noisy, ratios are the signal**. Samples: samples.

## TL;DR
1. **On M1, VideoToolbox hwaccel is 4-20x SLOWER than software for framesift's workload (tiny output).** Do not use it. Measured, not assumed.
2. The 3 s "scan" cost is mostly a 10-bit H.264 sample (cs50_lecture = High10, BT.2020 HDR). 8-bit 720p H.264 scans at ~0.6 s / 143 s video (~230x RT) with plain SW ffmpeg.
3. Biggest win is (3): keyframe extraction in ONE ffmpeg pass (select filter) = 5x faster than N procs; torchcodec random access even faster.
4. CUDA: NVDEC useful only for CPU offload/concurrency, not single-video latency. NVDEC does NOT decode H.264 High10 before Blackwell, AV1 only Ada/Ampere GA10x+. Need SW fallback anyway.
5. Inference: **PyTorch (MPS + CUDA) is the one-API answer.** ORT CoreML EP is bad on ViT (measured 2 img/s). MLX CUDA backend exists but Linux-only, missing ops, no ViT-speed win on Mac. Inference for 100-500 frames is seconds on M1, sub-second on RTX: not the bottleneck.

## (1) ffmpeg hwaccel + GPU scaling

### [GEM] Own measurement: VideoToolbox is slower than SW for 128x72 output (M1 Pro, loaded machine)
Source: own runs (commands in Appendix). 
| file | path | time |
|---|---|---|
| python_100s.mp4 (8-bit h264 720p, 143 s) | SW: fps=5,scale=128:72:area,rgb24 | **0.63 s** (2.0 s with `-threads 8` - noise) |
| same | `-hwaccel videotoolbox` + CPU scale | 12.5 s |
| same | `-hwaccel videotoolbox -hwaccel_output_format videotoolbox_vld`, scale_vt=w=128:h=72,hwdownload,format=nv12 | 12.8 s |
| big_buck_bunny.mp4 (8-bit, 180 s, 60 fps) | SW | 5.7 s |
| same | VT + CPU scale / scale_vt fps-first / scale_vt scale-first | 36.6 / 29.4 / 23.3 s |
| cs50_lecture.mp4 (High10 HDR, 360 s) | SW (current framesift cmd) | 3.6 s |
| same | VT + CPU scale | 8.7 s |
| same | scale_vt + hwdownload,format=p010le | 13-19 s |
PyAV 19 `HWAccel("videotoolbox")` decode only (no scale) 4286 frames = 14.1 s (~300 fps) vs PyAV SW decode+reformat+5fps ~2 s.
Caveat: machine heavily loaded, VT media engine may be contended. But 4-20x gap far exceeds noise. Per-frame hw->CPU surface copy + sync dominates when you discard 5/6 frames anyway.
Verdict: SW decode wins on Apple for this job. Remove "GPU decode on Mac" from plan.

### [GEM] scale_vt / hwdownload pitfalls (verified locally + trac)
- `hwdownload,format=nv12` FAILS on 10-bit source: "Invalid output format nv12 for hwframe download". Sw format of VT frames follows source depth: need `format=p010le` for 10-bit. 8-bit -> nv12 works. Must probe pix_fmt first.
- `scale_vt` has NO `format=` option in ffmpeg 8.0.1 ("Option not found").
- `fps` filter works on hw frames; putting it BEFORE scale_vt/hwdownload reduces transfers (29 s vs 23 s in BBB was noise-level; scale-first not better).
- Upstream bugs: scale_vt sometimes doesn't scale w/o hwdownload [ffmpeg trac #10492](https://trac.ffmpeg.org/ticket/10492); wrong colors with videotoolbox_vld + scale_vt [trac #10884](https://trac.ffmpeg.org/ticket/10884). Doc: [scale_vt](https://ayosec.github.io/ffmpeg-filters-docs/8.1/Filters/VideoToolbox-Video/scale_vt.html).

### [GEM] Codec support on M1 via VideoToolbox
- AV1: `-hwaccel videotoolbox` on M1 -> "VideoToolbox decoder for this format not found. Failed setup ... Failed to get pixel format" (own test, synthetic svt-av1). AV1 HW decode only M3+ ([salivity](https://salivity.github.io/av1/article/apple-silicon-av1-hardware-decode-support), [Apple forum](https://developer.apple.com/forums/thread/722933): "Hardware decoder not available for codec: VP9" list). 
- VP9: ffmpeg accepted `-hwaccel videotoolbox` with no error (local); Apple forum says HW not available for VP9 on VT API (entitlement-gated, [MacRumors](https://forums.macrumors.com/threads/apple-m1-vp9-av1-decoding.2269938/page-2)). Treat as unreliable.
- Takeaway: if ever using hwaccel, must init-fail -> SW fallback (ffmpeg errors hard for AV1, does not silently fall back in `-hwaccel` CLI when setup fails here).

### [MEH] CUDA path: `-hwaccel cuda -hwaccel_output_format cuda` + scale_cuda/scale_npp
No NVIDIA HW available to me; no measured numbers found for tiny-output decode. Evidence:
- Generic advice: for frame extraction GPU accel helps less since bottleneck is not decode ([RenderIO](https://renderio.dev/blogs/ffmpeg-cuda-nvenc-gpu-acceleration)); "CUDA scaling slower than software" thread, culprit hwupload ([r/ffmpeg](https://www.reddit.com/r/ffmpeg/comments/10x2xh9/cuda_scaling_slower_than_software/)).
- scale_npp needs proprietary CUDA SDK build (`--enable-cuda-nvcc`), scale_cuda only needs ffnvcodec, fewer options ([r/ffmpeg](https://www.reddit.com/r/ffmpeg/comments/euiwtv/scale_cuda_vs_scale_npp/), [SO](https://stackoverflow.com/questions/55687189/how-to-use-gpu-to-accelerate-the-processing-speed-of-ffmpeg-filter)). Homebrew/apt stock ffmpeg often lacks npp.
- Pipeline shape: `-hwaccel cuda -hwaccel_output_format cuda -i X -vf fps=5,scale_cuda=128:72:format=nv12(or rgb?),hwdownload,format=nv12,format=rgb24` ([NVIDIA guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)). Needs validation on a real box.

### [GEM] NVDEC codec limits (official)
[NVDEC app note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html): H.264 High10/High422 = **N on everything except Blackwell**; AV1 = Ada/GA10x+ only; HEVC Main10 ok Pascal+; multiple NVDECs Turing+. framesift needs CPU fallback for 10-bit H.264 (e.g. the cs50 sample) on most RTX cards.

### [MEH] SW tuning knobs tested
`scale=...:flags=fast_bilinear` 2.74 s vs area 3.6 s (cs50); `-skip_loop_filter all` 2.78 s (visual diff negligible at 128x72). Both ~25% gain, noisy. `-skip_frame nokey` gives keyframes only but output cfr dup'd frames (use -fps_mode vfr) - not tested further.

## (2) Python GPU decode libraries

| lib | status (gh, 2026-09-30) | Apple GPU decode | CUDA |
|---|---|---|---|
| torchcodec | meta-pytorch, v0.16.0 (Aug 2026; 0.17.0 on PyPI), pushed today, 1.2k stars [repo](https://github.com/meta-pytorch/torchcodec) | **No.** macOS wheels CPU-only (README). Works via Homebrew ffmpeg 4-8 | NVDEC `device="cuda"`, linux wheels CUDA by default |
| PyNvVideoCodec | NVIDIA official, MIT, successor of VPF; pip; closed-ish repo (deepstream_libraries mirror) [NVIDIA](https://developer.nvidia.com/pynvvideocodec) | no | yes, best control, cached decoder |
| VPF | **archived** (last push 2024-06) [repo](https://github.com/NVIDIA/VideoProcessingFramework) | no | dead |
| VALI | RomanArzumanyan fork, v4.8.7 Jan 2026, 71 stars, single maintainer [repo](https://github.com/RomanArzumanyan/VALI) | no (NVDEC + VAAPI/ other) | yes |
| decord | dmlc, last push 2024-07, no releases, **macOS/Apple Silicon/ffmpeg 8 broken**: [#372](https://github.com/dmlc/decord/issues/372) open, [#107](https://github.com/dmlc/decord/issues/107) | no | only via cuda-decord fork |
| PyAV | v19.0.0 released 2026-09-29, active; hwaccel via `av.codec.hwaccel.HWAccel` | VT works but slow (above) | NVDEC frames: issues #2208/#2199 show rough edges (to_ndarray errors on reserved colorspace, closed) |
| AVFoundation/pyobjc | not researched in depth; no pip ecosystem; same VT engine -> same per-frame overhead expected | - | - |

### [GEM] torchcodec on M1 CPU (own run; torchcodec 0.17.0, brew ffmpeg 8)
- 70 random-access full-res frames (`get_frames_played_at`) in **0.85 s** -> vs 70 ffmpeg procs 6.5 s. Returns [70,3,720,1280] uint8 tensor directly. No JPEG encode needed if downstream wants tensors.
- `transforms=[Resize((72,128))]` exists in 0.17 (decoder-side resize). 5fps scan of python_100s: 2.66 s (loaded) vs ffmpeg pipe SW 0.63 s -> **not faster for the dense scan; keep ffmpeg pipe for the scan**.
- Constraint: torch>=2.11 for newer torchcodec [HN](https://news.ycombinator.com/item?id=48476539); adds torch dep (heavy) - fine if torch already required for SigLIP.
- torchcodec authors: "solutions that shell out to ffmpeg binary are not going to perform well", benchmarks deliberately not published [HN](https://news.ycombinator.com/item?id=48476539).

### [GEM] CUDA decode benchmarks
- T4, 1080p H.264 300 frames full decode+materialize: torchcodec 434 fps, PyNvVideoCodec 433, decord 402, PyAV NV12 327, PyAV NV12+GPU RGB 196, PyAV RGB 100. torchcodec batch 30 = 3 GB GPU mem vs PyNvVideoCodec 246 MB -> keep batches small [LinkedIn](https://www.linkedin.com/pulse/pynvvideocodec-vs-torchcodec-pyav-video-decoding-benchmark-nekhai-vqubf) (blog-tier, single GPU).
- vLLM PR, 8 sampled frames/request, 20-core CPU: 1080p conc1 torchcodec cpu 158 ms vs cuda 374 ms (CUDA slower single-request: decoder setup + D2H); conc16 cpu 2379 ms vs cuda 1034 ms; CPU cores 8.4 -> 0.19. 4 CPU cores pinned: parity at conc1. [vllm#56333](https://github.com/vllm-project/vllm/pull/56333). **CUDA decode = CPU offload, not latency win, for a single video.**
- B200: raw 1080p H.264 12,180 fps on 2 cores (MPS) vs 4,760 fps on 57 CPU cores; NVDEC setup dominates short clips [pylerAI/nvdec-poc](https://github.com/pylerAI/nvdec-poc).
- PyNvVideoCodec decoder reuse, L40G, 2-s clips 720p: simple 1617 fps vs cached 5190 fps (3.2x); long clips 720p ~4800-5300 fps; 30 s 1080p ~2400-2500 fps [NVIDIA app note](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/application-note.html). Decoder init is the tax for one-shot processes.
- Context for framesift: SW 720p 8-bit on M1 already ~7000 frames/s in scan (4286 frames/0.63 s). NVDEC 720p ~5000 fps per L40G thread. **Consumer CUDA box gains nothing vs a modern CPU for decode speed**; only benefit if CPU is weak/busy.

## (3) Batch keyframe extraction

### [GEM] Own measurement: 70 JPEGs from python_100s.mp4 (full-res, -q:v 3)
| method | time |
|---|---|
| A. 70 sequential `ffmpeg -ss t -i X -frames:v 1` (current) | 6.49 s |
| B. same, 8-way parallel | 1.93 s |
| C. **single pass `select='lt(prev_pts*TB,t1)*gte(pts*TB,t1)+...'` -fps_mode vfr, image2** | **1.29 s** (70 files) |
| D. torchcodec `get_frames_played_at(70 ts)` (tensors, no JPEG write) | 0.85 s |
C is full decode of the file so cost scales with video length (C ~ scan cost); A/B scale with N. For long videos + few frames, B or D win; for typical 6-min/70 frames C/D win. Expression length limit: 70 terms fine; use `-filter_script` / `-/filter_complex` file for 500+.
- select by frame number `eq(n,..)` needs index from scan (scan pass already knows frame indices at 5 fps -> exact). 
- Best fold: do scan at full decode, retain JPEG-able frames? Not possible since keyframes chosen after scan. Two passes (scan -> select) or keep a ring of full-res decoded frames (720p*3B*N = 2.7 MB/frame; 1800 frames=5 GB: no). So: pass 2 via torchcodec/PyAV seeking or select filter.
- PyAV seek+decode inside one container (`container.seek(pts)` then decode to target) = equivalent to D, no torch dep. PyAV SW scan: decode+reformat(AREA) 5fps in 1.96 s loaded ([test](#appendix)).

## (4) Cross-backend inference

### [GEM] Numbers (SigLIP/CLIP vision tower, fp16, batch 32)
| hw | model | img/s | source |
|---|---|---|---|
| M1 Pro MPS (loaded machine) | open_clip ViT-B-16-SigLIP 224 | 79 | own |
| M1 Pro MPS | open_clip SigLIP-256 | 74 | own |
| M1 Pro MPS | HF transformers SigLIP B/16-224, sdpa / eager | 91 / 80 | own |
| M1 Pro MPS | ViT-B/32 224 (CLIP) | 346 | own |
| M1 Pro MPS | SigLIP2 B/16-256 (real run, 1637 frames) | 58 | [mlx-embeddings#23](https://github.com/Blaizzy/mlx-embeddings/issues/23) |
| M1 Pro MPS | ViT-L/16-256 SigLIP 18; L/384 7.2; SO400M/384 3.4 | | same |
| M1 Pro MLX | siglip-so400m-224 15.4; so400m-384 4.8 (MPS so400m-384 3.4) ; large-384 9.05 vs MPS large-384 7.2 | | same |
| RTX 4090 (timm, AMP, bs512, torch 2.4) | SigLIP B/16-224 2830; CLIP B/16 2841; SigLIP B/16-256 2176; CLIP B/32 10,419 | | [timm csv](https://github.com/huggingface/pytorch-image-models/blob/main/results/benchmark-infer-amp-nhwc-pt240-cu124-rtx4090.csv) |
- M1 Pro MPS 90 img/s ~ 30x below 4090. 500 frames B/16: ~6 s M1 (B/32 ~1.5 s) vs ~0.2 s 4090. M-series fine for 100-500 frames.
- My random-init weights (no download): throughput is architecture-determined; fine.
- MLX gives only ~1.25x over MPS on large SigLIP (same issue data), one user "expected more". Not worth a second framework.

### [GEM] PyTorch MPS vs CUDA maturity
- MPS works for ViT inference fp16; stricter than CUDA; op gaps fall back via `PYTORCH_ENABLE_MPS_FALLBACK=1`. Known perf trap: **`nn.MultiheadAttention` on MPS ~9x slower than direct F.scaled_dot_product_attention** [pytorch#181725](https://github.com/pytorch/pytorch/issues/181725) -> open_clip's OpenAI-style blocks use nn.MultiheadAttention; HF transformers SigLIP (sdpa) measured 91 vs 79 img/s (only 15%, so either already patched or other overhead; recheck on quiet machine). Prefer HF transformers SigLIP w/ sdpa, or timm.
- Custom MPS attention op reports wins for seq>=1024 only (irrelevant, ViT-B seq=196) [forum](https://discuss.pytorch.org/t/faster-attention-with-mps-backend/224884).
- M3+ native bf16; M1/M2 use fp16 [r/pytorch](https://www.reddit.com/r/pytorch/comments/1szy1bs/faster_attention_on_apple_silicon/). Use fp16 on both backends.

### [GEM] ONNX Runtime CoreML EP = do not use for ViT (own test)
ORT 1.30-era wheel, HF SigLIP B/16 exported opset 17, bs32 fp32, M1 Pro loaded: **CoreML EP 2 img/s vs CPU EP 8 img/s** (vs MPS ~90). Log: 53 partitions, 459/511 nodes supported -> many CPU<->CoreML round trips. `ModelFormat=MLProgram` fails to compile ("Conv: Required param 'pad' is missing"). Matches known issues: [#17654](https://github.com/microsoft/onnxruntime/issues/17654), [#16934](https://github.com/microsoft/onnxruntime/issues/16934), [#17033 FP16 slow](https://github.com/microsoft/onnxruntime/issues/17033), [#28022](https://github.com/microsoft/onnxruntime/issues/28022), [MACGPU blog](https://macgpu.com/en/blog/2026-0420-mac-onnx-runtime-coreml-ep-vs-cpu-dynamic-shapes-remote.html) (blog tier). ORT CUDA EP is fine on NVIDIA, but then two runtimes differ -> the point of ORT (one API) collapses on Mac.

### [GEM] MLX CUDA backend
- Exists: `pip install "mlx[cuda12]"` / `mlx[cuda13]`; needs Linux glibc>=2.35, SM>=7.5, driver>=550.54.14 [install docs](https://ml-explore.github.io/mlx/build/html/install.html). MLX 0.32.3 released 2026-09-29, 28.6k stars.
- **Linux only; no CUDA-on-Mac, no Windows** ([discussion #2422](https://github.com/ml-explore/mlx/discussions/2422)). Missing at launch: quantized matmul, gather-mm, FFT, linalg; **unsupported op crashes (no CPU fallback)** - maintainer. Open CUDA bugs e.g. [#3876](https://github.com/ml-explore/mlx/issues/3876) Blackwell hang. Real-world use so far: exo on DGX Spark ([X post](https://x.com/AlicanKiraz0/status/2036935870612124030)) = LLM serving, not vision.
- SigLIP on MLX = mlx-embeddings (Blaizzy, 446 stars; models siglip-so400m etc.), no CLIP ViT-B/32 first-class. Would mean porting/maintaining model code per op coverage. MEH for framesift.

### [MEH] Core ML via coremltools / tinygrad
- coremltools 9.0 (Nov 2025), pushed Sep 2026, 5.4k stars: ANE path possible but two model artifacts (mlpackage vs torch), conversion step per model, Apple-only. No throughput numbers found for SigLIP. Not measured. Given ORT-CoreML result, expect partition pain for dynamic ops; a static-shape converted ViT may do well (ANE) but unverified.
- tinygrad 0.14.0 (Aug 2026), has extra/models/clip.py; Metal + CUDA backends = genuinely one API; but no published SigLIP benchmark, ecosystem/model-loading burden. Unbenchmarked.

### [NOISE] 
Generic RTX 4090 LLM tokens/s blogs; SEO "2026 ORT guide" pages except where noted.

## Forum coverage
HN: torchcodec 0.14 thread (used), social_search torchcodec/MLX CUDA on HN/Reddit mostly empty. Reddit r/ffmpeg cuda scaling threads (used). Reddit search via focus returned 0 for SigLIP MPS-vs-CUDA. Lobsters: no relevant thread found (search API needs ids). Reddit r/ffmpeg "videotoolbox slower than SW" not found; own measurement substitutes.

## Recommended design
Decode (all platforms): **software ffmpeg for the dense scan** (`fps=5,scale=128:72:flags=area,format=rgb24` pipe as now; optionally `-skip_loop_filter all`, fast_bilinear after visual QA). Measured 230x RT on 8-bit 720p M1; no hwaccel -> no codec-fallback matrix, no 10-bit/AV1/VP9 pitfalls. Add `-threads` default only.
Keyframe extraction: replace N procs with **one pass** `select=` (or `-filter_script`) when N>~20 and video <~15 min; else keep parallel (ThreadPool 8) `-ss`. Or torchcodec `get_frames_played_at` (0.85 s / 70, returns tensors; feed SigLIP directly w/o JPEG round trip, write JPEG only for output with PIL/torchcodec JpegEncoder). If avoiding torch dep in decode module, PyAV seek+decode.
CUDA box: optional opt-in `--hwaccel cuda` path ONLY if (a) probe says codec/profile in NVDEC matrix (no High10 H.264, AV1 only Ada+), (b) CPU cores scarce. Default off; it does not speed up single videos. If added, prefer torchcodec `device="cuda"` (already a torch dep, same API as CPU path) over PyNvVideoCodec/VALI/decord. Reuse decoder across videos.
Inference: **PyTorch only** - `device = "cuda" if available else "mps" else "cpu"`, fp16, HF transformers SigLIP (sdpa attention) or open_clip after checking MPS MHA slowdown, batch 32-64, `torch.inference_mode`, `PYTORCH_ENABLE_MPS_FALLBACK=1`. One codebase, one weights format, ~90 img/s on M1 Pro (B/16) = 500 frames in ~6 s; B/32 CLIP 4x faster. Skip ORT-CoreML, MLX, tinygrad until measured need. Decode is then the dominant cost again, not inference.

## Unresolved
1. VT slowness partially confounded by machine load; re-run on quiet M1 (and M3/M4) to confirm 4-20x. Is VT faster with `-threads 1`/`-hwaccel_flags` or with 4K sources? (likely yes for 1080p+/4K; not tested.)
2. No NVIDIA box: scale_cuda vs scale_npp vs CPU scale numbers for 128x72 output unmeasured; need one RTX run.
3. torchcodec `transforms=Resize` implementation (ffmpeg filter graph? swscale flags, area vs bilinear parity with current `flags=area`) unverified; torchcodec macOS GPU decode roadmap: no issue found.
4. MPS quiet-machine numbers; open_clip MHA slowdown effect vs HF; bf16/fp16 on M1 accuracy; torch.compile on MPS.
5. Core ML/ANE (coremltools static-shape SigLIP) throughput unknown - only path that might beat MPS on Mac.
6. AVFoundation/pyobjc not investigated; expected same VT per-frame overhead.
7. Real CLIP/SigLIP RTX numbers with bs32-ish and torch>=2.5 (timm number is bs512 pt2.4).

## Appendix: commands
- SW: `ffmpeg -v error -i F -vf "fps=5,scale=128:72:flags=area,format=rgb24" -f rawvideo -`
- VT: `ffmpeg -hwaccel videotoolbox -hwaccel_output_format videotoolbox_vld -i F -vf "fps=5,scale_vt=w=128:h=72,hwdownload,format=nv12,format=rgb24" -f rawvideo -` (p010le for 10-bit)
- select: `-vf "select='lt(prev_pts*TB,t)*gte(pts*TB,t)+...'" -fps_mode vfr -q:v 3 out%03d.jpg`
- Scripts: /tmp/sig_bench.py, /tmp/hf_bench.py, /tmp/ort_bench.py, /tmp/tc.py, /tmp/tc2.py, /tmp/pyav_t.py

## Sources
torchcodec repo/HN/PRs, vllm#56333, pylerAI/nvdec-poc, NVIDIA PyNvVideoCodec + NVDEC docs, LinkedIn benchmark, VPF/VALI/decord repos+issues, PyAV issues, ffmpeg trac #10492/#10884, scale_vt docs, r/ffmpeg threads, MLX install doc + discussion #2422 + issues, mlx-embeddings#23, timm benchmark csv, ORT issues, pytorch#181725, coremltools/tinygrad repos (URLs inline above).
