# framesift complementary libs/models (EN) - gathered 2026-09-30
Tags: GEM / MEH / NOISE. Repo stats via `gh api` on 2026-09-30.

## 0. Bottom line
- PySceneDetect detectors = same idea as framesift's heuristics (HSV/hist/hash diffs + rolling avg). NOT an upgrade. NOISE for replacement.
- Real upgrade over pixel stats = semantic embeddings (SigLIP2 / MobileCLIP2 / DINOv2) on kept-keyframe candidates: fixes gesture-redundancy class.
- Screen text: Apple Vision (ocrmac) on Mac, RapidOCR on CUDA/CPU. PaddleOCR proper = Mac install pain.
- Silero VAD = drop-in replacement for adaptive-RMS pause logic. MIT, tiny, CPU.
- Diarization only if speaker identity wanted: pyannote community-1 (MPS 24x RT, CUDA 32x RT) or speakrs (Rust, 529x RT on M4 Pro CoreML).

## 1. Shot boundary detection

### PySceneDetect [MEH for framesift]
- v0.7.1 2026-07-22, BSD-3, 5.2k stars, pushed 2026-09-21. https://github.com/Breakthrough/PySceneDetect ; docs https://www.scenedetect.com/docs/latest/api/detectors.html
- Detectors: Content (HSV weighted diff, thr 27), Adaptive (rolling avg of Content score, window 2, ratio 3.0), Threshold (fade in/out via avg intensity), Histogram (Y-channel YUV hist), Hash (perceptual hash). All classical = what framesift already does. framesift's local-outlier test ~ AdaptiveDetector.
- AutoShot paper: PySceneDetect F1 < 0.6 on SHOT dataset, "hardly handles challenging" transitions. https://ar5iv.labs.arxiv.org/html/2304.06116
- CPU-only, no GPU concerns. Speed: no hardware-named figure found (unresolved).
- Open issue #511 "Deep Learning Detector with Trained Model": maintainer invited TransNetV2-ONNX PR; contributor's detector has BGR-vs-RGB bug (cv2 frames passed unconverted). Still OPEN = no official neural detector. https://github.com/Breakthrough/PySceneDetect/issues/511 [GEM: shows integration pitfalls]

### TransNetV2 [GEM for hard-cut + dissolve, with caveats]
- Original soCzech/TransNetV2: MIT, 1.0k stars, last push 2023-12-04 (dormant, TF). https://github.com/soCzech/TransNetV2
- F1: ClipShots 77.9, BBC Planet Earth 96.2, RAI 93.9 (PyTorch port README). Claimed ~250 fps on moderate hardware (emergentmind summary, unverified, hardware not named). Input is tiny 48x27 frames, 100-frame windows, so decode dominates cost.
- PyTorch port `transnetv2-pytorch` 1.0.5 (allenday): "identical results as TF". CPU/CUDA/MPS. WARNING in its README: MPS has numerical inconsistency (ops fall back to CPU inconsistently -> different detections vs CPU). `--device auto` picks reliable device. https://pypi.org/project/transnetv2-pytorch/ [GEM pitfall: run on CPU on Mac; net is small]
- ONNX: elya5/transnetv2 on HF (input 100 x 48x27). https://huggingface.co/elya5/transnetv2 . tf2onnx conversion of original fails at ColorHistograms/UnsortedSegmentSum (https://github.com/onnx/tensorflow-onnx/issues/2106) -> use existing ONNX, don't reconvert.
- Failure mode: near-identical long take (same digital-human presenter) got 0 boundaries, logits below 0.5 (third-party eval, Chinese-language repo note) https://github.com/yangohuang/OmniShotCut [MEH: anecdotal]. Also trained on movie/web video, not slides/screen recordings.

### AutoShot (CVPR-W 2023) [MEH]
- wentaozhu/AutoShot, MIT, 260 stars, last push 2023-04-18. Reports +4.2% F1, +3.5% precision over TransNetV2 on its SHOT short-video set. https://ar5iv.labs.arxiv.org/html/2304.06116 . Dormant, NAS-searched, no PyPI/ONNX. Skip.

### OmniShotCut (newest) [GEM-watch, not yet adoptable]
- UVA-Computer-Vision-Lab/OmniShotCut, MIT code, 317 stars, pushed 2026-09-09, arXiv 2604.24762 (Apr 2026, NeurIPS 2026). `pip install omnishotcut`, weights on HF (v1.5: 52.8M params). Explicitly claims domains incl. **screen recording, game, vlog**, and labels dissolve/fade/wipe (inter-shot types). https://github.com/UVA-Computer-Vision-Lab/OmniShotCut ; https://arxiv.org/abs/2604.24762
- Install says "install PyTorch with CUDA"; MPS not documented/tested (unresolved). Needs ffmpeg binary; README says benchmark/training code not yet released. No speed numbers found (paper HTML fetch returned nothing matching). Paper PDF license CC-BY-NC-ND is for the paper only.
- Verdict: evaluate on a few framesift videos; if gradual-change detection improves, could replace windowed novelty. Risk: young, single lab, CUDA-first.

## 2. VAD

| | Silero VAD | TEN VAD | WebRTC VAD | pyannote seg |
|---|---|---|---|---|
| Latest | v6.2.3 2026-09-23 | v1.0 2025-07-11, last push 2026-02 | py-webrtcvad no release, push 2024-07 | 4.0.7 2026-06-30 |
| License | MIT | "Apache 2.0 + additional conditions" (NOASSERTION) | BSD/other | MIT code, gated HF weights |
| Apple/CUDA | CPU/ONNX, any; GPU optional | prebuilt libs macOS/Linux/Win/web/iOS/Android; CPU | CPU | torch MPS/CUDA |
| Speed | <1 ms per 30+ms chunk, 1 CPU thread | RTF 0.016 on M1; 0.0086 Xeon 6348; Silero 0.0127 same Xeon | fastest, frame 10-30ms | heavy |

- Silero quality table (wiki): v5 beats WebRTC on most sets (e.g. 0.96 vs 0.82 first col). https://github.com/snakers4/silero-vad/wiki/Quality-Metrics [GEM] . Main repo: https://github.com/snakers4/silero-vad
- TEN VAD claims lower latency on speech-to-silence transitions, catches short gaps Silero misses; vendor benchmark (TEN-framework), 2.3k stars. https://github.com/TEN-framework/ten-vad ; HN Show HN thread has no substantive critique https://news.ycombinator.com/item?id=44441458 [MEH: vendor numbers, prebuilt binary lib, odd license]
- Silero failure modes (GitHub): livekit/agents#6956 v6.2 checkpoint -> false speech / agent self-interruption (https://github.com/livekit/agents/issues/6956); snakers4/silero-vad#396 speech on perfect silence regression v3->v4; faster-whisper#843 near-silence not removed (https://github.com/SYSTRAN/faster-whisper/issues/843). Pin checkpoint version. Silero is speech-only (ignores music/noise) - desirable for "pause" but means music-only segments read as silence.
- WebRTC VAD: 10-30 ms frames, high resolution but noisy; maintenance stale. Issue #68 benchmark discussion https://github.com/wiseman/py-webrtcvad/issues/68 [NOISE for new adoption]
- New contender: FireRedVAD claims SOTA over Silero/TEN. https://github.com/FireRedTeam/FireRedVAD [unverified, NOISE until tested]
- vs framesift adaptive RMS: VAD separates speech from non-speech loud (keyboard, music, room noise). GEM: small win, near-zero cost.

## 3. Speaker diarization (only if needed)
- pyannote.audio 4.0.7 (2026-06-30), MIT code, 10.6k stars. community-1 DER: AMI-IHM 17.0 (3.1: 18.8), DIHARD3 20.2 (21.4), VoxConverse 11.2 (11.2). https://github.com/pyannote/pyannote-audio
  - Speed (vendor, H100): community-1 31-37 s per hour audio (~100x RT). Independent via speakrs: community-1 on M4 Pro MPS = 24x RT (2999 s for VoxConverse dev), RTX 4090 CUDA = 32x RT. https://github.com/attevon-llc/speakrs [GEM: rare MPS vs CUDA numbers, same benchmark] . So MPS ~ CUDA-4090 class; both slow vs Rust.
  - Friction: HF gated model + token; torchcodec + ffmpeg required (libtorchcodec load errors: https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/5851); no aarch64 wheel pain on ARM Linux https://github.com/pyannote/pyannote-audio/discussions/1990 . 
- speakrs (Rust reimplementation of community-1 pipeline): Apache-2.0, 134 stars, pushed 2026-09-26 (upstream avencera/speakrs; the attevon-llc page is a fork). M4 Pro CoreML 7.1% DER at 529x RT; RTX 4090 CUDA 7.0% at 59x (121x cuda-fast); matches pyannote DER. No Python. https://github.com/avencera/speakrs [GEM, young project]
- NeMo Sortformer (streaming 4spk v2/v2.1, 117M params): hard cap 4 speakers (NeMo#14546 https://github.com/nvidia/nemo/issues/14546), NVIDIA Open Model License, NeMo = CUDA-centric; CoreML port requested in soniqo/speech-swift#82. [MEH for framesift: Mac friction, 4-speaker cap]
- diart (2.0k stars, pushed 2026-06, last release v0.9.2 2025-02): streaming pyannote wrapper; stale releases. [MEH]
- DiariZen CoreML port: RTF 0.037 on M2 Pro, but weights CC-BY-NC, ANE unused. https://github.com/NikiO-INO/diarizen-apple [MEH: non-commercial]
- Note: Reddit/HN/Lobsters produced nothing on diarization speed (Reddit search returned no relevant posts).

## 4. Semantic frame embeddings (the gesture-redundancy fix)
Mechanism: cosine distance between embeddings of consecutive candidate keyframes ignores pose/gesture jitter of same presenter/background but jumps on new slide/scene. Use drift-from-last-kept on embedding instead of pixel stats. Caveat: CLIP-family may collapse distinct slides that look alike (same template); DINO-family is better at fine visual detail (see below), so test both on slides.

- **MobileCLIP2** (Apple, TMLR Aug 2025) repo 1.7k stars, pushed 2026-09-11. Code MIT; **model weights under Apple ML Research Model TOU** (check commercial terms). Latency (iPhone 12 Pro Max, from README): S0 1.5 ms img, S2 3.6, B 10.4, S3 8.0, S4 19.6; IN-1k zero-shot S0 71.5, S2 77.2, B 79.4. S4 ~ SigLIP-SO400M accuracy at 2x fewer params. https://github.com/apple-aiml-research/ml-mobileclip [GEM: best speed/accuracy]. Works via open_clip (needs patch) or ml-mobileclip package; needs `timm` from git. Install friction moderate. No MPS-measured fps found (unresolved).
- **open_clip** 14.2k stars, v3.3.0 2026-02-27, pushed 2026-09-29. Hosts SigLIP, SigLIP2?, MobileCLIP, DFN. Throughput gist, RTX 3090 (unbatched loop, so understated): ViT-B-16-SigLIP 152 img/s, ViT-B-32 170, ViT-L-14 54, SigLIP-SO400M 50, SO400M-384 17.6. https://gist.github.com/TACIXAT/ecd4f636bf6af28cb69d641e29d7b362 [MEH: unbatched; still gives relative order]
- **SigLIP2** (Google, Apache-2.0 via HF transformers/ google-research/big_vision 3.5k stars, pushed 2025-05). Ships as HF `google/siglip2-*` checkpoints; runs on MPS/CUDA through transformers. Used in the wild: B0yko/tern (Apple Silicon local video search: Whisper + Silero VAD + Apple Vision OCR + SigLIP-2 768-dim embeddings on keyframes). Notes "SigLIP always returns nearest frames -> need noise floor". https://github.com/B0yko/tern [GEM: same stack pattern as framesift, proprietary license but architecture notes useful]
- **DINOv2/DINOv3**: edenai blog claims vision-only models beat CLIP on near-duplicate/visual dedup (secondary source). https://www.edenai.co/post/best-image-embeddings [MEH: blog]. DINOv3 repo 11.5k stars, pushed 2026-07-15, license shown NOASSERTION (custom DINOv3 License, verify). DINOv2 Apache-2.0 (from memory, unverified here).
- Rec: default MobileCLIP2-S0/S2 or SigLIP2-B/16 at 224; compare against DINOv2-S/14 (21M params) for slide-vs-slide separation. Run only on candidate frames (post pixel-stat filter), so throughput (~100+ img/s on any modern GPU, tens/s on M-series est.) is irrelevant vs ffmpeg decode.
- No reliable Apple-Silicon fps figures found for any of these (search empty). Must benchmark locally.

## 5. OCR for screen recordings

| Engine | Mac | CUDA | Last release | License | Note |
|---|---|---|---|---|---|
| Apple Vision via ocrmac | native, ANE/GPU | no (Mac only) | ocrmac push 2026-01 (548 stars, MIT) | MIT wrapper, OS API | 207 ms accurate / 131 ms fast / 174 ms livetext per image on M3 Max (README) |
| RapidOCR v3.9.2 (2026-07-21) | CPU via onnxruntime; torch backend | onnxruntime/torch/TensorRT; docs say ORT-GPU weak, torch CUDA supported | 8k stars | Apache-2.0 | PP-OCRv4/v5 ONNX models, pip only |
| PaddleOCR v3.7.0 (2026-06-11) | install history broken on Apple Silicon (#13061, #11588, #10820); Apple support only for PaddleOCR-VL via MLX | yes | 90k stars | Apache-2.0 | strongest ecosystem, worst Mac friction |
| EasyOCR v1.7.2 (2024-09-24) | MPS added 2023 | yes | 30k stars | Apache-2.0 | stale, slow |
| Surya v0.22.1 (2026-07-20), 650M VLM | llama.cpp Metal: 0.108 pages/s (~30 W) | vLLM 5.35 pages/s on RTX 5090 | 21k stars | code Apache-2.0, **weights modified OpenRAIL-M** (free <$5M) | olmOCR-bench 83.3; too slow on Mac |
| docTR v1.1.0 (2026-08-21) | CPU/torch | yes | 6.4k stars | Apache-2.0 | configurable det/rec |

Sources: ocrmac README https://github.com/straussmaximilian/ocrmac ; Surya README https://github.com/datalab-to/surya ; RapidOCR discussion #225 https://github.com/RapidAI/RapidOCR/discussions/225 ; Paddle Mac issues https://github.com/PaddlePaddle/PaddleOCR/discussions/13061 , https://github.com/PaddlePaddle/PaddleOCR/issues/10820 ; Paddle-VL Apple doc https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html

- **Apple Vision** [GEM on Mac]: HN thread: ocrmac/`VNRecognizeTextRequest` is the legacy API; newer Swift-only ImageAnalyzer/LiveText is more accurate; vertical Japanese works; on-device. ocrmac exposes livetext (macOS 14+) which needs no pyobjc Swift. https://news.ycombinator.com/item?id=39154829 . ocrmac confidence output is 1.0 for livetext. Fast enough to OCR every kept keyframe. Mac-only => CUDA box needs RapidOCR.
- **RapidOCR** [GEM as cross-platform fallback]: one pip, ONNX models. Discussion #515 reports ORT text detector slower than native Paddle (det 100-210 ms/img CPU); #237 "slower and worse" than PaddleOCR. RapidOCR's own blog benchmarks vs PaddleOCR on Mac https://rapidai.github.io/RapidOCRDocs/main/en/blog/2026/02/06/rapidocr-vs-paddleocr/ (numbers not extracted).
- Contra Collective M5 Max table (PaddleOCR 6.8 pages/s, docTR 4.1, Surya 1.9) [NOISE/MEH]: SEO-style blog, claims Paddle "Metal accelerated path" which conflicts with Paddle's own Apple-Silicon issues. Don't cite. https://contracollective.com/blog/local-ocr-document-extraction-apple-silicon-m5-max-2026
- HN OCR accuracy list (surya 97.4, paddle 92.96, tesseract 92.4) is on document scans not screens. https://news.ycombinator.com/item?id=43047121 [MEH]
- Screen-text use: OCR text diff between keyframes is also a strong "new information" signal (slide changed vs gesture) - second fix for the known weakness.

## 6. Forum coverage
- HN: Apple Vision wrapper thread, TEN VAD Show HN, OCR-accuracy thread, PaddleOCR threads (cited above).
- Reddit: site: searches returned only generic subreddit landing pages; search_reddit with focus returned 0. No usable r/MachineLearning / r/computervision / r/LocalLLaMA evidence. Lobsters: parallax requires a story id; no relevant stories located. Gap documented.

## 7. Shortlist (adopt order)
1. **Silero VAD** (pin version, v6.2.3 has regressions reported at livekit): replaces adaptive RMS pauses; MIT, CPU, <1ms/chunk, same on Mac/CUDA. Cheapest win.
2. **Embedding-drift keyframe gate** with MobileCLIP2-S2 or SigLIP2-B/16 (benchmark DINOv2-S against them on slide decks): directly fixes gesture redundancy. Keep pixel stats as cheap pre-filter; embed only candidates. Runs on MPS and CUDA via torch.
3. **OCR sidecar**: ocrmac (livetext) on Mac, RapidOCR (torch/onnx) elsewhere behind one interface; emit text per keyframe, use text-diff as keep/drop signal and give the agent text not pixels. Skip PaddleOCR proper (Mac install) and Surya on Mac (0.1 pages/s).
4. **TransNetV2 via ONNX (elya5) or transnetv2-pytorch on CPU**: optional cut/dissolve validator on the 48x27 stream; do not use MPS. Evaluate OmniShotCut v1.5 in parallel on screen-recording material (CUDA first).
5. Diarization: only if speaker labels required. pyannote community-1 (MPS ~24x RT) simplest; speakrs for speed. Skip Sortformer (4-speaker cap, CUDA-only ecosystem), diart (stale).
- Do not adopt: PySceneDetect as replacement (duplicates current code), AutoShot (dormant), WebRTC VAD, TEN VAD (license + vendor benchmarks), EasyOCR (stale).

## 8. Unresolved
- MobileCLIP2/SigLIP2/DINOv2 measured img/s on M-series MPS and on CUDA with batching: no credible number found; needs local benchmark.
- OmniShotCut speed and MPS support; F1 table not extracted; screen-recording accuracy unverified.
- TransNetV2 fps with hardware named; PySceneDetect fps with hardware named.
- Does embedding cosine separate consecutive slides with shared template (CLIP collapse risk)? Needs test on framesift's own corpus.
- MobileCLIP2 weights "Apple ML Research Model TOU" terms vs framesift's distribution.
- DINOv3 license exact terms; SigLIP2 license (believed Apache-2.0) not re-verified on HF.
- Apple Vision legacy VNRecognizeTextRequest vs newer Swift RecognizeTextRequest accuracy gap on screen text (no measurement).
- RapidOCR ORT CoreML EP viability on Mac.
