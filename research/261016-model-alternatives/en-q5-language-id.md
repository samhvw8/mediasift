# Q5: Fast spoken language ID alternatives to whisper-tiny (EN gatherer, 2026-10-16)

Baseline: faster-whisper tiny, int8, CPU, 30 s window. All timings below are MEASURED by me on this M1 Pro 16 GB, Python 3.14 (3.12 for coremltools), onnxruntime 1.31 CPU EP 4 threads, unless a source is cited. Scripts: `scratchpad/{bench,fleval,sil,mms,turbo,cmbench}.py`.

## Bottom line (brutal)

1. whisper-tiny is already about as fast as anything else: ~95-130 ms per detect, constant because it pads to 30 s. ECAPA/AmberNet ONNX on CPU is 4-6x SLOWER on a 30 s window (470-710 ms). Only at 3-5 s windows do they match or beat it (60-120 ms).
2. The only real speed win is ECAPA via Core ML on GPU/ANE (`ComputeUnit.ALL`): 55-60 ms model + ~20 ms numpy frontend for 30 s, ~2x better than whisper-tiny. Not worth a new stack on speed alone.
3. Accuracy win is not ECAPA either. On FLEURS (my run, 80 clips/lang) ECAPA ~ whisper-tiny for Vietnamese (80/80 full clip), and ECAPA/AmberNet are WORSE on Indonesian (confuse with Javanese) and Cantonese (read as Vietnamese/Thai). Independent source says ECAPA/AmberNet also mislabel accented English as Dutch/Danish.
4. Best accuracy upgrade for small cost: whisper-base (~185 ms, 2x tiny) or silero lang95 (14 ms, 17 MB, MIT) as a cheap second opinion. whisper-small (~600 ms) is 80/80 on every language I tried at full clip. large-v3-turbo detect = ~2.8 s, no.
5. Every closed-set classifier returns a confident label on music. AmberNet said `war` (Waray) at 0.97-0.99 on music; whisper-tiny said `en` at 0.5-0.6. Whisper's low confidence is gateable; ECAPA/AmberNet confidence is not. Gate with CED sound tags / VAD first.
6. MMS-LID: reject. 5.1-5.5 s per 30 s clip (q8, 974 MB), CC-BY-NC.
7. Apple has no audio LID API I could find (SpeechAnalyzer needs a locale; NLLanguageRecognizer is text only). Not a candidate.

Recommendation: keep whisper-tiny, or move to whisper-base if the id/yue/3-5 s errors matter. If Vietnamese-vs-Cantonese/Thai routing matters, do not use ECAPA/AmberNet as the only LID.

## Coverage pass, GEM counts

| Sub-area | GEMs | Notes |
|---|---|---|
| (a) ECAPA/VoxLingua | 5 | CoreML port, GGUF with FLEURS number, accent-failure report, raw-audio ONNX, my FLEURS run |
| (b) MMS-LID / XLS-R | 3 | MMS paper table, my MMS latency, TitaNet paper XLS-R/ECAPA table |
| (c) whisper-family tricks | 4 | my size sweep + turbo cost, whisper-base LID-only ONNX, faster-whisper detect_language API, whisper paper LID claim |
| (d) ONNX/CoreML packages | 5 | ambernet.onnx, ECAPA CoreML, ECAPA ONNX raw-audio, silero95 ONNX, transcribe.cpp |

## My measurements (MEASURED, M1 Pro)

### Accuracy: FLEURS test, first 80 utterances per language, crop from clip start, top-1 over each model's own label set

Counts out of 80. "full" = whole clip up to 30 s (FLEURS clips ~8-14 s). Cantonese (yue): zh also counted correct for models without a yue label.

| Model | crop | vi | zh | th | id | en | yue |
|---|---|---|---|---|---|---|---|
| whisper-tiny | 3s | 63 | 62 | 60 | 51 | 79 | 61 |
| whisper-tiny | 5s | 77 | 76 | 74 | 70 | 80 | 68 |
| whisper-tiny | full | 80 | 80 | 79 | 74 | 80 | 76 |
| whisper-base | 3s | 74 | 59 | 58 | 60 | 79 | 58 |
| whisper-base | 5s | 80 | 76 | 71 | 76 | 80 | 72 |
| whisper-base | full | 80 | 80 | 78 | 79 | 80 | 78 |
| whisper-small | 3s | 75 | 62 | 70 | 62 | 80 | 74 |
| whisper-small | 5s | 80 | 77 | 77 | 75 | 80 | 79 |
| whisper-small | full | 80 | 80 | 80 | 80 | 80 | 80 |
| AmberNet (NVIDIA, ONNX) | 3s | 72 | 58 | 69 | 63 | 66 | 13 |
| AmberNet | 5s | 79 | 74 | 76 | 66 | 76 | 19 |
| AmberNet | full | 80 | 80 | 78 | 62 | 78 | 18 |
| ECAPA VoxLingua107 (ONNX) | 3s | 71 | 56 | 63 | 48 | 76 | 14 |
| ECAPA | 5s | 79 | 74 | 71 | 56 | 79 | 27 |
| ECAPA | full | 80 | 80 | 78 | 59 | 80 | 44 |

Silero lang95 ONNX (separate run, 40 clips/lang, crops 5 s and 15 s): 5 s: vi 39/40, zh 36, th 33, id 29, en 33, yue(zh-*) 31; 15 s: vi 40, zh 40, th 35, id 35, en 37, yue 35.

Key confusions (5 s crop):
- Vietnamese is easy for everyone at >=5 s. At 3 s whisper-tiny drops to 63/80 vs ECAPA/AmberNet 71-72/80 (this is the one place ECAPA beats tiny).
- Cantonese -> AmberNet: vi 33, th 17 of 80. ECAPA: th 24, vi 20 (5 s). Neither has a Cantonese label (also stated in transcribe.cpp docs). Danger if Vietnamese routing is keyed on LID: Cantonese audio can be sent to the Vietnamese ASR.
- Indonesian -> Javanese: ECAPA 21/80, AmberNet 10/80 at 5 s.
- Thai -> Lao: ECAPA 8, AmberNet 4.
- Mandarin -> Vietnamese 2/80 for both ECAPA and AmberNet at 5 s (small).
- Caveats: one run, first-N slice not random, FLEURS is read speech (clean), 80 clips = +-4 pts noise. Not noisy/real-world.

### Speed per clip (ms), MEASURED, best of 3 after warmup

| Model | 30 s | 5 s | 3 s | Notes |
|---|---|---|---|---|
| whisper-tiny int8 faster-whisper | 98-145 | 92-137 | 96-130 | constant, pads to 30 s |
| whisper-small int8 | ~582-612 | | | |
| whisper-large-v3-turbo int8 | 2768-2925 | | | |
| AmberNet ONNX fp32, 29 M | 470-710 | 95-123 | 58-71 | scales with length; 117 MB |
| ECAPA VoxLingua ONNX (christopherthompson81), 21 M | 470-690 | 97-118 | 60-104 | 87 MB, raw audio in |
| ECAPA Core ML (aufklarer), ComputeUnit.ALL | 55-61 (+20 frontend) | 22-38 | 6-11 | 41 MB |
| ECAPA Core ML CPU_ONLY | 73-77 | 17-21 | 13-14 | |
| ECAPA Core ML CPU_AND_NE | 374-496 | 62-85 | 46-52 | ANE path slower; ALL means GPU |
| Silero lang95 ONNX, ~4 M | ~15 avg (5-15 s inputs) | | | 17 MB |
| MMS-LID-126 ONNX q8 (Xenova), 1 B | 5060-5460 | 594-599 | | 974 MB |

Frontend timings for the numpy Core ML frontend were noisy (3-165 ms), treat as ~10-20 ms typical.
Third-party claims, not re-measured: AmberNet card "10 s in 138 ms, 4 CPU threads" (consistent with mine roughly); transcribe.cpp ECAPA on M4 Max CPU F16 242 ms / 30 s, Metal 21 ms; aufklarer CoreML "3.7 ms warm" (not reproducible with 30 s input in my run; probably short input).

### Music / noise (60 s music clip `samples/music/LrM_Y39Gmhk.wav`, one sample only)
- whisper-tiny: `en` conf 0.51-0.61 (gateable)
- AmberNet: `war` conf 0.97-0.99 (not gateable)
- ECAPA: `gn` conf 0.38-0.73
- whisper-small: `en` 0.70; turbo: `en` 0.51
- Every classifier is closed-set, no non-speech class (stated by AmberNet card and transcribe.cpp docs). N=1 clip, weak evidence.

## Candidates

### 1. SpeechBrain lang-id-voxlingua107-ecapa (GEM)
- ECAPA-TDNN, 21 M params, Apache-2.0, 107 languages. HF `speechbrain/lang-id-voxlingua107-ecapa`, 116k downloads. https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
- Reported: 6.7% error on VoxLingua107 dev (1,609 clips, only 33 languages). Paper table (AmberNet/TitaNet paper): ECAPA 11.9% error for 0-5 s, 6.7% for 5-20 s, avg 7.5%. https://arxiv.org/abs/2210.15781
- FLEURS 15-language mean, first 5 s, F32 GGUF: 85.23% (CI 84.07-86.47). https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf
- Model card limitations: poor on foreign accents, children, small languages. Issue still open: Indian-accent English misclassified. https://github.com/speechbrain/speechbrain/issues/2826
- Independent: ECAPA and AmberNet 0.986 on FLEURS at 10 s but ~0.86 on AMI meeting windows (accented English read as Dutch/Danish); whisper-base 1.000 on AMI. https://huggingface.co/beshkenadze/whisper-base-language-id-onnx (HF card, a single author's test, UNVERIFIED beyond that, but matches speechbrain issue 2826).
- Python 3.14: speechbrain 1.1.2 is pure py3 wheel but pulls torch (torch 2.14.1 has cp314 macOS arm64 wheels). Easier: ONNX/CoreML ports below.
- Adoption: Kesha Voice Kit uses it for "audio language detection (107)". https://github.com/drakulavich/kesha-voice-kit/blob/main/docs/languages.md

Ports (all Apache-2.0):
- Core ML: `aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML`, 40.8 MiB, fp16, macOS 15+, input log-mel [1,frames,60], frames 10-3001, pure-numpy `frontend.py`, cosine 0.99997 vs PyTorch, 47 downloads, updated 2026-08-09, part of soniqo/speech-swift (1.2k stars, pushed 2026-10-14). https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML. Requires `coremltools==9.0` which has NO cp314 wheel (I used 3.12 and numpy<2). On 3.14 call MLModel through pyobjc or a Swift CLI.
- ONNX raw audio in, FBANK folded in as Conv1D: `christopherthompson81/voxlingua107-lid-onnx`, 83 MB fp32, no STFT op, parity 3e-11 to 6e-5, built for Vernacula (23 stars). Ran fine on onnxruntime 1.31 Python 3.14 in my test. https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx . Its own note: clips under 5 s are the noisy regime.
- ONNX needing external fbank: `beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx` (0 downloads; MEH).
- GGUF / C++: `handy-computer/transcribe.cpp` (2,048 stars, MIT, pushed 2026-10-16), docs/models/lang-id-voxlingua107-ecapa.md: Q8_0 27 MB, "no Cantonese label: classified as zh", min 500 ms of audio, scores first 30 s. M4 Max CPU 30 s: 242 ms F16; Metal 21 ms; Ryzen 4750U CPU 800 ms. https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/lang-id-voxlingua107-ecapa.md
- MLX / `vokra` GGUF: MEH, tiny repos (vokra 13 stars).
- TalTechNLP/voxlingua107-epaca-tdnn (536 downloads): original author's version of the same recipe, speechbrain checkpoint; MEH, no advantage found.

### 2. NVIDIA AmberNet / TitaNet-LID (GEM, license caveat)
- 28.99 M params, 107 languages (VoxLingua107 labels), no STFT op so loads on restricted runtimes. Paper: TitaNet-LID-3x5x1024 5.6% avg error VoxLingua107 (7.5% for 0-5 s); FLEURS fine-tune 88.2-93.8% macro acc. https://arxiv.org/abs/2210.15781. MMS paper cites AmberNet at 95.3% on VoxLingua dev (33 langs).
- Runnable repo: `surogate/ambernet-langid`, ONNX 117 MB, raw audio in, onnxruntime only, created 2026-08-13, 135 downloads, 1 like. License `other`: NVIDIA NGC terms (not clear for redistribution/commercial). The repo is a third-party re-upload. https://huggingface.co/surogate/ambernet-langid
- Same failure modes as ECAPA, plus music overconfidence in my test.
- NOISE: "langid_ambernet" NeMo original needs NeMo toolkit (heavy), no need.

### 3. Meta MMS-LID (126/256/512/1024/2048/4017) (MEH for this use)
- wav2vec2 1 B params, CC-BY-NC-4.0 (non-commercial). facebook/mms-lid-1024 350k downloads; -126 22k.
- MMS paper Table 7: 126-lang model FLEURS 97.5% (102 langs, in-domain), VoxLingua 33 langs 93.9%, BABEL 84.1%, VoxPopuli 87.3%; scaling to 4017 langs costs ~0.3 pt. Whisper 64.5% on same table (Whisper counted wrong on unsupported langs), SpeechBrain 93.3, AmberNet 95.3, XLS-R 94.3 (VoxLingua dev). https://arxiv.org/abs/2305.13516
- Speed: measured 5.1-5.5 s per 30 s clip on M1 Pro CPU with Xenova q8 ONNX (974 MB), ~0.6 s per 5 s. 40-50x slower than whisper-tiny. Not usable as a fast LID. Core ML twin exists (`brbousnguar/localingo-mms-lid-coreml`, not tested; 0 downloads).
- Accuracy ceiling is the best but at 1 B params; it could serve as a slow, high-accuracy verify step only.

### 4. XLS-R / wav2vec2 LID (MEH)
- XLS-R 300 M fine-tuned on VoxLingua107: 5.7% avg error, 4.7% with attentive pooling (0-5 s 9.1%, 5-20 s 5.0%). 300 M params, best short-clip accuracy among trained-on-VL models in the paper table. No maintained, packaged checkpoint found on HF; only task-specific (Nigerian, FLEURS-research) fine-tunes. https://arxiv.org/abs/2210.15781
- Verdict: nothing to install; MMS-LID is the packaged descendant.

### 5. Silero language classifier 95 (GEM for cheap second opinion)
- ~4 M params, 16-17 MB F32 (quantization breaks accuracy per GGUF card), MIT, 95 languages incl. vi, th, zh, zh-HK, zh-TW. ONNX copy: `deepghs/silero-lang95-onnx` (0 downloads, 2 yrs old, re-upload). Trained on <15 s audio. Upstream: snakers4/silero-vad wiki "Other Models". https://github.com/snakers4/silero-vad/wiki/Other-Models
- Measured 14.6 ms avg per call on M1 Pro CPU (5 s and 15 s inputs). vi 39/40 at 5 s, 40/40 at 15 s; weaker on id (29/40) and en (33/40 at 5 s).
- Upstream user complaint: "the accuracy is not so good" (silero-vad discussion #116). https://github.com/snakers4/silero-vad/discussions/116 . Output shape quirk: model emits an extra [8,58] group output and ORT warns on shapes; harmless.
- Weakness: Vietnamese strong, other languages mediocre. No published benchmark; I found none. The Silero repo is the "Other" license on the models repo (silero-models) vs MIT in the VAD repo: license ambiguity, check before shipping.

### 6. sherpa-onnx spoken language identification (MEH)
- It is whisper encoder + decoder (tiny/base/small…) run through onnxruntime; same accuracy as whisper-of-same-size, no speed-up. sherpa-onnx 15.3k stars, Apache-2.0, pushed 2026-10-15, PyPI wheels exist for cp314 incl. macOS universal2. Example: https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/spoken-language-identification.py . Benefit: it is already a dependency (CED, TitaNet), so whisper-tiny/base LID could run in sherpa-onnx and drop faster-whisper for LID only. Not benchmarked by me. `kouhxp/whisper-tiny-lid` has the int8 encoder/decoder in this format (0 downloads).

### 7. Whisper family tricks (GEM)
- Whisper LID = first decoded token's language logits. Cost = encoder on a 30 s padded window. So tiny/base/small cost is constant and small cost is ~6x tiny.
- Measured sizes above: tiny ~95 ms, base ~185 ms, small ~600 ms, large-v3-turbo ~2.8 s (int8 CPU, same encoder as large-v3). Turbo is not a speed trick; only better for ASR.
- LID-only Whisper-base ONNX (graph keeps encoder + 1 decoder step, log-mel inside, 185 MB fp32, MIT): `beshkenadze/whisper-base-language-id-onnx`. Reports FLEURS-19-lang 0.946 at 10 s / 0.999 at 30 s, AMI 1.000; 25-lang renormalised set. Single-author test. https://huggingface.co/beshkenadze/whisper-base-language-id-onnx . Same card: "Whisper tiny sent three non-Russian FLEURS triples to Russian".
- Constraining languages: compute `detect_language()` probs then renormalise over allowed set (faster-whisper maintainers' answer, https://github.com/SYSTRAN/faster-whisper/issues/1164). Restricting to {vi, en, zh, th, ...} is the free accuracy gain; ECAPA via transcribe.cpp has `--allow` too.
- faster-whisper API `detect_language(audio, vad_filter, language_detection_segments, language_detection_threshold)` returns top lang, prob, all probs. https://github.com/SYSTRAN/faster-whisper/issues/1245 . Multi-segment detection (`language_detection_segments`) costs N x encoder runs; good for 3-5 s clips and music (average over several windows).
- Whisper paper: best Whisper gets 80.3% on the 82 overlapping FLEURS languages (ceiling 80.4% over 102), "not competitive with supervised SOTA". https://cdn.openai.com/papers/whisper.pdf . Tiny is worse than best; paper chart not read for numbers.
- WhisperKit has `detectLanguage` and an open PR to expose full log-probs (argmaxinc/argmax-oss-swift PR #555, 2026-10-10). Repo 6.4k stars, MIT, pushed 2026-10-11. Core ML Whisper-tiny LID is available there but not benchmarked; Swift only.
- FluidAudio (3.1k stars, Apache-2.0): no LID model in README. NOISE for this question.

### 8. Apple (NOISE)
- SpeechAnalyzer/SpeechTranscriber take a locale; I found no audio-LID API in docs or wrappers. Kesha uses `NLLanguageRecognizer` on text only and ECAPA for audio. https://developer.apple.com/documentation/speech/speechanalyzer
- One cheap path: run ASR output (Parakeet/QUAN) through NLLanguageRecognizer. Untested here.

## Python 3.14 compatibility (checked against PyPI JSON today)
- onnxruntime 1.31.0: cp314 macOS arm64 wheel. Used here.
- sherpa-onnx 1.13.8: cp314 macOS universal2 wheel.
- faster-whisper: pulled and ran on 3.14 in my env.
- torch 2.14.1: cp314 macOS arm64 wheel. speechbrain 1.1.2: pure py3, but not needed with ONNX.
- coremltools 9.0: wheels up to cp313 only, no cp314. aufklarer CoreML card pins coremltools 9.0 + numpy 1.26.4 (needs 3.12 or a pyobjc/MLModel shim).

## Not checked / unresolved
- No benchmark of whisper-tiny vs ECAPA on noisy or music-heavy real video, only one music clip. Run on framesift sample set with ground-truth labels before deciding.
- Core ML ECAPA speed on 30 s measured only with the numpy frontend; no pyobjc route measured.
- Whisper-tiny LID FLEURS accuracy per language from the Whisper paper (appendix) not extracted; only my own run.
- aufklarer's "3.7 ms warm" and handy-computer numbers not independently reproduced beyond my table.
- Vietnamese tone confusion with Thai/Lao/Cantonese tested only on FLEURS read speech, 80 clips.
- Silero model licence for lang95 weights unclear.
- HN and Lobsters: no relevant threads found (searched). Reddit not called per instruction. Social-only claims: none used.

## Sources
- https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
- https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML
- https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf
- https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/lang-id-voxlingua107-ecapa.md
- https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx
- https://huggingface.co/beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx
- https://huggingface.co/surogate/ambernet-langid
- https://huggingface.co/beshkenadze/whisper-base-language-id-onnx
- https://huggingface.co/facebook/mms-lid-126 , https://huggingface.co/Xenova/mms-lid-126 , https://huggingface.co/brbousnguar/localingo-mms-lid-onnx
- https://huggingface.co/deepghs/silero-lang95-onnx , https://huggingface.co/cstr/silero-lid-lang95-GGUF , https://github.com/snakers4/silero-vad/wiki/Other-Models , https://github.com/snakers4/silero-vad/discussions/116
- https://arxiv.org/abs/2210.15781 (TitaNet-LID/AmberNet)
- https://arxiv.org/abs/2305.13516 (MMS)
- https://cdn.openai.com/papers/whisper.pdf
- https://github.com/speechbrain/speechbrain/issues/2826
- https://github.com/SYSTRAN/faster-whisper/issues/1164 , /issues/1245
- https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/spoken-language-identification.py
- https://github.com/argmaxinc/argmax-oss-swift (WhisperKit) issues/PR #555
- https://github.com/drakulavich/kesha-voice-kit/blob/main/docs/languages.md
- https://developer.apple.com/documentation/speech/speechanalyzer
- https://scholarcompass.net/en/tasks/spoken-language-identification/benchmarks/fleurs (FLEURS leaderboard: GeoLID 97.7, MMS-126 97.5, OWSM v4 medium 95.6, TitaNet 88.2)
- PyPI JSON for onnxruntime, sherpa-onnx, speechbrain, coremltools, torch (wheel tags, 2026-10-16)
- FLEURS data: https://huggingface.co/datasets/google/fleurs (parquet refs/convert/parquet, test split)
