# Q1: Vietnamese ASR + punctuation alternatives to QUAN-Transcribe-1.0 (EN gatherer, 2026-10-16)

Tags: GEM = number/incident/workaround on a page I fetched. MEH = real but thin. NOISE = skip.
Not done: Reddit (assigned elsewhere), `gh search` (banned). Cohere Transcribe card, nmcuong/parakeet-tdt-vi card: 401, unread.

## TL;DR (brutal)
1. Only open model found that outputs punctuation+casing AND is trained on Vietnamese-English code-switch: **nvidia/parakeet-ctc-0.6b-Vietnamese** (600M FastConformer CTC, NeMo). Its own averages: blind-set 9.30% WER, in-domain 9.73% (QUAN avg 6.87% on different sets, not comparable; no head-to-head exists). Apple path = CoreML port (36.9x RT on M4 Pro, unverified accuracy) or `onnx-asr` int8 ONNX. License nvidia-open-model-license (commercial OK), gated=false.
2. No QUAN v1.1/v2, no Vietnamese EG2 fine-tune by anyone else after 2 days. Author (@stablequan) promised recipe repo; derivatives are Norwegian, Hindi, Cantonese, Bashkir, English only. QUAN issue tracker: 1 open PR (OpenAI-compatible server).
3. Cheapest fix to QUAN's lowercase/no-punct output: bolt on a restorer. Best measured on ASR-like Vietnamese input = **Dewpoint mmBERT** (MIT, 79 langs, vi included): case F1 0.895 / punct F1 0.820 on FLEURS-vi, 1.2 GB RSS, p95 124 ms CPU. vibert-capu-onnx int8 is smaller (299 MB) but worse on speech and rewrites number words.
4. Code-switching: nobody solved it. QUAN drops English names; the only Vietnamese CS benchmark (ViMedCSS) shows ALL models bad (CS-WER 46-68%). Best CS-WER there = Whisper-Large-v3 46.69%, not PhoWhisper (55.05%). Your mlx-whisper large-v3-turbo choice for English-heavy content is defensible.

## (a) Vietnamese ASR models

| Model | Size / license | Numbers (test set) | Speed / HW | Python 3.14 / run path | Tag |
|---|---|---|---|---|---|
| nvidia/parakeet-ctc-0.6b-Vietnamese | 600M, nvidia-open-model-license, 92 likes, 518 dl, lastMod 2026-02-07 | Blind avg 9.30 (Giga2 11.23, VLSP21-T2 8.99, ViMD 11.02, VIVOS 5.96); in-domain avg 9.73 (MCV20 8.58, FOSD 9.67, LSVSC 5.15, FLEURS 6.86, VLSP21-T1 13.60, VietMed 14.52). WER excludes PnC errors. Trained 2000h, PnC labels from Qwen3 (FLEURS gold). 4-gram KenLM shipped. | CoreML port (leakless): 36.9x RT, 30 s in 0.81 s, 1.23 GB peak, M4 Pro (not M1 Pro). 15 s chunks, 2 s overlap | NeMo = Linux/Nvidia only per card (Ampere/Hopper/Volta/Blackwell). 3060 = Ampere OK. Mac: CoreML mlmodelc (250 MB encoder) or onnx-asr ONNX (fp32+int8, OpenVoiceOS) or HF-transformers `ParakeetForCTC` bf16 (giangndm, no remote code) | GEM |
| QUAN-Transcribe-1.0 (baseline) | 356M, Gemma ToU, 207 dl, 46 likes | Author table avg 6.87 (bud500 1.62, cv17 10.19, fleurs 6.75, giga2 12.34, lsvsc 5.48, phoaudiobook 1.35, vimd 9.05, VIVOS 8.15). + KenLM: FLEURS test 7.10 -> 5.03, CV17 test 9.64 -> 7.32 (GitHub README) | CoreML: 268x RT M3 Max fp16 GPU; ANE cannot compile; fp16 CPU-only loses accuracy (7.56 vs 6.70). Author says 200x RT on CPU (X, UNVERIFIED) | pip git install; CoreML pkg no torch | GEM |
| QUAN 4-gram KenLM (lm/vi_4gram.bin, 91 MB) | in both repos | lowers WER ~2 pts abs (CoreML card: 6.70 -> 4.80 FLEURS valid, beam 32, +15 ms/utt CPU) | needs kenlm build (C++ compiler) | **Untested by us: could cut the 5.45% on dub set. Try first.** | GEM |
| hynt/Zipformer-30M-RNNT-6000h | 30M, **cc-by-nc-nd-4.0**, 4507 dl | WER: VLSP20-T1 12.29 (ChunkFormer 14.09, PhoWhisper-L 13.75), VLSP23 pub 10.40/priv 11.10, VLSP25 pub 7.97/priv 8.10, GigaSpeech2 7.56 (VietASR-68M-70kh 6.88). Lowercase, no PnC. | "12 s clip in 0.3 s on CPU" (author claim) | k2/icefall; no sherpa-onnx export seen. NC-ND license kills redistribution | MEH |
| khanhld/chunkformer-ctc-large-vie | 110M, cc-by-nc-4.0, 488 dl, lastMod 2025-10-08 | CommonVoice 6.66, VIVOS 4.18, VLSP T1 14.09 (self-reported, normalized text). In PhoASR paper: N-WER 6.89 but O-WER 32.43 (no PnC) | long-form (hours) native | `pip install chunkformer`; NC license | MEH |
| vinai/PhoWhisper-large | 1.55B, BSD-3, 259 stars, repo last push 2024-11-12 (dead) | QUAN table avg 7.21; PhoASR paper PhoWhisper-small O-WER 33.9 vs N-WER 8.97 = bad PnC | 197x RT H100. CT2/WhisperKit-CoreML/OpenVINO ports exist (aoiandroid, thoaibuiic, jermalk) | faster-whisper works | MEH |
| Qwen3-ASR-1.7B / 0.6B | Apache-2.0 | QUAN table avg 7.01 (1.7B), 9.81 (0.6B); ViMD zero-shot 11.56 -> 8.59 full FT (duclvQ). Other community Qwen3-ASR vi LoRAs: Toan2408 (VIVOS+CV+FOSD, own split 4.15 -> 1.92 target) | GGUF ports exist (sakasegawa); 224x RT H100 | mlx-audio / GGUF; LLM decoder, can take context prompt with names/terms | MEH (prompt-with-terms is a real code-switch lever, untested) |
| Qualcomm PhoASR (EACL 2026 Findings, arXiv 2603.14779) | 500h dataset pipeline + PnC bartpho-punc-cap, BSD-3-Clear, 1 star | whisper-small-3100h: O-WER 11.70, N-WER 8.20 (own refined test set, 8 corpora) | n/a | **Only the data pipeline + `bartpho-punc-cap` ckpt released (GitHub release v1.0, 2026-08-18). ASR weights NOT found.** | MEH |
| nvidia/nemotron-3.5-asr-streaming-0.6b | 600M RNNT, OpenMDW-1.1, 40 locales, PnC native | FLEURS vi 11.18% (1.12 s chunk, LangID) up to 13.41% (80 ms) | H100 numbers only | NeMo | MEH (worse than QUAN, FLEURS 6.75) |
| Smart Systems & Devices paper: VietASR-68M + PnC in vocab | 68M RNN-T, Dollya 1000h data | VIVOS 6.41%, punct micro-F1 0.94 | 380 ms encoder CPU / 321 ms NPU for 30 s on QCS8550 | **no weights found** | MEH |
| QUAN ONNX int8 (braitoli) | 378 MB | FLEURS first 100 utt: 10.91 vs fp32 10.68; no-digit 7.23 vs 6.84 | 0.37 s per 10 s on 1 thread Cortex-X925; CoreML EP untested | onnxruntime | MEH |
| Whisper vi fine-tunes (kelvinbksoh lyrics, ThanThoai9x, etc.) | misc | no credible numbers | | | NOISE |
| Zalo AI, FPT, Viettel, VinBigData open ASR weights | | none found on HF in 2026 searches; Zalo's ChunkFormer is the khanhld model; FPT/VBD/Viettel only datasets (VLSP 100h, FOSD) and Viettel LLMs | | | NOISE |
| sherpa-onnx Vietnamese ASR zipformer | | no official k2-fsa vi ASR model. Multilingual `nytopop/zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10-320ms` (17 dl, streaming) exists; sherpa-onnx has vi TTS only | | | MEH |

### Leaderboards / public numbers with test set
- QUAN author table (8 sets, same normalization, H100): above. Gemini-Flash-3.8 6.92, MAI-Transcribe-2.0 6.96, Cohere Transcribe 03-2026 7.01, Qwen3-ASR-1.7B 7.01, ElevenLabs Scribe v2 7.08. Self-run by the model author with a Claude-Code-rebuilt eval harness ("no benchmarking suite for Vietnamese", @stablequan on X, UNVERIFIED). GEM.
- NVIDIA parakeet-vi card: tables above. Different test mix (VLSP21, VietMed incl.) from QUAN's, so no ranking possible. GEM.
- PhoASR paper (ACL Findings EACL 2026): O-WER (with PnC) vs N-WER on 8 corpora. ChunkFormer N-WER 6.89 / O-WER 32.43. GEM.
- ChunkFormer card: PhoWhisper-large VIVOS 4.67, CV 8.14, VLSP-T1 13.75; Whisper-large-v3 8.81 / 15.45 / 20.41. GEM.
- No active Vietnamese leaderboard (paperswithcode links in ChunkFormer card are legacy). HF Open ASR leaderboard Vietnamese track not verified.

## (b) Punctuation + casing restoration (run after lowercase CTC)

Source for all comparative numbers below: quanganh208/chatofy PR #192, journal 261001 (own harness, CPU 4 threads, same lowercase unpunctuated input). Rulers: FLEURS-vi test (347, gold), ViCapPunc test (500, gold), 36 prod ASR sessions vs Scribe (weak ref). GEM.

| Model | Size / license | FLEURS-vi case / punct F1 | Prod-ASR case / punct F1 | RSS / p95 | Notes |
|---|---|---|---|---|---|
| **Dewpoint mmBERT int8-emb** (valkayuh/dewpoint, HF created 2026-09-25, 0 dl, MIT, 79 langs) | mmBERT-base + optional XLM-R-large ensemble, ONNX included | 0.895 / 0.820 | 0.856 / 0.667 | 1.2 GB / 124 ms | Best on speech-like input; gazetteer for proper nouns; over-capitalizes after commas ("Anh, Xin"); 125/139 proper nouns |
| Dewpoint ensemble | | 0.899 / 0.803 | 0.869 / 0.689 | 3.2 GB / 390 ms | +0.006 not significant |
| vibert-capu (dragonSwing, cc-by-sa-4.0, OSCAR-2109, 5.6M samples) fp32 | viBERT, ONNX fp32 438 MB / **int8 110 MB** (welcomyou port) | 0.877 / 0.817 | 0.734 / 0.573 | 609 MB / 451 ms (int8: 299 MB / 131 ms) | Own card per-label F1: Upper .89, `.` .82, `,` .72, `?` .78, `:` .67. **Rewrites number words to digits and merges words in 6-29 rows/ruler**: breaks timestamp/word alignment |
| xlm-roberta-capu (dragonSwing) | XLM-R-base | 0.882 / 0.818 | 0.734 / 0.559 | 1.3 GB / 464 ms | same trait; Upper .89, `.` .82, `,` .72 (own card) |
| Horizon-Labs/punctuation-restoration-small (mmBERT-small 140M, Apache-2.0, created 2026-10-03, 278 dl) | ONNX int8 268 MB, transformers.js | vi FLORES lowercased punct F1 0.85 (micro over 5 marks), TED-vi 0.77; `.` .97 `,` .73 `?` .77 | not tested in PR | | **Punctuation only, no casing**; `base` sibling exists |
| tourmii/vietnamese-punc-cap-denorm-v1 (text2text, 380 dl) | seq2seq | no numbers | | | Also does ITN (hai mươi tám -> 28). Generative = can rewrite words. MEH |
| 1-800-BAD-CODE punct_cap_seg_47 | 71k dl, Apache-2.0 | **no Vietnamese in language list**; control xlmr variant scored 0.808/0.723 FLEURS | | 2 GB | NOISE for vi |
| capu-vi (VLSP fairseq) | no license | 0.724 / 0.426 | | | NOISE |
| Qualcomm bartpho-punc-cap | BSD-3-Clear, release v1.0 | no standalone numbers | | | generative, MEH |

Also: Smart-Systems paper punct micro-F1 0.94 (in-vocab, no weights); JointCapPunc (arXiv 2207.01312) dataset; no standalone Python 3.14 issues found for ONNX models (onnxruntime only).

Recommendation (b): if you want a restorer, Dewpoint mmBERT int8 via ONNX (CPU, ~124 ms/p95 per row) for casing+punct; guard against word edits (reject any output whose lowercase word list differs from input, as chatofy does). Alternative: just switch to parakeet-vi, which emits PnC natively and avoids a second model.

## (c) EmbeddingGemma 2 / QUAN fine-tunes

- HF `filter=base_model:qnguyen3/QUAN-Transcribe-1.0`: only CoreML (author, 43 dl) and ONNX int8 (braitoli, 0 dl). No v1.1/v2. GEM (negative).
- Other EG2 ASR derivatives found (none Vietnamese): mlboydaisuke EG2-Transcribe (CoreAI, LiteRT, Reader), Steveeeeeeen embeddinggemma2-en-{direct,stream}-1000h CTC, kuchris/embeddinggemma2asr-cantonese (20.8M reader on FROZEN encoder, 750-850x RT GPU), epoch-discovery/ep-hindi-asr (Apache-2.0, "recipe follows QUAN-Transcribe"), AigizK Bashkir, @AlfH Norwegian (10% vs 4% WER on FLEURS vs NB-Whisper, ~$15 H100; X, UNVERIFIED). GEM.
- HN 50069971 "We turned EmbeddingGemma 2 into a speech-to-text model" (432 pts, 97 comments, eg2.haku.sh): author Shershah1024 says "accuracy is good, not cloud range", plans more training + "ultra fast post processing layer". Shows the EG2 idea is spreading, not a Vietnamese release. GEM.
- X: @stablequan (author): $78 on RunPod, 8xH100 2.5 h; Gemini "failed mostly on dialects, especially central Vietnam"; will publish recipe repo/report; "12x faster than PhoWhisper large". UNVERIFIED (social). @tomaarsen (HF) 100 likes calls it "leading on Vietnamese ASR". No v1.1/v2 announcement seen up to 2026-10-16.
- GitHub qnguyen3/QUAN-Transcribe: 26 stars, pushed 2026-10-14, no license in metadata (model is Gemma ToU), only open item = PR #4 serving layer. GEM.

## (d) Code-switching (English names/terms inside Vietnamese)

- ViMedCSS (arXiv 2602.12911, LRLEC 2026): Vietnamese medical CS, test 3.38 h. Zero-shot test WER / CS-WER: PhoWhisper-Large 31.24 / 55.05; Whisper-Large-v3 34.47 / **46.69**; VietASR 27.56 / 58.38; PhoWhisper-Small 36.31 / 62.55. Hard split (rare CS terms) CS-WER best 50.10 (Whisper-L-v3). Adapters (LoRA, LID-guided AG) help. Conclusion: Vietnamese-tuned models are WORSE than multilingual Whisper on English terms; QUAN (syllable-token CTC, 14,917 Gemma tokens ~1 per Vietnamese syllable, no English subwords) is structurally weak here. GEM.
- parakeet-ctc-0.6b-Vietnamese card title "Unified Vietnamese-English CS": 4-gram LM file named `..._vi_en_dict_newStyleTone_PnC_vi_en_2.0.bin`, lexicon has English entries. Card gives no CS WER number (figure only as image). NVIDIA NIM card says >4,000h (HF card says 2,000h): inconsistency. MEH-GEM: best candidate, unmeasured on CS.
- TSPC (arXiv 2509.05983, v4 2026-03): two-stage phoneme-centric CS vi-en ASR; no weights found. MEH.
- VietPrism (arXiv 2609.30005): 993 h corpus with dialects + CS, open corpus; could be used as your own CS test. MEH.
- Qwen3-ASR accepts a prompt with names/terms (GGUF card, sakasegawa): practical lever for ChatGPT/AGI-style names. UNTESTED. MEH.
- Practical: QUAN drop of "ChatGPT/AGI" is a vocabulary limit, not tunable without retraining. Options: (1) route detected-Vietnamese-with-English-terms to whisper-turbo (current), (2) parakeet-vi, (3) hybrid: QUAN + alignment-diff against whisper only on low-confidence spans (idea, not researched).

## Per-sub-area GEM counts
- (a) ASR models: 9 GEM (parakeet-vi card, QUAN card, QUAN CoreML card, KenLM numbers, hynt zipformer card, chunkformer card, PhoASR paper, nemotron vi, QUAN ONNX). Met.
- (b) Punct/case: 6 GEM (chatofy benchmark, dragonSwing x2 cards, Horizon-Labs card+eval, welcomyou ONNX sizes, Smart Systems paper). Met.
- (c) EG2/QUAN fine-tunes: 4 GEM (HF base_model query negative, HN thread, derivatives list, GitHub repo/PR). Met (negative result).
- (d) Code-switching: 3 GEM (ViMedCSS, parakeet-vi card, Qwen3 prompt) + 2 MEH. Met, thin: no CS WER for parakeet-vi or QUAN.
Searches used ~25 tool calls; stopped when new queries returned only repeats.

## Unresolved
1. parakeet-vi vs QUAN on YOUR 47-min dub: no head-to-head. Needs a run (CoreML port on M1 Pro; check CoreML port accuracy, only argmax-agreement claimed by leakless).
2. Does parakeet-vi keep "ChatGPT"/"AGI" as English? Card shows no CS number.
3. QUAN + KenLM on the 47-min dub: WER change and whether LM hurts English names.
4. Dewpoint: how it handles QUAN's number-as-words output (ITN needed separately; chatofy has own ITN).
5. Verify onnx-asr Vietnamese parakeet runs on Python 3.14 (pure ONNX, likely OK, unchecked) and speed on M1 Pro.
6. HF Open ASR leaderboard has no verified Vietnamese track. Cohere Transcribe vi WER unread (gated).
7. When does author publish recipe/v1.1? No date given.

## Sources
- https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0/raw/main/README.md
- https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0-CoreML/raw/main/README.md
- https://huggingface.co/braitoli/QUAN-Transcribe-1.0-ONNX/raw/main/README.md
- https://github.com/qnguyen3/QUAN-Transcribe (+ issue/PR #4)
- https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese/raw/main/README.md
- https://build.nvidia.com/nvidia/parakeet-ctc-0_6b-vi/modelcard
- https://huggingface.co/leakless/parakeet-ctc-0.6b-Vietnamese-coreml/raw/main/README.md
- https://huggingface.co/OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx/raw/main/README.md
- https://huggingface.co/giangndm/parakeet-ctc-0.6b-vietnamese-bf16/raw/main/README.md
- https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- https://huggingface.co/khanhld/chunkformer-ctc-large-vie/raw/main/README.md
- https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h/raw/main/README.md
- https://arxiv.org/pdf/2603.14779 ; https://github.com/qualcomm-ai-research/PhoASR
- https://jst.vn/index.php/ssad/article/view/1348
- https://arxiv.org/html/2602.12911v2 (ViMedCSS)
- https://arxiv.org/abs/2509.05983 (TSPC); https://arxiv.org/html/2609.30005v1 (VietPrism)
- https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-full ; https://huggingface.co/Toan2408/qwen3-asr-1.7b-vietnamese-phonetic-lora
- https://github.com/quanganh208/chatofy/pull/192 (+ journal 261001-vi-punct-case-benchmark-dewpoint-mmbert-holds.md)
- https://huggingface.co/valkayuh/dewpoint
- https://huggingface.co/dragonSwing/vibert-capu ; https://huggingface.co/dragonSwing/xlm-roberta-capu ; https://huggingface.co/welcomyou/vibert-capu-onnx
- https://huggingface.co/Horizon-Labs/punctuation-restoration-small (+ eval/this_model.json)
- https://huggingface.co/1-800-BAD-CODE/punct_cap_seg_47_language
- https://huggingface.co/tourmii/vietnamese-punc-cap-denorm-v1
- https://news.ycombinator.com/item?id=50069971
- X: https://x.com/tomaarsen/status/2110278060717089203 ; https://x.com/AlfH/status/2110637017813967262 ; @stablequan replies 2026-10-14/15 (UNVERIFIED)
- https://huggingface.co/kuchris/embeddinggemma2asr-cantonese ; https://huggingface.co/epoch-discovery/ep-hindi-asr
- HF API queries (model lists) run 2026-10-16
