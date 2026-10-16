# VI gatherer: Vietnamese ASR + punctuation (searched in Vietnamese, 2026-10-16)

Saved by the lead from the gatherer's returned message (the gatherer did not write the file).

## Bottom line
1. `nvidia/parakeet-ctc-0.6b-Vietnamese` is the only candidate that adds punctuation, casing and some English words in one pass, with WER roughly equal to QUAN on shared sets. Costs: slower on Mac, license unclear, NeMo or CoreML/ONNX port toolchain.
2. Nothing beats QUAN on speed (156x M1 Pro). Parakeet-vi CoreML: 36.9x on M4 Pro (30 s clip). Zipformer-30M fast on CPU but cc-by-nc-nd, no punctuation.
3. Post-CTC punctuation: `dragonSwing/vibert-capu` (ONNX int8 at welcomyou/vibert-capu-onnx).
4. English words inside Vietnamese are unsolved; Whisper-large-v3 beats Vietnamese-specific models on English terms (ViMedCSS).
5. No open ASR weights from Zalo, Viettel, FPT, VinBigData. PhoWhisper repo not pushed since 2024-11.

GEMs: (a) ASR 8, (b) punctuation 6, (c) benchmarks 6, (d) code-switching 4. viblo/voz had nothing on QUAN or Parakeet-vi; Threads was down.

## Parakeet-ctc-0.6b-Vietnamese
- NeMo FastConformer-CTC 600M, "Unified Vietnamese-English CS". Card: https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese ; NIM card: https://build.nvidia.com/nvidia/parakeet-ctc-0_6b-vi/modelcard (upper/lower case, period, comma, question mark; punctuation labels from Qwen3; WER excludes punctuation/case).
- Training hours: HF card ~2,000 h, NIM card >4,000 h (unreconciled).
- License: HF "NVIDIA Open Model License"; NIM page "Community Model License"; CoreML port says CC-BY-4.0. Check.
- HF created 2026-01-15, modified 2026-02-07, 92 likes, 518 downloads.
- WER blind avg 9.30 (GigaSpeech2 11.23, VLSP2021-T2 8.99, ViMD 11.02, VIVOS 5.96); in-domain avg 9.73 (MCV-20 8.58, FOSD 9.67, LSVSC 5.15, FLEURS 6.86, VLSP2021-T1 13.60, VietMed 14.52).
- Indicative vs QUAN card (different normalization): FLEURS 6.86 vs 6.75; LSVSC 5.15 vs 5.48; VIVOS 5.96 vs 8.15; ViMD 11.02 vs 9.05; GigaSpeech2 11.23 vs 12.34; CV 8.58 (MCV-20) vs 10.19 (cv17).
- Code-switching: NVIDIA staff: "mainly focused on Vietnamese, it can transcribe some popular English words, but not all English audio" (https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese/discussions/3).
- Mac: CoreML fp16 258 MB (https://huggingface.co/leakless/parakeet-ctc-0.6b-Vietnamese-coreml), 36.9x RT, 1.23 GB peak on M4 Pro, for FluidAudio; only argmax agreement validated; sample output has a garbled word. ONNX fp32/int8: https://huggingface.co/OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx via `onnx-asr` (0.12.0, README lists Python 3.10-3.14; https://github.com/istupakov/onnx-asr). parakeet-mlx has ParakeetCTC class; loading this checkpoint untested.

## hynt Zipformer (VLSP 2025 winner)
- https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h (4,507 dl, modified 2026-09-09), https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h. License cc-by-nc-nd-4.0.
- Paper https://aclanthology.org/2025.vlsp-1.4.pdf. WER 30M/150M: VLSP2025-private 8.10/6.20; VLSP2023-public 10.40/6.59; GigaSpeech2 7.56/7.10 (PhoWhisper-Large 16.31/16.83/10.00).
- Speed: 30M 12 s clip in 0.3 s on HF basic CPU; 150M 1.0 s. Lowercase, no punctuation, English words as-is (VLSP2025 convention https://vlsp.org.vn/vlsp2025/eval/asr-ser).
- sherpa-onnx int8: https://huggingface.co/csukuangfj2/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09 . Reference app (Zipformer + vibert-capu ONNX + diarization, MIT, 173 stars, Windows): https://github.com/welcomyou/sherpa-vietnamese-asr

## ChunkFormer
- https://huggingface.co/khanhld/chunkformer-ctc-large-vie , 110M, cc-by-nc-4.0. Vivos 4.18, CV 6.66, VLSP-T1 14.09, avg 8.31. No punctuation. `pip install chunkformer` (py>=3.11).

## PhoASR-whisper-small (Qualcomm)
- https://arxiv.org/html/2603.14779 (EACL 2026 Findings). 3,100 h fine-tune, punctuation + case + word timestamps. O-WER 11.70 / N-WER 8.20 vs ChunkFormer 32.43 / 6.89 and PhoWhisper-small 33.90 / 8.97. Repo https://github.com/qualcomm-ai-research/PhoASR (BARTpho punc-cap checkpoint in releases).

## Others (MEH)
- Neeze/Giong-Flash: second EG2+CTC Vietnamese reader, Apache-2.0, no WER, 1 download. https://huggingface.co/Neeze/Giong-Flash
- Toan2408/qwen3-asr-1.7b-vietnamese LoRAs: no benchmarks.
- nmcuong/parakeet-tdt-0.6b-v3-vietnamese: gated, self-reported val WER 5.68 (UNVERIFIED).
- braitoli/QUAN-Transcribe-1.0-ONNX: FLEURS 100 utt: PyTorch 6.80, fp32 6.84, int8 7.23.
- Cohere Transcribe 03-2026: 7.01 avg in QUAN table; Facebook claims it degrades on mixed language (UNVERIFIED).

## Punctuation restoration
- dragonSwing/vibert-capu (cc-by-sa-4.0): F1 . 0.82, , 0.71, : 0.64, ? 0.78, Upper 0.89 (10k held-out). Fixes "năm g"->"5G". Needs repo gec_model.py. https://huggingface.co/dragonSwing/vibert-capu ; ONNX https://huggingface.co/welcomyou/vibert-capu-onnx
- dragonSwing/xlm-roberta-capu similar. PhoPunct (code only). BARTpho punc-cap (seq2seq, can change words).

## Code-switching
- TSPC (https://arxiv.org/abs/2509.05983) CS test WER: whisper-large-v3-turbo 31.60, PhoWhisper-base 27.90, Qwen3-ASR-0.6B 38.93, TSPC 19.06 (not released).
- ViMedCSS (https://arxiv.org/html/2602.12911v2): CS-WER/WER Whisper-L-v3 46.69/34.47, PhoWhisper-L 55.05/31.24, VietASR 58.38/27.56.

## Community on QUAN
- Facebook author post (UNVERIFIED): WER 6.87, 4,000x RT on H100, $78 training. X post 2026-10-14 https://x.com/stablequan/status/2110209860511932554 . No independent benchmark of QUAN found.
