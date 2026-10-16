# EN gatherer: EmbeddingGemma 2 audio fine-tunes (2026-10-16)

Saved by the lead from the gatherer's returned message.

## Bottom line
- No usable English open-vocab EG2 reader: Steveeeeeeen/gemma-audio-asr-en-1000h-20261015 (Gemma-token CTC, 1 epoch YODAS 1000h) = 18.08% LS test-clean, 25.49% test-other.
- No EG2 fine-tunes for sound tagging, speaker ID or LID. Zero-shot ESC-50 = 24% (50 classes), FluidInference engineer, https://github.com/FluidInference/FluidUse/pull/31 ("speech queries work well; sound-event search is weak in the model itself").
- Official audio numbers: MSEB retrieval MRR@10 69.54, MAEB mean 49.39 (768-d), 46.92 (128-d). https://huggingface.co/google/embeddinggemma-2 . No per-task rows published.
- ~9 other-language CTC readers copy the QUAN recipe; 10-15% FLEURS WER typical, trail Whisper-class.

## (a) ASR readers (all dated 2026-10-13..16)
| id | lang | metric | license | format |
|---|---|---|---|---|
| mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader | EN fixed 200k vocab | 4.21 test-clean, 10.42 test-other, 3.81 on 54-min audiobook | apache-2.0 | PyTorch |
| mlboydaisuke/EmbeddingGemma-2-Transcribe-CoreAI | EN | 3.63 on 200 test-clean utts; 548x RT on M4 Max (encoder GPU, reader ANE) | apache-2.0 | Core AI .aimodel (macOS 27 only) |
| mlboydaisuke/EmbeddingGemma-2-Transcribe-LiteRT | EN | 10.8x Pixel 8a GPU | apache-2.0 | LiteRT |
| Steveeeeeeen/gemma-audio-asr-en-1000h-20261015 | EN open vocab | 18.08 / 25.49; YODAS 24.9 | apache-2.0 | safetensors |
| Steveeeeeeen/embeddinggemma2-en-{direct,stream}-1000h-20261014-{ctc,tdt,rnnt,parakeet_ctc,cohere} | EN experiments | no WER published | mixed | safetensors |
| kuchris/embeddinggemma2asr-cantonese | yue+EN words | CER 4.5 MDCC w/ LM (SenseVoice-S 7.9); SpiCE 34.1 (SenseVoice 28.0); 750-850x RTX 5070 Ti | HF non-commercial vs GitHub MIT | PyTorch |
| AigizK/EmbeddingGemma2-Bashkir-CV25 | ba | WER 9.63 CV25 | apache-2.0 | |
| mstkyvz/embeddinggemma2-turkish-stt | tr | 15.5 FLEURS w/ KenLM (26.3 greedy); 72x M4 Pro MPS | apache-2.0 | |
| ahdrage/norsk-asr | no | FLEURS 10.0 w/ LM (NB-Whisper large 4.2) | apache-2.0 | |
| nehsan/KothaGemma | bn | FLEURS 15.12 | cc-by-sa-4.0 | |
| ovindu-a/sinhala-gemma-asr, serendipitylabs/dulu-malayalam-asr, epoch-discovery/ep-hindi-asr | si/ml/hi | 14.7 / 44.1 / gated | | |

UNVERIFIED (X): @b1rkh0ff 6.43% LibriSpeech from 100h; Shahir app (eg2.haku.sh) ~5% WER, 13 s per hour on M3 Air, weights not public.

## (b) Tagging / zero-shot
- Zero-shot ESC-50 24% (above). matu79go/Reflex-2: logistic probe on 768-d embedding, 96.3% on a 10-class ESC-50 subset (raw zero-shot 35%), self-reported. https://huggingface.co/matu79go/Reflex-2
- MAEB paper https://arxiv.org/abs/2602.16008 : contrastive audio-text models excel at ESC50 (implying EG2 does not). No AudioSet mAP anywhere.
- jaichang/embeddinggemma2-global-birds-2000 / korean-birds-620: bird-call retrieval fine-tunes, no accuracy.

## (c) Speaker: none. Sibling Gemma 4 audio tower (rnagabh/gemma4-audio-encoder): same-speaker cos 0.656 vs different 0.622. Keep TitaNet.
## (d) LID: none. Indirect: speech->transcript Recall@1 0.938 across 14 FLEURS langs (https://huggingface.co/naijaml/embeddinggemma-2-african-speech-lora).

## Ports
- bdrgon/embeddinggemma2-audio-tower-onnx: int8 vs fp32 top-1 neighbour agreement 59% (tower), 30% (final embedding). Do not int8.
- FluidInference/embeddinggemma-2-coreml: Core ML audio encoder 561 MB fp16, 617x RT on M5 Pro GPU, 883 downloads.
- Luigi/embeddinggemma-2-570m-text-audio-litert-lm; lethalbeats 0.8bpw audio 57 MB (unmeasured).
