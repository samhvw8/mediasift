# framesift model alternatives (2026-10-16)

Machine: M1 Pro 16 GB, macOS 27.0.1, Python 3.14. Ten gatherer reports in this folder
(`en-q1..q6`, `vi-q1`, `zh-q3q4`, `zhtw-asr-pipeline`, `en-social-reddit`, `en-code-github-x`) hold the sources.
**[measured here]** = run on this Mac; everything else is a published number on the chip named.
**[UNVERIFIED]** = social post or vendor self-report only.

## Bottom line

1. **Vietnamese: nothing replaces QUAN for bulk work.** No QUAN v1.1/v2 (0 issues, no roadmap); the only other
   Vietnamese EG2 reader (`Neeze/Giong-Flash`) has no WER. Untested cheap upgrades: QUAN's KenLM (author: FLEURS
   valid 6.70 -> 4.80, CV17 test 9.64 -> 7.32) and a punctuation pass with a word-identity guard.
2. **EmbeddingGemma 2 audio is a speech model.** Zero-shot ESC-50 24%; no tagging / speaker / LID fine-tunes; the
   open-vocabulary English reader scores 18.08% test-clean; int8 ONNX audio embeddings keep only 30% top-1 neighbours.
3. **Biggest non-Vietnamese ASR speed-up: FluidAudio** (Swift, ANE) parakeet-tdt-0.6b-v3 at 129-210x on M4 Pro vs
   parakeet-mlx 25x on M1 Pro; `ultra` variant FLEURS-24 14.81 -> 11.67. Swift-only subprocess; no M1 Pro number;
   its build hung here on a dependency; concurrent calls must be serialized (#661).
4. **Apple SoundAnalysis beat sherpa CED-small on this Mac [measured here]:** ~310x realtime vs ~25-37x; ESC-50 top-1
   75% vs 67.5%. 303 classes, no male / female / child, macOS only.
5. **Diarization:** keep TitaNet + greedy (voice bench 1.00 / 8 of 8; Senko 0.958 / 6 of 8 [measured here]); try
   Nemotron-3-Diarization (openmdw, 100M, MPS via transformers) on real many-speaker audio.
6. **Language ID:** keep whisper-tiny [measured here: 95-130 ms, Vietnamese 80/80 on full clips]; renormalise over
   the routed languages. ECAPA / AmberNet misroute Cantonese to `vi`.

## Vietnamese ASR candidates

| Candidate | License | Speed (chip) | Accuracy | Notes |
|---|---|---|---|---|
| QUAN + KenLM | Gemma | +15 ms / utterance CPU | FLEURS valid 6.70 -> 4.80 (author) | needs `kenlm` built from source |
| nvidia/parakeet-ctc-0.6b-Vietnamese | NVIDIA Open Model (listed 3 ways: conflict) | Core ML 36.9x on M4 Pro | blind avg 9.30; vs QUAN: FLEURS 6.86/6.75, VIVOS 5.96/8.15, ViMD 11.02/9.05 | casing + punctuation, some English words; ONNX int8 via `onnx-asr` |
| Qwen3-ASR-1.7B | Apache-2.0 | ~13x M4 Pro, 9.6x M5 Air | FLEURS vi 5.55 | best English retention; pin the language (mlx-audio #928 loses most of a long file otherwise) |
| Qwen3-ASR-0.6B | Apache-2.0 | 34-55x M4 Pro | FLEURS vi 8.52 | |
| hynt Zipformer 30M/150M | cc-by-nc-nd | 12 s in 0.3 s CPU | VLSP2025 8.10/6.20 | license rules it out |

Code-switching: Vietnamese-tuned models lose English terms (ViMedCSS: Whisper-L-v3 46.69 vs PhoWhisper-L 55.05).
Punctuation restorers (chatofy PR #192, lowercase FLEURS-vi): Dewpoint mmBERT int8 (MIT, case/punct F1
0.895/0.820, 0 downloads, new), vibert-capu (cc-by-sa, 0.875/0.814, but rewrites numbers / merges words): any
restorer needs a guard that rejects output whose lowercase word list differs from the input.

## Sound tagging and diarization

| Candidate | Speed | Accuracy | Notes |
|---|---|---|---|
| Apple SoundAnalysis | 180 s in 0.58 s [measured here] | ESC-50 75% / 89% top-1/3 vs CED 67.5 / 87% [measured here] | pyobjc; different (303-class) taxonomy |
| EfficientAT mn10_as (Core ML) | 1.97 ms / 5 s window [measured here] | 0% on 3 s windows | rejected |
| Nemotron-3-Diarization | Core ML 546-904x M5 Pro; MPS ~44x M5 | AMI SDM 11.1 vs Sortformer 21.4 | up to 8 speakers |
| speakrs (Rust) | 529x M4 Pro | VoxConverse 7.1 | CLI |

## Shortlist to benchmark in framesift

1. Apple SoundAnalysis as the macOS tagger (keep CED for voice() and Linux).
2. QUAN + KenLM, then a word-guarded punctuation pass.
3. parakeet-ctc-0.6b-Vietnamese (ONNX int8) vs QUAN on the 47 min dub.
4. Qwen3-ASR-1.7B, language pinned, as an opt-in accurate / code-switching mode.
5. FluidAudio Parakeet v3 / ultra vs parakeet-mlx for non-Vietnamese (serialize calls).
6. Nemotron-3-Diarization vs TitaNet + greedy on many-speaker audio.

## Checked and rejected

EG2 for tagging / speakers / LID; EG2 open-vocab English (18.08%); EG2 int8 ONNX; Apple SpeechTranscriber (no
Vietnamese [measured here]); Cohere Transcribe vi (9.55%, 1.55x); Nemotron 3.5 streaming vi (11-13.6%); Moonshine
tiny-vi (13.2%); Dolphin; Fun-ASR-MLT-Nano; SenseVoice / FireRedASR / Kimi / Step / Voxtral / Canary / Phonon /
Granite (no Vietnamese); Breeze-ASR-26 (1.3x); Zipformer (cc-by-nc-nd); ChunkFormer (cc-by-nc); PhoWhisper (stale);
Zalo / FPT / Viettel / VinBigData (no open weights); MMS-LID (5 s/call, NC); AmberNet; Senko (no 3.14); DiariZen (NC);
Sortformer; EfficientAT; YAMNet-CoreML; Dasheng 7B; onnxruntime / sherpa CoreML EP (slower than CPU); WhisperKit
(duplicates mlx-whisper); whisper.cpp Core ML (encoder only); candle / mlx-swift / parakeet-rs (no ANE).

## Unresolved

No M1 Pro numbers for FluidAudio, Qwen3-ASR, parakeet-vi, Nemotron-3, speakrs. Python 3.14 wheels unconfirmed for
mlx-audio, mlx-qwen3-asr, onnx-asr. parakeet-vi license and punctuation quality. X coverage partial (timeouts).
