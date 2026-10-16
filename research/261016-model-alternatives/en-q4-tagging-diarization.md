# EN gatherer: sound tagging + diarization on Core ML / ANE (2026-10-16)

Saved by the lead from the gatherer's returned message. The gatherer ran local measurements on this M1 Pro (scripts in the session scratchpad: sa/cmp.py, sk/bench.py, eat/c5.py). It also cloned oss/EfficientAT and oss/FluidAudio into ~/workspace/oss/.

## Recommendation
1. Tagging: Apple SoundAnalysis (SNClassifySoundRequest, SNClassifierIdentifierVersion1) as the macOS tagger. Faster than CED-small and slightly more accurate on ESC-50. No male/female/child voice classes, so keep CED for voice() and for Linux/CUDA.
2. Diarization: keep TitaNet + greedy for now (beat Senko on framesift's voice bench). Real upgrades are Nemotron-3-Diarization CoreML or speakrs, both Swift/Rust only, not run on M1 Pro.
3. Skip EfficientAT Core ML: fast on ANE but less accurate than CED; 3 s windows fail.

## Tagging
| Candidate | Result |
|---|---|
| Apple SoundAnalysis v1 (303 classes, 3 s window, 1.5 s hop) | M1 Pro measured: 180 s audio in 0.58 s (~310x RT); 5 s ESC-50 clip ~35 ms. 200 held-out ESC-50 clips with bench/sense/real_sound.py accept-words: 75% top-1 / 89% top-3 vs CED 67.5% / 87%; 7.8 s vs 15.6 s total. Python 3.14 via pyobjc-framework-SoundAnalysis; observer needs protocols=[objc.protocolNamed("SNResultsObserving")] else addRequest -> invalidArgument. Hardware (ANE/GPU/CPU) undocumented. |
| EfficientAT mn10_as (own Core ML conversion) | 1.97 ms per 5 s window ComputeUnit.ALL vs 236 ms torch CPU; 4.9M params, AudioSet mAP 47.1 (dymn10 47.7); ESC-50 200 clips 61.5% top-1 / 80% top-3; 3 s windows (300 frames) 0% top-1, cause unknown. fp16 vs fp32 top-1 agree 9/12. MIT, 367 stars, last push 2024-11. https://github.com/fschmid56/EfficientAT |
| CED (current) | mAP tiny 48.1, mini 49.0, small 49.6, base 50.0. No Core ML port on HF. |
| YAMNet-CoreML (Yehor) | hobby port, 31 downloads |
| Dasheng (mispeech/dasheng-base) | SSL encoder, not a ready tagger |
| BEATs, M2D, AST, PANNs, CLAP | no Core ML ports found |

## Diarization
| Candidate | DER + set | Speed | License | Python |
|---|---|---|---|---|
| Nemotron-3-Diarization CoreML (FluidInference/nemotron-3-diarization-coreml, 2026-09-23) | AMI MHM 9.36-9.75 (16 test meetings, collar 0); NVIDIA card 9.25 offline; AMI SDM 11.1 vs Sortformer v2.1 21.4; up to 8 speakers | M5 Pro: fast128 546x, offline 904x | openmdw-1.1, 100M, 190 MB (95 MB W8A8) | Swift via FluidAudio |
| speakrs (pyannote community-1 in Rust, CoreML) https://github.com/avencera/speakrs | VoxConverse dev 7.1 (216 files) vs pyannote MPS 7.2; AMI-IHM 17.0 | M4 Pro 529x (pyannote 24x) | Apache-2.0, 135 stars, v0.5.0 2026-07-07 | Rust crate/CLI |
| FluidAudio offline (community-1 seg + WeSpeaker + VBx) | AMI SDM 10.6 avg; FLEURS clips 15.07 | M2: ~323x AMI | CC-BY-4.0 models; repo 3,102 stars | Swift/CLI |
| Senko 0.2.1 (CAM++ + pyannote seg, CoreML) | OpenBench M3: VoxConverse 13.9, AMI-IHM 27.5, AISHELL-4 13.6 | M1 Pro measured: 240 s in 0.29 s, first load 22-43 s | MIT, 310 stars | cp311 wheel only, requires_python <3.14 |
| Sortformer / LS-EEND (FluidAudio) | AMI SDM 31.7 / 20.7 | 127x / 74.5x M2 | | Swift |
| DiariZen-large-s80-v2 | VoxConverse 9.1, AMI-SDM 13.9 | heavy WavLM | CC-BY-NC-4.0 | PyTorch |
| Argmax SpeakerKit | VoxConverse dev 7.8, AMI-IHM 18.0 (speakrs table) | M4 Pro 312-814x | commercial/Swift | |

Senko vs TitaNet greedy on bench/sense/voices.py (8 LibriSpeech dialogues, seed 11, true turn spans given): TitaNet agreement 1.00, right count 8/8, 2.52 s; Senko 0.958, 6/8, 1.08 s.

FluidAudio docs: "offline" Sortformer 56.7% DER on AMI vs 26.4% streaming — avoid. FluidAudio SwiftPM build hung downloading the NemoTextProcessing binary target on this Mac.

Sources: https://github.com/FluidInference/FluidAudio , https://raw.githubusercontent.com/FluidInference/FluidAudio/main/Documentation/Benchmarks.md , https://huggingface.co/FluidInference/nemotron-3-diarization-coreml , https://huggingface.co/nvidia/Nemotron-3-Diarization , https://github.com/narcotic-sh/senko , https://soniqo.audio/benchmarks/diarization , https://github.com/BUTSpeechFIT/DiariZen , https://github.com/RicherMans/CED , https://developer.apple.com/documentation/soundanalysis , https://pyobjc.readthedocs.io/en/latest/apinotes/SoundAnalysis.html , https://news.ycombinator.com/item?id=48282551
