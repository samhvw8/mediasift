# Reddit slice (social-sourced, all UNVERIFIED), 2026-10-16

Saved by the lead from the slice's returned message. Parallax Reddit hit 429; slice used reddit-read.sh.

Threads opened: r/LocalLLaMA 1nf10ye (Parakeet v3 vs Whisper, M2 Pro 16 GB, 167 up), 1r56ak1 (mlx-qwen3-asr), 1rz920q (speech-swift), r/WhisperNotes 1wc8itp (Qwen3-ASR + SpeechAnalyzer benchmarks), r/LocalLLaMA 1n71s27 (FluidAudio SDK).

- No Reddit thread on QUAN or EG2 audio (too new). No Vietnamese ASR thread.
- Qwen3-ASR 0.6B fp16 M4 Pro (mlx-qwen3-asr): RTF ~0.08 (~12x); 4-bit 4.7x faster, LS test-clean 2.29 -> 2.72 (n=100); memory 1.2 GB (0.6B), 3.4 GB (1.7B). Counter-report: 37 GB memory and slower than Whisper for one user (1r56ak1).
- Parakeet v3: ~30 s per hour on M2 Pro (~120x RT) (1nf10ye).
- speech-swift: Qwen3-ASR 1.7B 8-bit 2.35% LS test-clean, 40x RT MLX; Parakeet TDT CoreML ANE 634 MB 2.74% (author self-promotion, 1rz920q).
- WhisperNotes 1.6.1: Qwen3-ASR 1.7B ~3.7x faster than Whisper large-v3-turbo on their Mac; lowest WER on 7/30 FLEURS langs. SpeechAnalyzer ~10x faster than turbo, weaker accuracy, no word timestamps (1wc8itp, 1nf10ye).
- pyannote + Parakeet "close to Deepgram Nova", weak on overlap and speaker counting (1nf10ye).
- FluidAudio bundles pyannote seg + WeSpeaker CoreML + Silero VAD + parakeet-tdt-v3 on ANE; >10k HF downloads/month (1n71s27).
- Silero VAD drift workaround: reset every ~10 s of speech.
- Audio tagging: no data (login wall).
