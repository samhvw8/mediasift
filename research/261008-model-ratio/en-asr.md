# en-asr — ASR accuracy per resource (gathered 2026-10-08, EN sources)

Saved by the lead from the gatherer's returned findings. WER = leaderboard/vendor numbers; RTFx = H200 leaderboard, use only to rank.

## English
| model | params | license | avg WER / RTFx | timestamps | Mac / 3060 | tag |
|---|---|---|---|---|---|---|
| [granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | 470M | Apache-2.0 | 5.84 / 21,222 | CTC frames (80 ms) | MLX, Core ML, ONNX, GGUF; [Granite-MLX](https://github.com/kylehowells/Granite-MLX): 102-min lecture in 24.5 s Core ML / 37.1 s MLX on M1 Max (~165-250x RT), 1.7-2.0 GB; lowercase, no punctuation (separate punct model) | GEM |
| parakeet-tdt-0.6b-v3 (current) | 0.6B | CC-BY-4.0 | 5.66 / 6,076 | TDT | parakeet-mlx, 24x RT on our M1 Pro | baseline |
| Qwen3-ASR-1.7B | 2.0B | Apache-2.0 | 4.95 / 820 | en via aligner | mlx-audio | MEH (slow, AR) |
| cohere-transcribe-03-2026 | 2B | Apache-2.0 gated | 5.40 / 907 | **none** | mlx-audio | MEH |
| TheStageAI/thewhisper-large-v3-turbo | 0.8B | CC-BY-4.0 | 5.42 / 1,181 (OpenAI turbo 7.03 / 795) | Whisper | Core ML, ONNX; vi listed, no vi number | MEH |
| futo-org/asr4all-m | 55.6M | FUTO ethical-use | 7.64 / 47,741 | ? | en only | MEH |

Source: [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (updated 2026-10-02), [arXiv 2609.20104](https://arxiv.org/abs/2609.20104).

## Vietnamese
- **[VietAudio Vietnamese ASR Leaderboard](https://huggingface.co/spaces/VietAudio-team/Vietnamese-ASR-Leaderboard)** (natural speech, 7 sets, normalized WER; updated 2026-05-25): Zipformer-30M 6.77, PhoWhisper-medium 7.61, Qwen3-ASR-1.7B 9.84, parakeet-ctc-0.6b-vi 12.81 (its card says 9.3). No whisper-turbo or ChunkFormer. GEM: agrees with our ranking of Zipformer near the top.
- Qwen3-ASR-1.7B: FLEURS-vi 4.91 vs turbo 8.10 ([Whisper Notes](https://whispernotes.app/blog/qwen3-asr-vs-whisper)); no vi timestamps. vi dialect fine-tune [duclvQ/qwen3-asr-1.7b-vimd-full](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-full) (8-bit lossless).
- [BuzzASR/vietnamese](https://huggingface.co/BuzzASR/vietnamese): large-v3 fine-tune 1.55B, FLEURS 7.26 — MEH.
- Cohere Transcribe supports vi but no timestamps — disqualified.

## Recommendation from this track
English: test Granite TurboCTC vs parakeet-mlx on our English audio (speed win, same accuracy, needs punctuation step). Vietnamese: keep whisper turbo; Zipformer-30M is the cheap CPU option (non-commercial). Reddit/HN search was blocked (502/429): no community signal on new models.
