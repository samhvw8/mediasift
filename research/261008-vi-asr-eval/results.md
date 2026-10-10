# Vietnamese ASR eval (framesift)

Slice 600–1500 s (900 s) of "OpenAI Founder Breaks Down His Writing Process — Sam Altman [vi].mp4", 16 kHz mono.
Reference: the `dub.vi.srt` cues whose midpoint falls in the slice (2487 words; Vietnamese words are syllables, so WER is syllable-level).
Both sides are normalized the same way (NFC, lowercase, punctuation removed, diacritics kept); WER and CER come from jiwer.
Run on `gpu-box` (RTX 3060 12 GB, Xeon E5-2683 v4). RTF = 900 s audio / processing s (higher is faster); download excluded, load time listed separately.

| Model | WER | CER | RTF | Load s | Punctuation / casing | Timestamps | License | Notes |
|---|---|---|---|---|---|---|---|---|
| faster-whisper large-v3-turbo (CUDA fp16) | **2.37%** | **1.39%** | 29.6x (30.4 s) | 1.8 | yes / yes | segment (+word optional) | MIT | beam 5, no VAD, whole file at once. Fewest deletions (4) |
| ChunkFormer RNNT large vie (113M, CUDA fp32) | 2.61% | 1.78% | 27.6x (32.6 s) | 2.5 | no / lowercase | segment (split at silence) | CC-BY-4.0 | `endless_decode` (chunk 64, ctx 128/128); 7.3 GiB peak VRAM; the encoder takes about 1 s, RNNT search the rest |
| sherpa-onnx Zipformer vi 30M int8 (CPU) | 2.57% | 1.55% | 27.7x (32.5 s) | 1.1 | no / UPPERCASE | token-level | **CC-BY-NC-ND-4.0** (upstream hynt/Zipformer-30M-RNNT-6000h): non-commercial | 8 CPU threads, greedy, 45 chunks of about 20 s cut at quiet points. No GPU |
| PhoWhisper-large CT2 (kiendt port, CUDA fp16) | 3.98% | 2.69% | 11.4x (78.6 s) | 3.3 | partial / lowercase | segment (+word optional) | BSD-3 (vinai), port MIT | beam 5, no VAD. Hallucinated a 10-word sentence at the slice start; 2.6x slower than turbo |
| Parakeet CTC 0.6B vi (giangndm HF port, bf16) | 5.27% | 2.67% | **370x (2.4 s)** | 4.3 | yes (space before marks) / yes | CTC frames (HF `decode` does not expose them) | NVIDIA Open Model License | 21 chunks of about 45 s cut at quiet points; 1.2 GiB VRAM. fp32 gives the same errors (5.5%), so they are not caused by bf16 |

Most common errors (from `errors.py`):
- Every model misses English names and loanwords in the Vietnamese speech ("Joe Hudson" became Johansson, john hodgson, youtube or Johan; "pitch" became pick or pic; "PG" became pj or pri; "tagline").
- Common tone and diacritic confusions: đọng/động, nghĩ/nghỉ, sao/sau, sử/sự, khoăn/khoan. Every model except turbo drops the filler "ừ" (6–7 times each).
- Parakeet loses or merges characters at word and sentence edges ("với Joe" became "vớio", "Sự" became "ự", "hiếmao"). This produces many insertions and deletions, so its CER is close to PhoWhisper's while its WER is the highest.

Caveat: the audio is a synthetic TTS voice (edge-tts vi-VN-NamMinhNeural) over a ducked English original, so the absolute WER is optimistic compared with real Vietnamese speech. The ranking is what matters. On this clean read speech the top three (turbo, ChunkFormer, Zipformer) are within 0.25 points of each other, which is about 6 words out of 2487 and is noise at this sample size.

Files: `common.py` (reference, normalization, scoring), `run_fw.py`, `run_parakeet.py`, `run_chunkformer.py` (env `envs/chunkformer`, Python 3.11), `run_sherpa.py`, `chunks.py`, `errors.py`, `hyp/*.txt`, `res/*.json`, `reference.txt`.
