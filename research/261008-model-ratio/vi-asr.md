# Vietnamese ASR accuracy-per-resource, delta since 2026-10-01 (gather, Vietnamese queries)
Date: 2026-10-08. Builds on research/261001-models/vi-asr.md (not repeated). All WER = vendor/self-reported unless marked INDEPENDENT.

## TL;DR
- Biggest new signal: **Gipformer 1.5 (68M, MIT, 2026-08-21, card edited 2026-09-17)** + **hynt ZipFormer-150M (2026-09-09)**. Both small Zipformer-RNNT, sherpa-onnx, CPU-fast. No PnC, no whisper-turbo in any table.
- Only INDEPENDENT third-party run found (Techainer summo-app, FLEURS vi, 100 clips): gipformer 1.5 int8 WER 8.3% / CER 6.2%, RTF 0.017-0.020 on 8 Xeon threads (~50-60x RT on CPU). whisper-tiny 67.6% (irrelevant to turbo).
- G-Group's multi-domain table is the first one with podcast-like domains (tech reviews, finance commentary "dense in English loanwords and product names"). Nvidia Parakeet-vi-CTC does BADLY there (tech 48.69 vs gipformer1.5 27.49). Contradicts the vendor-claimed 5-9% numbers => domain shift matters; do NOT trust Parakeet 5.27% (TTS audio) for real speech.
- Qwen3-ASR-1.7B best on tech domain (27.95) and FLEURS (10.13) among non-Zipformer, but NO vi timestamps (see prior notes) and 1.7B.
- VLSP 2026 has NO ASR task (tasks: speaker recog edge, TTS, SQA, GEC, QA, MT). No new VLSP ASR results this cycle; latest = VLSP 2025 (Zipformer winner).
- Forums: site:viblo.asia returns only tag pages (no content); site:voz.vn returns only unrelated threads. Facebook = announcement reposts only. No community long-form/podcast benchmark exists in Vietnamese. Say so plainly.

## New / updated models (since 2026-10-01 or missed before)

### 1. g-group-ai-lab/gipformer1.5-68M-rnnt [GEM]
https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt ; code https://github.com/ggroup-ai-lab/gipformer (148 stars, pushed 2026-09-17)
- 68.6M params (renamed from 65M on 2026-09-17), Zipformer-RNNT, ONNX fp32+int8 (sherpa-onnx), PyTorch/icefall. License MIT (HF + repo README; repo has no LICENSE file in API). Created 2026-08-21; v1 gipformer-68M 2026-03-12 (1371 dl, 26 likes). 1.5: 259 dl, 14 likes.
- Training data size not disclosed (issue #1 "training details" open, unanswered). Pitched at call-center + 4 domains.
- WER (vendor; normalized: lowercase, no punct, numbers spoken): vi-asr-tech 27.49, edu 23.32, finance 22.34, pubadmin 14.82, vivos 4.25, CV 6.45, vlsp-t1 13.37, VietMed 19.23, MultiMED 19.17, LSVSC 8.97, Fleurs 12.65, ViMD 7.00, tele-medium 15.44. Same table: hynt-30M tech 29.77/finance 25.86; chunkformer-large tech 39.81/finance 34.31; PhoWhisper-large tech 42.94/finance 40.54; Parakeet-vi tech 48.69/finance 43.16; Qwen3-ASR-1.7B tech 27.95/finance 34.09/FLEURS 10.13.
  - Caveat: absolute domain WERs are high (22-49%) = hard YouTube-style data, noisy labels; relative ordering is the signal. Test sets small (tech n=508 utts, CC-BY-NC-4.0).
  - Whisper-large-v3(-turbo) NOT in table.
- INDEPENDENT (Techainer summo-app docs/benchmarks.md, 2026-09-22): FLEURS vi first 100 clips (21.3 min), Xeon Gold 6226R, sherpa-onnx int8: WER 8.3 / CER 6.2, RTF 0.020 (4 thr) / 0.017 (8 thr). 68M-v1: 8.5/6.7, RTF 0.023. They note 16/100 clips contain digits; excluding => 5.3% (number formatting inflates WER). Source https://github.com/Techainer/summo-app/blob/master/docs/benchmarks.md ; release https://github.com/Techainer/summo-app/releases/tag/v0.13.1 (Gipformer 1.5 made default VI, 2026-09-22).
- Speed: above RTF ~0.02 CPU => 1h podcast ~ 1-1.2 min on 8 threads. Model ~73 MB int8 (summo).
- Timestamps: not in card. Generic sherpa-onnx transducer gives token timestamps (known sherpa feature, UNVERIFIED for gipformer scripts). Their skill returns `chunks` (VAD <=20s chunks) => segment-level timestamps via VAD.
- Punctuation: NONE (uppercase/lowercase no punct per summo notes). Needs ViBERT-capu or similar.
- Long-form: VAD chunking <=20s, batch inference (gipformer ClawHub skill, via skills.lc page).
- Hotwords: sherpa-onnx supports hotwords/contextual biasing for transducer models only (Aho-Corasick) https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html => direct fix path for names. Not tested on gipformer.

### 2. hynt/ZipFormer-150M-CR-CTC-RNNT-6000h [GEM for accuracy on newer VLSP sets; license blocks commercial]
https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h (created/upd 2026-09-09; 458 dl, 10 likes)
- ~153M, CR-CTC+CTC+RNNT loss, 6000h public vi data (VLSP20/21/23, FPT, Viet-BUD500, FLEURS, VietMed, GigaSpeech2 sub, ViVoice, PhoAudiobook), icefall/k2; ONNX int8/fp32 + jit in repo. License CC-BY-NC-ND-4.0.
- WER (author table, mixed normalization): VLSP20-T1 13.50, VLSP23 pub 6.59 / priv 7.80, VLSP25 pub 6.69 / priv 6.20, GigaSpeech2 7.10. vs 30M: 12.29 / 10.40 / 11.10 / 7.97 / 8.10 / 7.56. vs PhoWhisper-large 13.75 / 16.83 / 17.10 / 16.14 / 16.31 / 10.00; ChunkFormer-110M 14.09/16.15/17.12/15.55/16.07/10.35. => ~2x lower WER than ChunkFormer/PhoWhisper on VLSP23/25.
- Speed: 12 s audio <0.3 s on RTX 3090; CPU (HF basic) 1.0 s. => ~12-40x RT.
- No PnC, no timestamps in card (sherpa token timestamps generic). Not in third-party tables yet (absent from gipformer's table).
- Fits 3060 trivially. Hynt also published Zipformer-30M streaming (2026-02).

### 3. Qualcomm-AI-Research/PhoASR-whisper-small (244M) [GEM for punctuation+word timestamps in a Whisper drop-in; MEH for WER]
https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small (2026-07-06; 853 dl) ; paper https://aclanthology.org/2026.findings-eacl.345/ ; https://arxiv.org/abs/2603.14779 ; code https://github.com/Qualcomm-AI-research/PhoASR
- whisper-small finetuned 3000h cleaned; outputs punctuation+capitalization+reliable timestamps (word-level via HF pipeline, chunk_length_s=30). License BSD-3-Clause-Clear + Qualcomm responsible-AI license; "research and educational purposes" => not clean for product.
- Paper O-WER (ready-to-use formatted text) on 8 test sets: PhoASR-small-3100h overall 11.7; 469h 12.46; ChunkFormer-large (25K h) 32.43; PhoWhisper-small 33.90; wav2vec2 51.15. N-WER (normalized): ChunkFormer 6.89 overall; PhoASR-small-469h 8.69 vs PhoWhisper-small 8.97 (paper Table 6). I.e. beats others on formatted output, not on normalized WER.
- Third-party (gipformer table): PhoASR-small tech 44.35, finance —, LSVSC 7.59 (best), ViMD 9.75 => decent, not top on domain speech.
- Paper notes weakness on "emerging or rare terminology" + Central accent (card).
- No whisper-turbo comparison. Whisper-small => fast, but accuracy below Zipformer on tech domain.

### 4. Nampfiev1995/nemotron35-encoder-only-vi-asr [MEH]
https://huggingface.co/Nampfiev1995/nemotron35-encoder-only-vi-asr (2026-09-25; 149 dl; Apache-2.0)
- Fine-tune of nvidia/nemotron-3.5-asr-streaming-0.6b, vi+en, HF Transformers RNNT (AutoModelForRNNT), `language="vi-VN"|"en-US"|"auto"`. Mixed vi/en target but NO code-switch eval.
- Self-reported (normalized): PW733-406 (private set) base 35.62 -> FT 13.65 (NeMo) / 14.82 (HF); LSVSC-1000 11.18 -> 7.62. English HeySQuAD 11.89-27.29 depending on set. Single-author, unverifiable data; interesting only as proof Nemotron-3.5 fine-tune reaches ~Parakeet-class on LSVSC. Same author has an odd "qwen3.8-27b ... vi-en-mixed-asr" audio-LLM [NOISE].

### 5. luvox-ai/vietasr-ssl-char42-canonical-vi-ctc [NOISE/MEH]
https://huggingface.co/luvox-ai/vietasr-ssl-char42-canonical-vi-ctc (2026-09-14; 15 dl). CTC, WER 7.03 on own BTN+Golden+LSVSC; 13.19 on PW733-406. Research checkpoint with hardcoded cluster paths. Skip.

### 6. Fine-tunes of Qwen3-ASR for vi [NOISE]
duclvQ/qwen3-asr-1.7b-vimd-{full,lora,qlora} (2026-10-04, ~10-33 dl), Toan2408/qwen3-asr-1.7b-vi-* (2026-08/10), quangkmhd/Qwen3-ASR-VI (GitHub, 2026-03, 0 stars): README reads LLM-generated template (placeholder `your-org`, claims "novel non-autoregressive forced aligner" for vi with no weights/eval seen). Do NOT rely on it for vi forced alignment [NOISE, UNVERIFIED].

### 7. Whisper-turbo variants [NOISE]
alphaedge-ai/whisper-large-v3-turbo-vie-{16384,32768} (2026-06): vocab-trimmed turbo, "should perform similarly", no WER. Only value: tiny memory saving. No vi-finetuned turbo with credible numbers exists (confirmed again).

### 8. PhoWhisper ports (activity, no new model)
- OpenVINO int8 small/medium/large by jermalk (2026-09-30; issue VinAIResearch/PhoWhisper#17: "5 synthetic clips, 85 words" PhoWhisper-large 1.2% on Intel NPU vs 3.5-4.7% stock Whisper => anecdote, n=85 words) https://github.com/VinAIResearch/PhoWhisper/issues/17
- WhisperKit CoreML medium (aoiandroid, 2026-09-30), Qualcomm QNN small (2026-08).
- No new VinAI model since 2024-11.

### 9. Other hits
- VietPrism corpus arXiv 2609.30005 (2026-09-24): 993h, 1,262 speakers, 8,388 real videos, ~half duration has natural vi-en code-switching, 5 dialects; ICASSP 2027 preprint; transcripts included. Useful EVAL SET for loanword/names; checked abstract only (release link/license not verified). https://arxiv.org/abs/2609.30005
- TSPC (arXiv 2509.05983 v4 2026-03-04): two-stage phoneme-centric vi-en code-switch ASR, WER 19.06% vs PhoWhisper-base; no released weights seen. [MEH]
- "Augmentation-Enhanced Whisper for Vietnamese-English Code-Switching" ISCIT 2026 (doi 10.1109/iscit70453.2026.11695357) — not fetched (paywalled).
- VietSuperSpeech (arXiv 2603.01894): 3-30s conversational dataset pseudo-labeled by Zipformer-30M — shows Zipformer-30M is used as the community pseudo-labeler [MEH].
- welcomyou/sherpa-vietnamese-asr (172 stars, pushed 2026-07-07; prior notes had 09-26 date for a different metric): practitioner Windows app, modes Zipformer 30M / 68M / ROVER ensemble, ViBERT-capu punctuation, CAM++/pyannote diarization, **hotwords for proper names + domain terms**, DNSMOS. Best "real practitioner pipeline" found. https://github.com/welcomyou/sherpa-vietnamese-asr
- mochiai.vn blog (2026-06-29) recommends PhoWhisper first; no numbers, generic [NOISE]. vexascribe "turbo degrades +4-10pp on Vietnamese" = approximate SEO table [NOISE]. congdongai.vn "medium for Vietnamese" [NOISE].

## Names / English loanwords inside Vietnamese
- Evidence: G-Group tech test set explicitly "dense in English loanwords and product names": Zipformer family and Qwen3-ASR-1.7B lead (27-30%), Whisper-FT/PhoWhisper (43-57%) and Parakeet-vi (48.7%) trail. Qwen3-ASR (multilingual LLM-ish) is the best non-Zipformer on tech => multilingual pretrained models handle English tokens better than vi-only CTC.
- Practitioner levers found:
  1. sherpa-onnx transducer hotwords (Aho-Corasick, BPE-tokenised hotwords, boost score) — Zipformer/gipformer qualify; welcomyou app ships it. A LeDinhLongAnh/STT_en_grammar repo has Vietnamese dashboard on tuning boost + measuring WER (small, unvalidated).
  2. Whisper `initial_prompt` with names/glossary (Vietnamese tutorial Evotek) — generic, only for Whisper-family.
  3. Qwen3-ASR hotwords: QwenLM/Qwen3-ASR issue #157 asks if hotword biasing exists (no official support seen; third-party phonetic-hotword server flyw/Qwen3-ASR-Phonetic-Hotword-Server, not vi-specific). Context-text prompting not confirmed for vi.
  4. Post-ASR LLM correction with glossary (not seen as a Vietnamese-specific recipe; speculation).
- No Vietnamese community post quantifying name/loanword accuracy found. Tone errors (đọng/động, nghĩ/nghỉ): no source discussing fixes; gipformer issue #2 shows community cares about diacritic-aware scoring (author clarified normalization strips punctuation, NOT diacritics, so reported WERs DO count tone errors).

## Long-form podcast with timestamps (what practitioners do)
- Pattern everywhere: Silero VAD (<=20-30 s chunks) -> batch transducer/CTC -> merge; timestamps = chunk boundaries or token times. Seen in gipformer ClawHub server, summo-app (Silero v5 chosen, MIT), welcomyou app, Kartoff3l/Vietnamese-VAD-ASR (compares whole-audio vs WebRTC-VAD vs Silero-VAD with faster-whisper; README only, no numbers fetched).
- Native long-form: ChunkFormer (prior notes). Native word timestamps + PnC: PhoASR-whisper-small (research license), Parakeet-vi (prior notes).
- Whisper hallucination on silence under long audio persists (VAD gating mitigates) — consistent with Kartoff3l repo purpose; no numbers.

## Summary table (accuracy per resource, vendor numbers; bold = best seen)
| Model | Params | License | Tech-domain WER | FLEURS | Speed | Timestamps | PnC | Tag |
|---|---|---|---|---|---|---|---|---|
| gipformer1.5-68M | 68M | MIT | **27.49** | 12.65 (vendor) / 8.3 (indep.) | RTF 0.017-0.02 CPU 8thr | via VAD/sherpa tokens | no | GEM |
| hynt Zipformer-150M | 153M | NC-ND | n/a | n/a | <0.3s/12s 3090 | sherpa tokens | no | GEM (non-commercial) |
| hynt Zipformer-30M | 30M | NC-ND | 29.77 | 13.03 | ~40x RT CPU | sherpa tokens | no | GEM |
| Qwen3-ASR-1.7B | 1.7B | Apache | 27.95 | **10.13** | heavy | none for vi | yes | MEH |
| chunkformer-large-vie | 110M | CC-BY(RNNT)/NC(CTC) | 39.81 | 14.17 | long-form native | segment | no | GEM (long-form) |
| PhoASR-whisper-small | 244M | BSD-Clear+RAIL (research) | 44.35 | 13.97 | fast (small) | word | yes | MEH |
| PhoWhisper-large | 1.5B | BSD-3 | 42.94 | 12.62 | slow | whisper | yes | MEH |
| Parakeet-ctc-0.6b-vi | 600M | NVIDIA OML gated | 48.69 | 16.11 | fast GPU | word | yes | MEH on domain speech |

## Unresolved
- whisper-large-v3-turbo on gipformer's 16-set table / summo FLEURS set: nobody has run it. Must self-run (user baseline 2.37% on TTS audio is not predictive: Parakeet 5.27% on TTS vs 48.69% on real tech-domain speech in third-party table).
- Does Gipformer give word/token timestamps out of the box (check infer_onnx.py / sherpa `result.timestamps`)? Not verified.
- Gipformer training data size/licence of data (issue #1 open).
- Hotword boost quality for Vietnamese BPE (diacritics, multi-syllable English names) untested.
- ZipFormer-150M third-party validation; ONNX int8 speed on CPU/Mac not measured.
- VietPrism: release status, license, transcripts quality.
- Whether any forum content exists (Viblo/VOZ search backends returned nothing relevant; may be index gap, not absence).
- ChunkFormer RNNT punctuation (still open from prior notes).

## Sources
https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt ; https://github.com/ggroup-ai-lab/gipformer ; https://github.com/ggroup-ai-lab/gipformer/issues/1 ; https://github.com/ggroup-ai-lab/gipformer/issues/2 ; https://huggingface.co/datasets/g-group-ai-lab/vi-asr-tech-test ; https://github.com/Techainer/summo-app/blob/master/docs/benchmarks.md ; https://github.com/Techainer/summo-app/releases/tag/v0.13.1 ; https://skills.lc/openclaw/skills/openclaw-skills-skills-ai-ggroup-gipformer-skill-md ; https://huggingface.co/hynt/ZipFormer-150M-CR-CTC-RNNT-6000h ; https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small ; https://arxiv.org/abs/2603.14779 ; https://aclanthology.org/2026.findings-eacl.345/ ; https://huggingface.co/Nampfiev1995/nemotron35-encoder-only-vi-asr ; https://huggingface.co/luvox-ai/vietasr-ssl-char42-canonical-vi-ctc ; https://github.com/quangkmhd/Qwen3-ASR-VI ; https://huggingface.co/alphaedge-ai/whisper-large-v3-turbo-vie-32768 ; https://github.com/VinAIResearch/PhoWhisper/issues/17 ; https://arxiv.org/abs/2609.30005 ; https://arxiv.org/abs/2509.05983 ; https://arxiv.org/html/2603.01894v1 ; https://github.com/welcomyou/sherpa-vietnamese-asr ; https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html ; https://github.com/QwenLM/Qwen3-ASR/issues/157 ; https://github.com/Kartoff3l/Vietnamese-VAD-ASR ; https://vlsp.org.vn/index.php/vlsp2026/eval/ ; https://www.facebook.com/hakien/posts/g-group-ai-lab-gi%E1%BB%9Bi-thi%E1%BB%87u-gipformer-m%C3%B4-h%C3%ACnh-nh%E1%BA%ADn-d%E1%BA%A1ng-gi%E1%BB%8Dng-n%C3%B3i-ti%E1%BA%BFng-vi%E1%BB%87t-c%E1%BB%B1c-n/10241470074719580/ ; https://mochiai.vn/vi/blog/asr-nhan-dang-giong-noi-tieng-viet-2026 ; https://vexascribe.com/whisper-large-v3-vs-turbo ; https://viblo.asia/tags/speech-recognition ; https://voz.vn/t/chiec-podcast-dang-nghe-cua-cac-thim.190126/
