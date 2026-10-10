# Long-video baseline (2026-10-08)

Target case from `research/261002-architecture.md`: long talk video, cost per
question. Two videos, analysed and indexed on an M1 Pro:

| video | length | language | ASR | analyse + index | storyboard |
|---|---|---|---|---|---|
| `cancer` podcast | 1:45:36 | English | parakeet-mlx, 24x realtime | 8.2 min | 207 KB |
| `altman_vi` (anydub dub) | 47:18 | Vietnamese (TTS) | mlx-whisper turbo, 23x realtime | 4.6 min | 77 KB |

Questions: 14 per video written from the oracle (`writer_long.md`; for
`altman_vi` the oracle transcript is the exact dub script). Baseline = 4 of
them per video (2 speech, 1 image, 1 both) as one-question sessions in three
arms; tokens are the sub-agent's total usage (`answers_single/usage.tsv`).

| arm | correct | total tokens per session (mean, min-max) |
|---|---|---|
| T transcript file only | 8/8 (2 were unsure guesses: "globe", "denim jacket") | 81.5k (63-92k) |
| S storyboard + sheets | 8/8 | 78.7k (66-93k) |
| F storyboard + find/look | 8/8 | 88.0k (85-91k) |

Fixed overhead of an answerer session is ~63-65k (instructions and tools), so
the marginal cost is roughly S 14k, T 17k, F 23k.

Findings:
- Agents do not read a 207 KB storyboard; they grep it. A grep-able text file
  already gives near-constant cost per question on a 1h45m video, so the
  plan's "~60k tokens per question today" does not hold for a coding agent.
- `find` is the most expensive arm: each call prints the speech at all 6 hits
  plus 3 speech matches, which costs more than grepping.
- The visual questions were guessable from context; they did not separate
  the arms. Specific-fact questions favour grep. Untested: global questions
  ("what are the main claims"), paraphrased questions with no shared keyword,
  and names that ASR misspelled, which is where grep should fail.

Vietnamese ASR (plan step 2), 15-min slice of `altman_vi` against the exact
script (`research/261008-vi-asr-eval/results.md`): whisper large-v3-turbo
2.37% WER (the same on the Mac via mlx-whisper), ChunkFormer 2.61%,
Zipformer 2.57%, PhoWhisper 3.98%, Parakeet-CTC-vi 5.27% (370x realtime).
TTS audio, so absolute numbers are optimistic; framesift's default stays.
