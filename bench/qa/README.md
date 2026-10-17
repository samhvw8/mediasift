# Agent QA bench

Does an agent understand a video better per image token with `find`/`look`
than with contact sheets? mediasift has no LLM, so the harness's own
sub-agents write, answer and grade the questions, in separate contexts.

| role | sees | writes |
|---|---|---|
| writer (one per video) | `oracle/` only: dense frames every 2 s + timed transcript | `<video>/key.json` (14 questions: answer, accepted variants, evidence spans) |
| answerer (one per video x arm) | its arm's material only, `questions.json` (no answers) | `answers_<round>/<video>.<arm>.json` |
| grader (one per video) | `key.json` + the answer files | `answers_<round>/<video>.grades.json` |

Arms: **A** transcript only · **B** transcript + uniform sheets (same image
count as C) · **C** storyboard + mediasift sheets · **F** storyboard +
`find`/`look` (no sheets). The writer never sees mediasift output, so the
questions do not favour what mediasift chose.

```bash
uv run python bench/prep_qa.py bench/qa bench/results/runs_v03 samples/*.mp4   # oracle + arm folders
python bench/qa_prompts.py writer            # then run each brief as a sub-agent
python bench/qa_prompts.py answer r5 CF      # briefs for a new round
python bench/qa_prompts.py grade r5 CF
uv run python bench/qa_report.py bench/qa bench/results/runs_v03 answers_r5
uv run python bench/search_eval.py bench/results/runs_v03 bench/qa            # LLM-free retrieval check
```

Before an F round, move `find/`, `look/` and `queries.jsonl` out of the run
dirs: the report counts every grid there as read.

## Caveats

- 56 questions on 4 videos: a 1-2 question difference is noise (C scored 53
  in two rounds; F 54, 54, 53).
- Image tokens use width x height / 750 after Claude's 1568 px resize. Agents
  choose how much to read, so tokens vary between rounds more than accuracy.
- Rounds 1-4 briefs repeated the arm's material three times (a template
  substitution bug, the same for every arm); `qa_prompts.py` fixes it.
- `tears_of_steel` q03: the writer misread the headline from a 480 px oracle
  tile; corrected to "creates" after a 768 px `look` (noted in the key).
