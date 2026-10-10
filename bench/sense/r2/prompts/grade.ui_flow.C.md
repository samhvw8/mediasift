You grade answers in a video benchmark. Read bench/sense/clips/ui_flow.key.json (questions, reference answers and accepted variants)
and the answer files listed below. For each arm and question, decide correct (1) or wrong (0).
An answer is correct when it means the same as the reference or an accepted variant; ignore case,
articles and small spelling differences; a number must match; a list must have the same items in the same
order; extra detail is fine if it does not contradict the reference; a vague answer that would fit many
videos is wrong.

Answer files:
- bench/sense/r2/answers/ui_flow.C.json

Write JSON (nothing else) to bench/sense/r2/answers/ui_flow.grades.C.json:
{"video": "ui_flow", "grades": {"<arm>": {"q01": 1, ...}, ...}, "notes": {"<arm>/q01": "only if borderline"}}
Then reply with one line: GRADED
