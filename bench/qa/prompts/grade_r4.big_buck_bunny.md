You grade answers in a video benchmark. Read bench/qa/big_buck_bunny/key.json (questions, reference answers and accepted variants)
and the answer files listed below. For each arm and question, decide correct (1) or wrong (0).
An answer is correct when it means the same as the reference or an accepted variant; ignore case,
articles and small spelling differences; a number must match; extra detail is fine if it does not
contradict the reference; a vague answer that would fit many videos is wrong.

Answer files:
- bench/qa/answers_r4/big_buck_bunny.F.json  (arm F)

Write JSON (nothing else) to bench/qa/answers_r4/big_buck_bunny.grades.json:
{"video": "big_buck_bunny", "grades": {"<arm>": {"q01": 1, ...}, ...}, "notes": {"<arm>/q01": "only if borderline"}}
Then reply with one line: GRADED
