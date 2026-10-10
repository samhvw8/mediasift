You grade answers in a video benchmark. Read the key files (bench/qa/big_buck_bunny/key.json, bench/qa/cs50_lecture/key.json, bench/qa/python_100s/key.json, bench/qa/tears_of_steel/key.json; each answer file holds one question of that video) (questions, reference answers and accepted variants)
and the answer files listed below. For each arm and question, decide correct (1) or wrong (0).
An answer is correct when it means the same as the reference or an accepted variant; ignore case,
articles and small spelling differences; a number must match; extra detail is fine if it does not
contradict the reference; a vague answer that would fit many videos is wrong.

Answer files:
- bench/qa/answers_single/big_buck_bunny.q09.C.json  (arm C, video big_buck_bunny, question q09)
- bench/qa/answers_single/big_buck_bunny.q09.F.json  (arm F, video big_buck_bunny, question q09)
- bench/qa/answers_single/big_buck_bunny.q14.C.json  (arm C, video big_buck_bunny, question q14)
- bench/qa/answers_single/big_buck_bunny.q14.F.json  (arm F, video big_buck_bunny, question q14)
- bench/qa/answers_single/cs50_lecture.q06.C.json  (arm C, video cs50_lecture, question q06)
- bench/qa/answers_single/cs50_lecture.q06.F.json  (arm F, video cs50_lecture, question q06)
- bench/qa/answers_single/cs50_lecture.q07.C.json  (arm C, video cs50_lecture, question q07)
- bench/qa/answers_single/cs50_lecture.q07.F.json  (arm F, video cs50_lecture, question q07)
- bench/qa/answers_single/python_100s.q02.C.json  (arm C, video python_100s, question q02)
- bench/qa/answers_single/python_100s.q02.F.json  (arm F, video python_100s, question q02)
- bench/qa/answers_single/python_100s.q06.C.json  (arm C, video python_100s, question q06)
- bench/qa/answers_single/python_100s.q06.F.json  (arm F, video python_100s, question q06)
- bench/qa/answers_single/tears_of_steel.q01.C.json  (arm C, video tears_of_steel, question q01)
- bench/qa/answers_single/tears_of_steel.q01.F.json  (arm F, video tears_of_steel, question q01)
- bench/qa/answers_single/tears_of_steel.q08.C.json  (arm C, video tears_of_steel, question q08)
- bench/qa/answers_single/tears_of_steel.q08.F.json  (arm F, video tears_of_steel, question q08)

Write JSON (nothing else) to bench/qa/answers_single/grades.json:
{"video": "single", "grades": {"<video>.<qid>.<arm>": 1, ...}, "notes": {"<arm>/q01": "only if borderline"}}
Then reply with one line: GRADED
