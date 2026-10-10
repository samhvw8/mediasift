You grade answers in a video benchmark. Read the key files (bench/qa_long/cancer/key.json and bench/qa_long/altman_vi/key.json; each answer file holds one question of that video) (questions, reference answers and accepted variants)
and the answer files listed below. For each arm and question, decide correct (1) or wrong (0).
An answer is correct when it means the same as the reference or an accepted variant; ignore case,
articles and small spelling differences; a number must match; extra detail is fine if it does not
contradict the reference; a vague answer that would fit many videos is wrong.

Answer files:
- bench/qa_long/answers_single/altman_vi.q03.T.json  (arm T, video altman_vi, question q03)
- bench/qa_long/answers_single/altman_vi.q03.S.json  (arm S, video altman_vi, question q03)
- bench/qa_long/answers_single/altman_vi.q03.F.json  (arm F, video altman_vi, question q03)
- bench/qa_long/answers_single/altman_vi.q07.T.json  (arm T, video altman_vi, question q07)
- bench/qa_long/answers_single/altman_vi.q07.S.json  (arm S, video altman_vi, question q07)
- bench/qa_long/answers_single/altman_vi.q07.F.json  (arm F, video altman_vi, question q07)
- bench/qa_long/answers_single/altman_vi.q10.T.json  (arm T, video altman_vi, question q10)
- bench/qa_long/answers_single/altman_vi.q10.S.json  (arm S, video altman_vi, question q10)
- bench/qa_long/answers_single/altman_vi.q10.F.json  (arm F, video altman_vi, question q10)
- bench/qa_long/answers_single/altman_vi.q12.T.json  (arm T, video altman_vi, question q12)
- bench/qa_long/answers_single/altman_vi.q12.S.json  (arm S, video altman_vi, question q12)
- bench/qa_long/answers_single/altman_vi.q12.F.json  (arm F, video altman_vi, question q12)
- bench/qa_long/answers_single/cancer.q03.T.json  (arm T, video cancer, question q03)
- bench/qa_long/answers_single/cancer.q03.S.json  (arm S, video cancer, question q03)
- bench/qa_long/answers_single/cancer.q03.F.json  (arm F, video cancer, question q03)
- bench/qa_long/answers_single/cancer.q08.T.json  (arm T, video cancer, question q08)
- bench/qa_long/answers_single/cancer.q08.S.json  (arm S, video cancer, question q08)
- bench/qa_long/answers_single/cancer.q08.F.json  (arm F, video cancer, question q08)
- bench/qa_long/answers_single/cancer.q11.T.json  (arm T, video cancer, question q11)
- bench/qa_long/answers_single/cancer.q11.S.json  (arm S, video cancer, question q11)
- bench/qa_long/answers_single/cancer.q11.F.json  (arm F, video cancer, question q11)
- bench/qa_long/answers_single/cancer.q14.T.json  (arm T, video cancer, question q14)
- bench/qa_long/answers_single/cancer.q14.S.json  (arm S, video cancer, question q14)
- bench/qa_long/answers_single/cancer.q14.F.json  (arm F, video cancer, question q14)

Write JSON (nothing else) to bench/qa_long/answers_single/grades.json:
{"video": "long-single", "grades": {"<video>.<qid>.<arm>": 1, ...}, "notes": {"<arm>/q01": "only if borderline"}}
Then reply with one line: GRADED
