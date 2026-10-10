You are tested on how well you understand a video from limited material. Answer every question in
bench/qa/cs50_lecture/questions.json using ONLY the material described under YOUR Everything in bench/qa/cs50_lecture/C/ : storyboard.md (segments, transcript per segment), the sheet_*.jpg contact sheets it lists (12 labelled keyframes each), and frames/*.jpg if you need full resolution.. Do not open any other file in the
repository (in particular never open key.json, oracle/, or other arms' folders), and do not use the web.

YOUR Everything in bench/qa/cs50_lecture/C/ : storyboard.md (segments, transcript per segment), the sheet_*.jpg contact sheets it lists (12 labelled keyframes each), and frames/*.jpg if you need full resolution.:
Everything in bench/qa/cs50_lecture/C/ : storyboard.md (segments, transcript per segment), the sheet_*.jpg contact sheets it lists (12 labelled keyframes each), and frames/*.jpg if you need full resolution.

Rules:
- Answer each question with a short answer (a word, number, name or short phrase).
- If the material does not let you know, give your best guess and set "sure": false. Never leave one blank.
- Keep a count of every image you read (each Read of a .jpg counts as one image).

Write JSON (nothing else) to bench/qa/answers_r2/cs50_lecture.C.json:
{"arm": "C", "images_read": <int>, "answers": [{"id": "q01", "answer": "...", "sure": true}, ...]}
Then reply with one line: ANSWERED <n> IMAGES <images_read>
