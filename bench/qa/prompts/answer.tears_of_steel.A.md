You are tested on how well you understand a video from limited material. Answer every question in
bench/qa/tears_of_steel/questions.json using ONLY the material described under YOUR The transcript only: bench/qa/tears_of_steel/A/transcript.txt. Do not open any other file in the
repository (in particular never open key.json, oracle/, or other arms' folders), and do not use the web.

YOUR The transcript only: bench/qa/tears_of_steel/A/transcript.txt:
The transcript only: bench/qa/tears_of_steel/A/transcript.txt

Rules:
- Answer each question with a short answer (a word, number, name or short phrase).
- If the material does not let you know, give your best guess and set "sure": false. Never leave one blank.
- Keep a count of every image you read (each Read of a .jpg counts as one image).

Write JSON (nothing else) to bench/qa/answers/tears_of_steel.A.json:
{"arm": "A", "images_read": <int>, "answers": [{"id": "q01", "answer": "...", "sure": true}, ...]}
Then reply with one line: ANSWERED <n> IMAGES <images_read>
