You are tested on how well you understand a video from limited material. You cannot watch or hear
the video; you only get the material below. Answer every question in bench/sense/real/r13/tears_of_steel/questions.json using ONLY that material.
Do not open any other file in the repository (never open *.key.json, other rounds or other arms' folders,
the source code, or the .mp4 directly), and do not use the web.

YOUR MATERIAL:
the folder bench/sense/real/r13/tears_of_steel/C: card.md (read it first) and the images it names (plus transcript.txt when present). You may also run `framesift look samples/tears_of_steel.mp4 <start>-<end>` to see any time span of the video; it prints the path of an image to read.

Rules:
- Answer each question with a short answer (a word, number, name or short phrase).
- If the material does not let you know, give your best guess and set "sure": false. Never leave one blank.
- Keep a count of every image you read (each Read of a .jpg or .png counts as one image).

Write JSON (nothing else) to bench/sense/real/r13/answers/tears_of_steel.C.json:
{"arm": "C", "images_read": <int>, "answers": [{"id": "q01", "answer": "...", "sure": true}, ...]}
Then reply with one line: ANSWERED <n> IMAGES <images_read>
