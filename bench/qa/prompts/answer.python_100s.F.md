You are tested on how well you understand a video from limited material. Answer every question in
bench/qa/python_100s/questions.json using ONLY the material described under YOUR The text file bench/qa/python_100s/C/storyboard.md (segments, times, transcript) and a search tool over the video. You may NOT open the sheet_*.jpg or frames/*.jpg files that the storyboard mentions. You see pictures only through these commands, run with Bash:
  .venv/bin/framesift find bench/results/runs_v03/python_100s "<what you want to see>"        # top 6 distinct moments as one grid; prints the grid path and the speech at each moment
  .venv/bin/framesift find bench/results/runs_v03/python_100s "<text>" --between 60-120 -k 4  # limit to a time range / fewer tiles
  .venv/bin/framesift look bench/results/runs_v03/python_100s --at 12.5,40,41.5                 # exact moments at full width, as one grid
Each command prints the path of a grid image; Read that image to see it. Describe what would be visible in the query (e.g. 'chart ranking programming languages', 'man with name caption'), not the question itself. Use as few grids as you need; read the storyboard text first, since many answers are in the speech.. Do not open any other file in the
repository (in particular never open key.json, oracle/, or other arms' folders), and do not use the web.

YOUR The text file bench/qa/python_100s/C/storyboard.md (segments, times, transcript) and a search tool over the video. You may NOT open the sheet_*.jpg or frames/*.jpg files that the storyboard mentions. You see pictures only through these commands, run with Bash:
  .venv/bin/framesift find bench/results/runs_v03/python_100s "<what you want to see>"        # top 6 distinct moments as one grid; prints the grid path and the speech at each moment
  .venv/bin/framesift find bench/results/runs_v03/python_100s "<text>" --between 60-120 -k 4  # limit to a time range / fewer tiles
  .venv/bin/framesift look bench/results/runs_v03/python_100s --at 12.5,40,41.5                 # exact moments at full width, as one grid
Each command prints the path of a grid image; Read that image to see it. Describe what would be visible in the query (e.g. 'chart ranking programming languages', 'man with name caption'), not the question itself. Use as few grids as you need; read the storyboard text first, since many answers are in the speech.:
The text file bench/qa/python_100s/C/storyboard.md (segments, times, transcript) and a search tool over the video. You may NOT open the sheet_*.jpg or frames/*.jpg files that the storyboard mentions. You see pictures only through these commands, run with Bash:
  .venv/bin/framesift find bench/results/runs_v03/python_100s "<what you want to see>"        # top 6 distinct moments as one grid; prints the grid path and the speech at each moment
  .venv/bin/framesift find bench/results/runs_v03/python_100s "<text>" --between 60-120 -k 4  # limit to a time range / fewer tiles
  .venv/bin/framesift look bench/results/runs_v03/python_100s --at 12.5,40,41.5                 # exact moments at full width, as one grid
Each command prints the path of a grid image; Read that image to see it. Describe what would be visible in the query (e.g. 'chart ranking programming languages', 'man with name caption'), not the question itself. Use as few grids as you need; read the storyboard text first, since many answers are in the speech.

Rules:
- Answer each question with a short answer (a word, number, name or short phrase).
- If the material does not let you know, give your best guess and set "sure": false. Never leave one blank.
- Keep a count of every image you read (each Read of a .jpg counts as one image).

Write JSON (nothing else) to bench/qa/answers/python_100s.F.json:
{"arm": "F", "images_read": <int>, "answers": [{"id": "q01", "answer": "...", "sure": true}, ...]}
Then reply with one line: ANSWERED <n> IMAGES <images_read>
