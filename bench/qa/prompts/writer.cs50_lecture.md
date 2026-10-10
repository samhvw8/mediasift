You write a question set for a video-understanding benchmark. You are the only one who sees the full video;
the agents tested later see much less. Your questions decide whether a frame-selection tool is better
than alternatives, so they must be fair, specific and gradable.

Video: cs50_lecture. Read ONLY these files (do not open anything else in the repository, and never run framesift):
- bench/qa/cs50_lecture/oracle/transcript.txt   (timed transcript; may be empty for videos without speech)
- bench/qa/cs50_lecture/oracle/dense_*.jpg      (every image; frames every 2 s, each tile labelled with its time)

Write 14 questions:
- 8 "image": the answer is visible on screen and NOT said in the speech (on-screen text, code, a number,
  an object, a colour, who/what is shown, an action). Spread them over the whole video, including some
  moments that are visible for only a few seconds. At least 3 should be about something brief.
- 3 "speech": the answer is only in what is said.
- 3 "both": needs the picture and the speech together, or two moments (e.g. "what changes between ...").
For a video without speech, make all 14 "image".

Rules:
- One clear short answer per question (a word, number, name or short phrase). No yes/no questions.
- Do not mention times in the question, and do not give away the answer in the wording.
- A question must have one right answer for someone who watched the video.
- "evidence": every time span in seconds where the answer can be seen or heard, from the tile labels
  and transcript times. Be generous by about 1 s at each end.
- "accept": alternative phrasings a grader should accept.

Write JSON (nothing else) to bench/qa/cs50_lecture/key.json:
{"video": "cs50_lecture", "questions": [
  {"id": "q01", "type": "image|speech|both", "question": "...", "answer": "...", "accept": ["..."],
   "evidence": [[start, end], ...]}
]}
Then reply with one line: WROTE <n>
