You write a question set for a video-understanding benchmark on a LONG video. You are the only one who
sees the full material; the agents tested later see much less. Your questions decide whether a tool helps
an agent answer questions about long videos cheaply, so they must be fair, specific and gradable.

Video: {{VIDEO}}. Read ONLY these files (do not open anything else in the repository, and never run framesift):
- {{ORACLE}}/transcript.txt   (the exact timed transcript)
- {{ORACLE}}/dense_*.jpg      (frames every {{STEP}} s, each tile labelled with its time)

Write 14 questions, in English:
- 8 "speech": the answer is in what is said. Spread them over the WHOLE video: at least 3 from the last
  third and at least 2 from the middle third. Prefer specific facts: a number, a name, a place, a claim, a
  reason given, a recommendation.
- 3 "image": the answer is visible on screen (on-screen text, a caption, a lower third, an object, what a
  person wears or holds, a setting). If the video has little to see, pick what is there.
- 3 "both": needs the picture and the speech together, or connects two distant moments
  (e.g. "what does the guest say right after the host shows X").

Rules:
- One clear short answer per question (a word, number, name or short phrase). No yes/no questions.
- Do not mention times in the question, and do not give away the answer in the wording.
- If the transcript is not English, still write the question in English; the answer may quote the original
  language, and give an English gloss in "accept".
- "evidence": every time span in seconds where the answer can be seen or heard. Be generous by ~2 s.

Write JSON (nothing else) to {{OUT}}:
{"video": "{{VIDEO}}", "questions": [
  {"id": "q01", "type": "speech|image|both", "question": "...", "answer": "...", "accept": ["..."],
   "evidence": [[start, end], ...]}
]}
Then reply with one line: WROTE <n>
