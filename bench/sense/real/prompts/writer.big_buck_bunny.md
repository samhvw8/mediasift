You write a test of how well someone understands a video they could not watch. You are the only
one who sees the full material: bench/sense/real/big_buck_bunny/oracle holds contact sheets with a frame every 0.5 s (each tile labelled
with its time in seconds), and bench/sense/real/big_buck_bunny/transcript.txt is the timed speech. Read ALL of the sheets in order and the transcript.

Write 8 questions about big_buck_bunny that need understanding of TIME in the video, not just a single picture:
- order of events or steps ("what happens right after X", "which came first"),
- motion and direction ("which way does X move", "where does X go"),
- counting repeated actions or appearances ("how many times does X"),
- duration or brief events ("what appears briefly when ...", "for about how long"),
- picture and speech together ("what is on screen when the speaker says '...'").
Each answer must be unambiguous and checkable from the sheets and transcript; avoid anything you are unsure
of in a 0.5 s sampling (do not ask about events shorter than about 1 s unless clearly visible in two tiles).
Prefer questions whose answer is NOT given away by the transcript alone. Use plain descriptions of what is
visible, never tile numbers or sheet names. Mix the five kinds.

Write JSON (nothing else) to bench/sense/real/big_buck_bunny/key.json:
{"video": "big_buck_bunny", "questions": [{"id": "q01", "type": "sequence|motion|count|duration|deixis",
 "question": "...", "answer": "...", "accept": ["...", "..."], "evidence": [[start_s, end_s]]}, ...]}
Then reply with one line: WROTE 8
