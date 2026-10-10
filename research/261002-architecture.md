# framesift architecture brainstorm: synthesis (2026-10-02)

Mode: heavy-think Brainstorm, architecture preset, decision rubric. Five Opus lenses on one shared brief:
- Agent consumer (custom)
- Home operator (custom)
- First Principles
- Minimalist
- Platform futurist (custom temporal lens)

## 1. Where all five agree (robust)

1. **The agent needs an index, not documents.** Today the podcast storyboard is about 188 KB, roughly 50–65k tokens:
   - 66% transcript
   - 15% headers for 487 segments
   - 10% stat lines such as "audio 0.0" that the agent cannot act on

   Reading cost grows with video length; answer cost should grow with the question.
2. **Segments are internal.** They drive keyframe choice and clip cuts, and the agent never reads them. Agents address **time ranges**. The "487 segments" problem goes away.
3. **No GPU on the query path.** GPU work happens only at ingest, as batch jobs. Queries run on the Mac with the box switched off.
4. **No long-running model servers.** vLLM's 3–5 minute startup, the Whisper+VLM OOM, and the FastAPI overlay all come from treating models as services.
5. **Speech recognition is the core and the biggest accuracy risk.** Vietnamese ASR on the owner's dubbed audio is unmeasured. Measure it before building on top of it.
6. **The headline metric changes** from frame coverage at equal image count to **correct answers per agent token** on a fixed question set. Coverage stays as an internal regression check.

## 2. Where they collide (the real decisions)

| Question | Positions | Resolution |
|---|---|---|
| Does the local VLM stay? | Minimalist and First Principles: delete it; Claude is the stronger VLM. Operator: keep it as an eager batch job. Consumer: describe only frames where the picture says something the speech does not. Futurist: eager for screen recordings, lazy for talking heads. | **Visual-delta routing, decided by the QA eval.** Describe or OCR a frame only when it carries information the speech lacks: OCR text present, a slide or code change, low speech ratio, or a caption far from the nearby transcript. These signals already exist in `analyze`, so this adds no new knob. Talking-head podcasts get almost no VLM text; screen recordings get most of it. If the eval shows precomputed text doesn't raise answers per token, delete the VLM. |
| Precompute summaries, or do it lazily? | Consumer: precompute the card (overview plus chapters) and a claims ledger, because global questions fail under top-k retrieval. Minimalist: extractive only, deterministic. First Principles: write the outline on first query. | **The card is always precomputed.** By default it is extractive: TextTiling chapters plus the 2–3 most central sentences, deterministic and CPU-only. An LLM-written title, summary and claims ledger are an *optional batch layer* on top. A 5-hour video without a card can't be navigated. |
| Where does truth live? | Minimalist: per-video files plus grep. Consumer: one SQLite file per video. Futurist and First Principles: one library SQLite rebuilt from folders. | **Folders of immutable layers are the source of truth. One library SQLite (FTS5 + sqlite-vec) is a disposable, derived index** rebuilt by `framesift reindex`, with no backup and no sync. The storyboard and meaning files become generated views, not parallel artifacts. |
| Mac or box? | Operator and Futurist: the library lives on the Mac and the box is stateless compute. Minimalist: the Mac can do everything. | **The library sits next to the agent (the Mac).** The box speeds up ingest when it is on; it is not a requirement. |

## 3. What only appears when lenses are combined

- **One mechanism solves three problems.** Combining:
  - First Principles' lazy memo cache,
  - the Futurist's layers keyed by `(stage, model, config-hash)`,
  - the Operator's per-stage done-markers

  gives one content-addressed layer store. That store gives crash-resume for batch jobs, cheap model upgrades (re-run one layer and its descendants), and lazy on-demand enrichment, plus a log of what agents actually asked for, which shows what is worth precomputing.
- **The consumer becomes a producer** (Minimalist notes + Futurist layers). When Claude looks at pixels to answer a question, it writes its observation back as a layer (`claude-look@<handle>`). Pixels are paid for once, which neutralises the main risk of deleting the VLM: repeated token cost.
- **Durable audio + keyframes enable two things** (Futurist + Operator). Keeping `audio.opus` (about 10 MB/h) and 768 px keyframes means future models can reprocess videos whose source is deleted, and the Mac can answer `look` without the source file.
- **An English gloss for Vietnamese** (Consumer). English chapter summaries cut read tokens (Vietnamese probably costs 1.5–2.5× more; unmeasured) and make cross-lingual search work.

## 4. Candidates scored (decision rubric: value 35, feasibility 25, running cost 20, novelty 10, specificity 10)

| Candidate | Value | Feas. | Run cost | Novelty | Spec. | Score |
|---|---|---|---|---|---|---|
| C1 Status quo v0.2: eager everything, documents | 5 | 9 | 3 | 3 | 8 | 57 |
| C4 Operator spool: batch jobs, eager VLM on all frames | 6 | 8 | 8 | 4 | 9 | 70 |
| C2 Minimalist: ASR + CPU embeddings + files, no VLM | 6 | 9 | 9 | 5 | 8 | 74.5 |
| **C3 Synthesis: time-indexed library + batch enrichment** | **9** | **6** | **8** | **7** | **8** | **77.5** |

C3 wins narrowly. Its feasibility cost is real, but C2 is C3's first milestone, so building C2 first loses nothing.

## 5. Recommended architecture (C3)

```
                    INGEST (Mac or box, CPU)                    BATCH ENRICH (box GPU, oneshot "drain"; Mac MLX fallback)
video ──► analyze: signals, candidate times, keyframes 768px,   ──► GPU-A  ASR → word-timed transcript (process exits, VRAM freed)
          audio.opus  ─► lib/<audio-hash>/source.json                GPU-B  VLM only on visual-delta frames (+ self-test), optional
                                                                         LLM card text / claims ledger / English gloss (optional)
                          every stage writes layers/<stage>@<model>-<cfghash>/ with an atomic done-marker
                                              │
                                              ▼
              LIBRARY INDEX (Mac, disposable): one SQLite — FTS5 + sqlite-vec over transcript windows, chapters, OCR, SigLIP frame vectors
                                              │
                                              ▼
              AGENT VERBS (CLI + skill now, thin read-only MCP later), all speaking handles  fs1:<vid12>@<t0>-<t1>
                card      overview + 12–20 chapters + entities         (~300–600 tokens)
                find "q"  top-k hits with handles, scores, a weak-match warning  (~800)
                read h    raw transcript span (+ --frames)              (~400)
                look h    contact sheet or full-res frame rendered on demand (~1.5k per frame)
                status    pending / failed layers and age of the oldest pending item
              LAZY, on demand, cached as layers: TimeLens2 grounding, clip reads, claude-look observations
```

Estimated cost of a typical question: about 2–3k tokens on the 1h45m podcast (today about 60k) and about 1.5k on a 6-minute lecture.

## 6. Do not build (yet)

- A long-running vLLM or any warm model daemon. Use a oneshot drain instead; prefer llama.cpp (startup in seconds) if a quality A/B ties.
- Segments, stat lines or reason codes in agent-facing output.
- The storyboard as the entry point for long videos (it remains a drill-down view).
- An MCP server now (verbs over files first), a vector DB server, Postgres, or a job-queue service.
- Two-way Mac↔box sync of the library. The box holds no state.
- A cross-video knowledge graph, diarization as a default layer, a schema-migration framework, or keeping more than one previous layer per stage.
- Per-keyframe captions for talking-head video.

## 7. Order of work (each step unlocks the next)

1. **QA eval harness.** About 15 questions per video type with known answers, scoring correct answers per token. It decides every open question below. `bench/prep_qa.py` is the start.
2. **Vietnamese ASR measurement.** A 20–30 minute hand-checked sample from the anydub podcasts, comparing turbo, ChunkFormer, Parakeet-vi and PhoWhisper.
3. **Lock the durable decisions.** The `fs1:` handle format, a time base of normalized PTS seconds, and keeping `audio.opus` plus keyframes. These are the only choices that cannot be made later.
4. **Layer store plus the `card / find / read / look / status` verbs.** Use the extractive card and gte-multilingual or e5 embeddings. This is C2, already useful on its own.
5. **GPU drain as a oneshot** with markers keyed by input, model, params and prompt hash, plus a GPU lock in case the home box's llama.cpp/Qwen stack competes for VRAM (unverified; check `infra/`).
6. **Run the eval:** transcript + on-demand `look` versus adding visual-delta VLM text. Keep the VLM layer only if it raises answers per token.

## 8. Risks to watch

- **Retrieval fails invisibly.** The agent doesn't know what it didn't retrieve. `find` must report scores and weak matches, the card lists entities, and the skill requires `read` before citing.
- **Repeated pixel cost** without write-back. The `claude-look` layer is mandatory, not optional.
- **ASR is a single point of accuracy.** Everything downstream inherits its errors.
- **Silent backlog** when the box is off. `status` and the skill must show pending or failed layers every time.
