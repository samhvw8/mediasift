# Music and voice mood: what the card can say (2026-10-17)

Scripts and pick lists behind `.okf/decisions/music-and-voice-mood.md`. Media is not kept; rebuild it:

- AudioSet eval clips: `as_hits.json` lists the 742 rows (single mood or single genre label);
  `as_extract.py OUT as_hits.json` streams the agkphysics/AudioSet eval parquet shards from Hugging
  Face (24 GB read, ~17 min) and keeps 16 kHz wavs. YouTube refused the same fetch (bot check).
- RAVDESS speech: zenodo record 1188976, `Audio_Speech_Actors_01-24.zip`.
- EmoNet-Voice bench: HF `t1a5anu-anon/emonet-voice-bench` (CC BY 4.0, May 2025);
  `emonet_extract.py` keeps clips every rater scored 2 for one mapped emotion.
- `ced_*.py` run in the framesift env (`uv run python ...`). `effnet_clap.py`, `e2v_*.py`, `sv.py`
  need essentia, onnxruntime, transformers 5, sherpa-onnx (a separate Python 3.14 env).
- Models: Essentia `discogs-effnet-bsdynamic-1.onnx`, `mtg_jamendo_moodtheme-discogs-effnet-1.onnx`;
  `laion/clap-htsat-unfused` (`larger_clap_music` loads with all-zero scores under transformers 5);
  emotion2vec+ base ONNX from `pankotaro/emotion2vec-plus-base-onnx` (int8 by onnxruntime dynamic
  quantisation); SenseVoice int8 2025-09-09 (sherpa-onnx).
