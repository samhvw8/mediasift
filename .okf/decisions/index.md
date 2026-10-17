# decisions

* [Codec as evidence](codec-as-evidence.md) - off-cadence I-frames and packet sizes as a free signal.
* [Keyframe drift](keyframe-drift.md) - I/P analogy, localized motion-tolerant change.
* [Screen vs camera](screen-vs-camera.md) - identical-frame test and per-content parameters.
* [Hard-cut floor](cut-threshold.md) - sweep against codec scene-cuts.
* [ASR shares PCM](asr-shares-pcm.md) - avoid the PyAV path in faster-whisper.
* [Embedding search](embedding-search.md) - EmbeddingGemma 2 `find`/`look`: frame + speech score, 70-token images, local server; bench evidence.
* [mediasift see](see-sensor.md) - v0.3 sensor card instead of a storyboard: the problem, the parts, the first measurements, open questions.
* [Picture sensing](picture-sensing.md) - flow tiles, event crops, camera moves, shot letters, drawn-in overlays, watermarks.
* [Sound sensing](sound-sensing.md) - CED-small on 3 s windows, counting without syllables or drums, hedged names, speaker letters, corrections.
* [Music and voice mood](music-and-voice-mood.md) - music beds carry CED's mood and genre words and a mood arc (88% right when stated, no new model); speech emotion measured at 45% held out and left out.
* [Speech and speed](speech-and-speed.md) - parakeet / whisper, the GPU box overlap and fallback, whisper cue splitting, the Mac lag measurements.
* [Distribution](distribution.md) - install from GitHub with uv or mise `pypi:`; the repo is a Claude Code plugin marketplace with the skill and a guarded prompt hook.
* [Roadmap](roadmap.md) - the ordered next steps from the 2026-10-17 council, their evidence, the stress-test changes each must carry, and status. Start here.
