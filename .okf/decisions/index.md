# decisions

* [Codec as evidence](codec-as-evidence.md) - off-cadence I-frames and packet sizes as a free signal.
* [Keyframe drift](keyframe-drift.md) - I/P analogy, localized motion-tolerant change.
* [Screen vs camera](screen-vs-camera.md) - identical-frame test and per-content parameters.
* [Hard-cut floor](cut-threshold.md) - sweep against codec scene-cuts.
* [ASR shares PCM](asr-shares-pcm.md) - avoid the PyAV path in faster-whisper.
* [Embedding search](embedding-search.md) - EmbeddingGemma 2 `find`/`look`: frame + speech score, 70-token images, local server; bench evidence.
* [framesift see](see-sensor.md) - v0.3 sensor card instead of a storyboard: the problem, the parts, the first measurements, open questions.
* [Picture sensing](picture-sensing.md) - flow tiles, event crops, camera moves, shot letters, drawn-in overlays, watermarks.
* [Sound sensing](sound-sensing.md) - CED-small on 3 s windows, counting without syllables or drums, hedged names, speaker letters, corrections.
* [Speech and speed](speech-and-speed.md) - parakeet / whisper, the GPU box overlap and fallback, whisper cue splitting, the Mac lag measurements.
