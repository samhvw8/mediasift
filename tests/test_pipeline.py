import json

import pytest

from framesift.pipeline import Params, analyze
from make_fixture import make

EXPECTED = [4.0, 8.0, 12.0, 16.0]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("fs")
    video = str(d / "fixture.mp4")
    make(video)
    return analyze(video, str(d / "out"), Params())


def boundaries(m):
    return [s["start"] for s in m["segments"][1:]]


def test_boundaries_match_ground_truth(result):
    got = boundaries(result.manifest)
    assert len(got) == len(EXPECTED), got
    for g, e in zip(got, EXPECTED):
        assert abs(g - e) <= 0.5, (got, EXPECTED)


def test_reasons_name_the_modality(result):
    segs = result.manifest["segments"]
    assert "visual_cut" in segs[1]["reasons"]
    assert "audio_change" in segs[2]["reasons"] and "visual_cut" not in segs[2]["reasons"]


def test_audio_boundary_snaps_into_pause(result):
    seg = result.manifest["segments"][2]
    assert "at_pause" in seg["reasons"] and 8.0 <= seg["start"] <= 8.6


def test_static_segments_cost_one_frame(result):
    for seg in result.manifest["segments"]:
        if seg["id"] != 4:  # the moving test pattern may earn more
            assert len(seg["keyframes"]) == 1, seg


def test_return_to_same_shot_is_deduplicated(result):
    last = result.manifest["segments"][-1]["keyframes"][0]
    first = result.manifest["segments"][0]["keyframes"][0]
    assert last.get("same_as") == first["file"]


def test_outputs_written(result):
    out = result.out_dir
    m = json.loads((out / "manifest.json").read_text())
    assert (out / "storyboard.md").exists()
    for seg in m["segments"]:
        for kf in seg["keyframes"]:
            if "file" in kf:
                assert (out / kf["file"]).stat().st_size > 0
    assert m["sheets"] and (out / m["sheets"][0]).exists()


def test_transcript_attaches_by_midpoint(tmp_path):
    from framesift.transcript import assign, load
    srt = tmp_path / "t.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:05,500 --> 00:00:06,000\nworld\n")
    assert assign(load(str(srt)), [(0, 4), (4, 8)]) == ["hello", "world"]


def test_words_group_into_sentence_cues():
    from framesift.asr import words_to_cues
    w = [(0.0, 0.3, " Hello"), (0.3, 0.6, " there."), (0.7, 1.0, " Next"), (2.0, 2.3, " after"), (2.3, 2.6, " pause")]
    assert [c.text for c in words_to_cues(w)] == ["Hello there.", "Next", "after pause"]


def test_cue_spanning_a_short_segment_reaches_it():
    from framesift.transcript import Cue, assign
    assert assign([Cue(1.0, 9.0, "long cue")], [(0, 6), (6, 8), (8, 20)]) == ["long cue", "long cue", ""]


def _timed(text: str, step: float = 0.35, pauses: dict | None = None):
    out, t = [], 0.0
    for i, w in enumerate(text.split()):
        t += (pauses or {}).get(i, 0.0)
        out.append((t, t + step - 0.05, " " + w))
        t += step
    return out


def test_long_cue_splits_at_a_comma_not_mid_phrase():
    """faster-whisper on the GPU box: an 8 s cap cut 'like web...' | 'apps with the Django framework'."""
    from framesift.asr import words_to_cues
    w = _timed("It's commonly used to build server-side applications, like web apps with the Django "
               "framework, and is the language of choice for big data analysis and machine learning.")
    texts = [c.text for c in words_to_cues(w)]
    assert all(t.endswith((",", ".")) for t in texts), texts
    assert all(c.end - c.start <= 8.0 + 1e-6 for c in words_to_cues(w))


def test_long_unpunctuated_cue_splits_at_its_widest_pause():
    """Vietnamese from whisper often has no commas: split where the speaker breathed."""
    from framesift.asr import words_to_cues
    text = "thì cái lưu ý đấy là một khi mà các bạn đã làm to ấy thì kiểu gì các bạn cũng sẽ phải hợp tác đúng không"
    w = _timed(text, pauses={15: 0.4})           # a 0.4 s breath before 'thì kiểu gì'
    texts = [c.text for c in words_to_cues(w)]
    assert texts[0].endswith("ấy") and texts[1].startswith("thì kiểu"), texts


def test_ellipsis_mid_sentence_is_not_a_sentence_end():
    """whisper wrote 'like web...' then 'apps with the Django framework' with no pause between."""
    from framesift.asr import words_to_cues
    w = _timed("used to build apps, like web... apps with the Django framework.")
    assert [c.text for c in words_to_cues(w)] == ["used to build apps, like web... apps with the Django framework."]
    w = _timed("so I was thinking... anyway the next part.", pauses={4: 0.8})
    assert len(words_to_cues(w)) == 2, "a trailing-off followed by a pause still splits"


def test_no_fragments_when_pauses_are_uniform():
    """Real whisper Vietnamese: uniform word gaps, no commas, a capital where a sentence starts.
    The widest-gap rule cut 'Thì cái lời khuyên không' | 'nên hợp tác nó không đúng cho' | 'lắm Mình ...'."""
    from framesift.asr import words_to_cues
    text = ("Thì cái lời khuyên không nên hợp tác nó không đúng cho lắm Mình nghĩ rằng là cái mình nên "
            "hướng tới đây là làm sao để hợp tác")
    cues = words_to_cues(_timed(text, step=0.3))
    assert all(c.end - c.start >= 3.0 for c in cues[:-1]), [c.text for c in cues]
    assert cues[0].text.endswith("cho lắm") and cues[1].text.startswith("Mình nghĩ"), [c.text for c in cues]


def test_ctc_frames_become_timed_words():
    """QUAN-Transcribe emits one CTC unit per 40 ms frame: '▁' starts a word, repeats extend it,
    blanks (0) end nothing, and a unit after a blank that repeats the previous one is a new token."""
    import numpy as np
    from framesift.asr import ctc_words
    pieces = ["<blank>", "<unk>", "▁xin", "▁chào", "▁b", "ạn"]
    best = np.array([0, 2, 2, 0, 3, 0, 0, 0, 0, 4, 5, 5, 0, 1, 0])
    words = ctc_words(best, pieces, offset=10.0)
    assert [w[2] for w in words] == [" xin", " chào", " bạn"]
    assert words[0][:2] == (10.04, 10.12)            # frames 1-2
    assert words[1][:2] == (10.16, 10.2)             # frame 4
    assert words[2][:2] == (10.36, 10.48)            # frames 9-11, across '▁b' + 'ạn'
    assert all(isinstance(t, float) for w in words for t in w[:2])


def test_quiet_split_keeps_every_sample_and_its_offset():
    import numpy as np
    from framesift.asr import split_quiet
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, 16000 * 75).astype(np.float32)
    x[16000 * 27:16000 * 27 + 1600] = 0.0            # a quiet 0.1 s at 27 s
    parts = split_quiet(x, 30.0)
    assert [o for o, _ in parts][0] == 0
    assert all(len(p) <= 16000 * 30 for _, p in parts)
    assert sum(len(p) for _, p in parts) == len(x)
    assert all(o2 == o1 + len(p1) for (o1, p1), (o2, _) in zip(parts, parts[1:]))
    assert 16000 * 27 <= parts[1][0] <= 16000 * 27 + 1600, "the cut lands in the quiet stretch"


def test_vietnamese_on_apple_silicon_picks_quan(monkeypatch):
    from framesift import asr
    monkeypatch.setattr(asr, "apple_silicon", lambda: True)
    monkeypatch.setattr(asr, "quan_available", lambda: True)
    monkeypatch.setattr(asr, "_has", lambda m: True)
    assert asr.pick("auto", "vi") == "quan-coreml"
    assert asr.pick("auto", "en") == "parakeet-mlx"
    assert asr.pick("auto", "ja") == "mlx-whisper"
    assert asr.pick("mlx-whisper", "vi") == "mlx-whisper"
    monkeypatch.setattr(asr, "quan_available", lambda: False)
    assert asr.pick("auto", "vi") == "mlx-whisper"


def test_ced_log_mel_matches_torchaudio():
    """The Core ML tagger gets its features from numpy: they must be CED's own torchaudio features
    (MelSpectrogram center/reflect, AmplitudeToDB top_db=120), fixture made with torchaudio 2.7."""
    import numpy as np
    from pathlib import Path
    from framesift import sound
    f = np.load(Path(__file__).parent / "fixtures" / "ced_logmel.npz")
    got = sound.log_mel(f["wave"][None])[0]
    assert got.shape == f["logmel"].shape == (64, 301)
    assert np.abs(got - f["logmel"]).max() < 0.05


def test_core_ml_tags_keep_window_order_and_top_k(monkeypatch):
    """Batches of 64 windows: each row must land on its own window, the last batch padded, and only
    each window's top-k classes kept (what the sherpa tagger returns, and what the rules were tuned on)."""
    import numpy as np
    from framesift import coreml, sound

    def fake_predict(model, inputs, outputs):
        mel = inputs["mel"]
        p = np.zeros((mel.shape[0], 527), np.float32)
        p[:, :20] = np.linspace(0.05, 0.6, 20)               # 20 classes heard, only the top 12 survive
        p[:, 100] = 0.7 + mel[:, 0, 0] / 1000                 # the window's marker
        return [p]
    monkeypatch.setattr(coreml, "predict", fake_predict)
    monkeypatch.setattr(sound, "window_log_mels", lambda y, st: np.broadcast_to(y[st][:, None, None], (len(st), 64, 301)).copy())
    step = int(0.5 * sound.SR)
    y = np.zeros(sound.SR * 80, np.float32)
    n = 1 + (len(y) - int(sound.WIN_S * sound.SR)) // step
    for i in range(n):
        y[i * step] = i + 1                                   # window i starts with the value i + 1
    tags = sound._tag_coreml(object(), y, hop=0.5, top_k=12, names=["x"] * 527)
    assert tags.probs.shape == (n, 527) and n > 64 * 2
    assert np.allclose(tags.probs[:, 100], 0.7 + (np.arange(n) + 1) / 1000, atol=4e-4)
    assert ((tags.probs > 0).sum(axis=1) == 12).all()
    assert tags.probs[:, :9].max() == 0, "classes below the top 12 must be dropped"


def test_shared_stft_gives_the_per_window_features():
    """Overlapping 3 s windows share frames: one STFT over the file plus each window's 4 reflect-padded
    edge frames must give exactly what log_mel computes window by window."""
    import numpy as np
    from framesift import sound
    y = np.random.default_rng(3).normal(0, 0.1, sound.SR * 20).astype(np.float32)
    y[sound.SR * 5:sound.SR * 6] = 0.0                    # silence: the per-window top_db floor
    for hop in (0.5, 1.0):
        step = int(hop * sound.SR)
        starts = np.arange(1 + (len(y) - 48000) // step) * step
        ref = sound.log_mel(np.stack([y[s:s + 48000] for s in starts]))
        got = sound.window_log_mels(y, starts)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-3, np.abs(got - ref).max()
