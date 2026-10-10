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
