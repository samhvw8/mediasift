"""index / find / look on the synthetic fixture, with a fake embedder: images
embed as their mean colour, text as the colour it names. The real model is
measured in bench/, not here."""

import json

import numpy as np
import pytest
from PIL import Image

from mediasift import index, query
from mediasift.pipeline import Params, analyze
from make_fixture import make

COLOURS = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1]}


class FakeEmbedder:
    def __init__(self, ident=None):
        self.ident = ident or {"model": "fake", "revision": "1", "image_tokens": 70, "dim": 4}

    def identity(self):
        return dict(self.ident)

    @staticmethod
    def _unit(v):
        v = np.asarray(v, np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def images(self, paths):
        return np.stack([self._unit([*np.asarray(Image.open(p).convert("RGB"), np.float32).mean((0, 1)) / 255, 0.05])
                         for p in paths])

    def texts(self, texts, kind):
        out = []
        for t in texts:
            c = next((v for k, v in COLOURS.items() if k in t.lower()), [0, 0, 0])
            out.append(self._unit([*c, 0.05]))
        return np.stack(out)


SRT = """1
00:00:00,500 --> 00:00:03,000
this part is red

2
00:00:06,500 --> 00:00:09,500
now everything is blue
"""


@pytest.fixture(scope="module")
def ix(tmp_path_factory):
    d = tmp_path_factory.mktemp("search")
    video = str(d / "fixture.mp4")
    make(video)
    (d / "subs.srt").write_text(SRT)
    out = d / "out"
    analyze(video, str(out), Params(transcript=str(d / "subs.srt")))
    index.build(str(out), FakeEmbedder())
    return query.load(str(out))


def test_index_covers_video_with_probes_and_keyframes(ix):
    assert len(ix.frame_t) >= 20  # 20 s at one probe per second, plus keyframes
    assert (ix.out / ix.frame_file[0]).exists()
    assert len(ix.text) == 2 and "red" in ix.text[0]
    assert ix.meta["model"] == "fake" and ix.meta["image_tokens"] == 70


def test_find_returns_distinct_moments(ix):
    res = query.find(ix, FakeEmbedder(), "red", k=3)
    first = res["hits"][0]
    assert first["t"] < 4 or first["t"] >= 16
    ts = [h["t"] for h in res["hits"]]
    assert all(abs(a - b) >= query.MIN_SEP_S for i, a in enumerate(ts) for b in ts[i + 1:])
    # the red shot appears twice in the video but is one picture, so it is returned once
    reds = [h for h in res["hits"] if h["t"] < 4 or h["t"] >= 16]
    assert len(reds) == 1
    assert (ix.out / res["grid"]).exists()


def test_find_between_limits_range_and_speech(ix):
    res = query.find(ix, FakeEmbedder(), "blue", k=4, between=(4, 12))
    assert res["hits"] and all(4 <= h["t"] <= 12 for h in res["hits"])
    assert res["speech"][0]["text"] == "now everything is blue"


def test_find_like_excludes_the_moment_itself(ix):
    res = query.find(ix, FakeEmbedder(), like=1.0, k=2)
    assert all(abs(h["t"] - 1.0) > 3.0 for h in res["hits"])
    assert res["hits"][0]["t"] >= 16  # the same red shot, later


def test_vectors_from_another_model_are_refused(ix):
    with pytest.raises(SystemExit, match="model"):
        query.find(ix, FakeEmbedder({"model": "other", "image_tokens": 70, "dim": 4}), "red")


def test_queries_from_another_engine_are_allowed(ix):
    # text vectors agree across engines, so only model and dim must match
    res = query.find(ix, FakeEmbedder({"model": "fake", "engine": "other", "image_tokens": 280, "dim": 4}), "red")
    assert res["hits"]


def test_look_and_query_log(ix):
    res = query.look(ix, [2.0, 13.5])
    assert len(res["hits"]) == 2 and (ix.out / res["grid"]).exists()
    log = [json.loads(line) for line in (ix.out / "queries.jsonl").read_text().splitlines()]
    assert {r["op"] for r in log} >= {"find", "look"}


def test_speech_windows_merge_and_split():
    cues = [[0, 2, "a"], [2.2, 5, "b"], [5.1, 9, "c"], [20, 21, "d"]]
    w = index.speech_windows(cues, window=8.0)
    assert [x[2] for x in w] == ["a b c", "d"]
