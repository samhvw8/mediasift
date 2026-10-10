"""The senses that keyframes + transcript miss, checked against scripted ground truth.

Clips come from bench/sense/make_synth.py (no speech needed for these two), so the
answer is the script: 8 lamp flashes, 5 beeps, two of them together; a toast shown
for 0.7 s; 38 keystrokes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench" / "sense"))
import make_synth  # noqa: E402

from framesift import audio, decode, events, sound  # noqa: E402


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    rng = np.random.default_rng(7)
    made = {}
    for make in (make_synth.clip_count, make_synth.clip_ui):
        name, frame, dur, size, track, _ = make(out, rng)
        make_synth.write(out / f"{name}.mp4", frame, int(dur * make_synth.FPS), size, track, out)
        made[name] = str(out / f"{name}.mp4")
    return made


def grid(path):
    return events.build_grid(decode.video_frames(path, 15, events.DECODE_W, events.DECODE_H))


def test_lamp_flashes_counted_and_timed(clips):
    evs = events.detect(grid(clips["count"]))
    groups = [g for g in events.group(evs) if g[0].kind == "blink"]
    assert len(groups) == 1 and len(groups[0]) == 8
    assert np.allclose([e.start for e in groups[0]], make_synth_flashes(), atol=0.1)


def make_synth_flashes():
    return [1.0, 2.1, 2.9, 4.6, 5.2, 7.0, 8.8, 10.3]


def test_beeps_counted_named_and_pitched(clips):
    pcm = decode.audio_pcm(clips["count"], sound.SR)
    tags = sound.tag(pcm)
    hits = sound.hits(pcm, tags, sound.speech_spans(tags))
    assert len(hits) == 1
    assert np.allclose(hits[0].times, [1.5, 2.9, 4.0, 7.0, 9.5], atol=0.05)
    assert "Beep" in hits[0].label
    assert "~1000 Hz" in hits[0].quality or "mid-pitched" in hits[0].quality


def test_toast_lifetime_and_keystrokes(clips):
    evs = events.detect(grid(clips["ui_flow"]))
    shown = [e for e in evs if e.shown_until is not None and e.region.colour in ("red", "pink")]
    assert shown, "the error toast was not linked to its disappearance"
    assert abs((shown[-1].shown_until - shown[-1].start) - 0.7) < 0.15
    pcm = decode.audio_pcm(clips["ui_flow"], sound.SR)
    tags = sound.tag(pcm)
    keys = max(sound.hits(pcm, tags, []), key=lambda h: len(h.times))
    assert len(keys.times) == 38


def test_silence_is_broken_by_clicks(clips):
    pcm = decode.audio_pcm(clips["ui_flow"], sound.SR)
    at = audio.build_track(pcm)
    tags = sound.tag(pcm)
    clicks = [t for h in sound.hits(pcm, tags, []) for t in h.times]
    for s in sound.silences(at.rms_db, at.times, clicks):
        assert not any(s.start < t < s.end for t in clicks), "a silence was claimed over a click"


LIBRI = Path(__file__).resolve().parent.parent / "samples" / "librispeech" / "LibriSpeech" / "dev-clean"


@pytest.mark.skipif(not LIBRI.exists(), reason="LibriSpeech dev-clean sample not fetched (samples/fetch.sh)")
def test_voice_letters_follow_real_speakers():
    import random
    import voices as V   # bench/sense/voices.py
    pcm, spans, truth = V.dialogue(("251", "1272", "1462"), random.Random(5))
    got = sound.voices(pcm, spans)
    assert V.pairwise(truth, got) == 1.0
