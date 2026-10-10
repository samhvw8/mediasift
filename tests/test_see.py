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
    for make in (make_synth.clip_count, make_synth.clip_ui, make_synth.clip_motion):
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


def test_short_low_sounds_are_not_swallowed_as_music(clips):
    """With 3 s tagging windows a lone low tone reads as 'music'; it must still be a counted hit."""
    pcm = decode.audio_pcm(clips["ui_flow"], sound.SR)
    tags = sound.tag(pcm)
    hits = sound.hits(pcm, tags, [])
    assert any(abs(h.times[0] - 11.8) < 0.1 and "low" in h.quality for h in hits), [(h.label, h.times[:3]) for h in hits]


def test_speech_syllables_are_not_counted_as_hits():
    """Under a voice, syllables the 3 s tagger did not mark as speech came back as 'hits'
    (49 stray hit times on the 200-clip speech-bed bench). Real LibriSpeech speech alone
    must yield no hits."""
    import random
    import voices as V
    if not V.LS.exists():
        pytest.skip("LibriSpeech sample not fetched")
    pcm, spans, _ = V.dialogue(("251", "1272"), random.Random(2))
    tags = sound.tag(pcm)
    hits = sound.hits(pcm, tags, [])          # no speech spans given: the hit test itself must know speech
    n = sum(len(h.times) for h in hits)
    assert n <= 2, f"{n} syllables counted as hits: {[(h.label, h.times[:4]) for h in hits]}"


def test_weak_lone_hit_inside_music_is_dropped_but_kept_in_silence():
    """Under a real song, drum and pluck transients the tagger could not name came back as lone
    'sharp sound' hits (16 stray lines on the 200-clip song bench, no right name among them)."""
    sr = sound.SR
    pcm = np.zeros(4 * sr, np.int16)
    k = int(2.0 * sr)
    pcm[k:k + 80] = (np.hanning(80) * 20000).astype(np.int16)        # one click, nothing else
    names = [""] * 527
    music = sorted(sound.MUSIC)[0]
    names[music] = "Music"
    probs = np.zeros((3, 527), np.float16)
    quiet = sound.Tags(np.array([1.5, 2.0, 2.5]), probs.copy(), names, 0.5)
    probs[:, music] = 0.6
    in_music = sound.Tags(np.array([1.5, 2.0, 2.5]), probs, names, 0.5)
    assert sum(len(h.times) for h in sound.hits(pcm, quiet, [])) == 1
    assert sum(len(h.times) for h in sound.hits(pcm, in_music, [])) == 0


def test_a_sound_under_music_is_named_at_a_lower_score():
    """Beside 'Music', CED scores a real sound 0.2-0.3 (thunder, ticks, a creak under the song
    bed): below the threshold set for quiet audio. In silence the threshold stays."""
    names = [""] * 527
    music, thunder = sorted(sound.MUSIC)[0], 300
    names[music], names[thunder] = "Music", "Thunder"
    probs = np.zeros((8, 527), np.float16)
    probs[:, thunder] = 0.22
    quiet = sound.Tags(np.arange(8) * 0.5 + 1.5, probs.copy(), names, 0.5)
    probs[:, music] = 0.6
    loud = sound.Tags(np.arange(8) * 0.5 + 1.5, probs, names, 0.5)
    assert not [b for b in sound.beds(quiet, []) if b.label == "Thunder"]
    assert [b for b in sound.beds(loud, []) if "Thunder" in b.label]


def test_low_score_names_are_hedged_on_the_card():
    """Under a real song, names scored below 0.3 were wrong more often than right (23 right,
    30 wrong): the card must say 'maybe', not state them."""
    from framesift import see
    weak = sound.SoundEvent(10.0, 12.0, "Thunder", "bed", score=0.24)
    sure = sound.SoundEvent(20.0, 22.0, "Siren", "bed", score=0.7)
    assert "[maybe thunder]" in see._sound_line(weak)[1]
    assert "[siren]" in see._sound_line(sure)[1]


def test_short_low_score_run_under_music_is_not_named():
    names = [""] * 527
    music, thunder = sorted(sound.MUSIC)[0], 300
    names[music], names[thunder] = "Music", "Thunder"
    probs = np.zeros((8, 527), np.float16)
    probs[:, music] = 0.6
    probs[3:5, thunder] = 0.22                # 1.0 s at a low score: the song bench's stray shape
    tags = sound.Tags(np.arange(8) * 0.5 + 1.5, probs, names, 0.5)
    assert not [b for b in sound.beds(tags, []) if "Thunder" in b.label]


def test_moving_balls_stay_motion_not_drawn_in(clips):
    """The drawn-in rule (net change over most of the region) must not swallow real travel:
    a 0.5 cut-off turned the synthetic balls into 'kept changing in place' and 'changed'."""
    evs = events.detect(grid(clips["motion"]))
    big = [e for e in evs if e.region.area > 0.015 and e.end - e.start >= 0.9]
    assert [e.kind for e in big if e.start < 9] == ["motion", "motion"], [(e.kind, round(e.start, 1)) for e in big]
