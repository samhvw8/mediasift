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

from mediasift import audio, decode, events, sound  # noqa: E402


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
    from mediasift import see
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


def _music_tags(n: int, mood: dict[int, float], genre: dict[int, float] = {}, hop: float = 0.5) -> sound.Tags:
    names = [""] * 527
    names[137] = "Music"
    for i in (*mood, *genre):
        names[i] = {276: "Happy music", 278: "Sad music", 279: "Tender music", 282: "Scary music",
                    239: "Electronic music", 253: "New-age music"}[i]
    probs = np.zeros((n, 527), np.float16)
    probs[:, 137] = 0.7
    for i, p in (*mood.items(), *genre.items()):
        probs[:, i] = p
    return sound.Tags(np.arange(n) * hop + 1.5, probs, names, hop)


def test_music_bed_says_its_mood_and_genre():
    """CED's own AudioSet mood and genre classes, on 742 held-out AudioSet eval clips: a mood scored
    >= 0.10 was right 88% of the time (7 moods), a genre >= 0.15 83% (9 genres). Free: same tagger."""
    bed = sound.beds(_music_tags(20, {282: 0.2}, {239: 0.2}), [])[0]
    assert bed.quality == "scary, electronic"
    assert "[music] (scary, electronic)" in __import__("mediasift.see").see._sound_line(bed)[1]


def test_weak_music_mood_is_hedged_and_faint_one_left_out():
    """Moods at 0.05-0.10 were right ~65% of the time (chance 14%): 'maybe'. Below 0.05: not said."""
    assert sound.beds(_music_tags(20, {278: 0.07}), [])[0].quality == "maybe sad"
    assert sound.beds(_music_tags(20, {278: 0.03}), [])[0].quality == ""


def test_long_music_bed_carries_its_mood_arc():
    """A score that turns from tender to scary is where a film turns: a long bed lists the stretches."""
    tags = _music_tags(240, {279: 0.2})                     # 120 s at a 0.5 s hop
    tags.probs[120:, 279] = 0
    tags.names[282] = "Scary music"
    tags.probs[120:, 282] = 0.2
    bed = sound.beds(tags, [])[0]
    assert [m[2] for m in bed.moods] == ["tender", "scary"]
    assert 55 < bed.moods[1][0] < 65
    line = __import__("mediasift.see").see._sound_line(bed)[1]
    assert "(tender 00:01.2-01:01.2, then scary 01:01.2-02:01.2)" in line, line


def test_music_merged_on_and_off_keeps_its_mood():
    """Tears of Steel: a score that drops out under dialogue merged into one bed and lost the
    'scary' its later stretch carried."""
    from mediasift import see
    a = sound.SoundEvent(10.0, 20.0, "music", "bed", score=0.6)
    b = sound.SoundEvent(23.0, 40.0, "music", "bed", score=0.6, quality="scary")
    c = sound.SoundEvent(43.0, 50.0, "music", "bed", score=0.6, quality="sad")
    merged = see._merge_beds([a, b, c])
    assert len(merged) == 1 and merged[0].quality == "scary, then sad"


def test_music_bed_reports_harmony_rhythm_and_beat_sync():
    """Music beds >= 5 s carry key/mode, harmonic shift, BPM, and cut-to-beat sync on the card."""
    from mediasift import see
    sr = 16000

    def _prog(root: int, minor: bool, sec: float) -> np.ndarray:
        y = np.zeros(int(sec * sr), np.float32)
        chords = [[0, 3, 7], [8, 12, 15], [5, 8, 12], [7, 10, 14]] if minor else [[0, 4, 7], [7, 11, 14], [9, 12, 16], [5, 9, 12]]
        for b in range(int(sec / 0.5)):
            t0 = int(b * 0.5 * sr)
            t = np.arange(int(0.48 * sr), dtype=np.float32) / sr
            ch = chords[(b // 2) % 4]
            sig = sum(np.sin(2 * np.pi * (440.0 * 2 ** ((root + o - 69) / 12)) * t) for o in ch) * 0.15 * np.exp(-1.5 * t)
            kick = 0.5 * np.sin(2 * np.pi * (60.0 + 80.0 * np.exp(-30 * t)) * t) * np.exp(-18 * t)
            y[t0:t0 + len(t)] += sig + kick
        return y

    pcm = np.clip(np.concatenate([_prog(48, True, 6.0), _prog(51, False, 6.0)]) * 32768.0, -32768, 32767).astype(np.int16)
    bed = sound.SoundEvent(0.0, 12.0, "music", "bed", score=0.8, quality="exciting")
    cuts = [2.0, 4.0, 6.0, 8.0, 10.0]
    sound.music_harmony_rhythm(pcm, [bed], cuts)
    assert bed.harmony["harmony_arc"][0][2] == "minor"
    assert bed.harmony["harmony_arc"][1][2] == "major"
    assert abs(bed.rhythm["bpm"] - 120.0) <= 3.0
    assert bed.beat_sync["synced"] and bed.beat_sync["hits"] == 5
    line = see._sound_line(bed)[1]
    assert "minor -> major" in line and "~120 BPM" in line and "cuts sync 5/5" in line, line

def test_talk_emotion_segments_normalizes_per_speaker_and_stays_silent_on_calm():
    """Speaker-relative segment emotion flags heated/subdued shifts and emits nothing on steady talk."""
    from mediasift import see
    t = np.arange(16000 * 30, dtype=np.float32) / 16000.0
    # Baseline 0-12s: moderate 160 Hz voice; 12-20s: heated (+12 dB + strong upper harmonics); 20-30s: subdued (-8 dB)
    y = np.zeros_like(t)
    base_wave = 0.08 * np.sin(2 * np.pi * 160.0 * t) + 0.03 * np.sin(2 * np.pi * 320.0 * t)
    y[:16000 * 12] = base_wave[:16000 * 12]
    y[16000 * 12:16000 * 20] = (0.32 * np.sin(2 * np.pi * 210.0 * t[16000 * 12:16000 * 20])
                                + 0.28 * np.sin(2 * np.pi * 840.0 * t[16000 * 12:16000 * 20]))
    y[16000 * 20:] = 0.025 * np.sin(2 * np.pi * 145.0 * t[16000 * 20:])
    pcm = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
    speech = [
        {"start": 0.5, "end": 3.5, "text": "Here is the normal baseline introduction.", "voice": "A"},
        {"start": 4.0, "end": 7.5, "text": "We continue explaining the regular setup.", "voice": "A"},
        {"start": 8.0, "end": 11.5, "text": "Everything is running at normal speed.", "voice": "A"},
        {"start": 12.5, "end": 19.5, "text": "Stop! That is completely wrong and broken!", "voice": "A"},
        {"start": 20.5, "end": 23.5, "text": "I don't know... maybe we missed it.", "voice": "A"},
        {"start": 25.0, "end": 28.5, "text": "Sorry... I am unsure now.", "voice": "A"},
    ]
    segs = sound.talk_emotion_segments(pcm, None, speech)
    emos = {s["emotion"] for s in segs}
    assert "heated" in emos and "subdued" in emos, segs
    assert all(s["start"] >= 12.0 for s in segs), segs

    # Steady calm talk across all segments must emit 0 emotion flags
    pcm_calm = np.clip(base_wave * 32768.0, -32768, 32767).astype(np.int16)
    speech_calm = [
        {"start": k * 4.0 + 0.5, "end": k * 4.0 + 3.5, "text": "Normal steady narration.", "voice": "A"}
        for k in range(6)
    ]
    assert sound.talk_emotion_segments(pcm_calm, None, speech_calm) == []


def test_look_say_matches_word_timestamps_subwords_vietnamese_and_fallback():
    """`look --say PHRASE` resolves exact word spans, merges SentencePiece subwords, matches
    accent-stripped Vietnamese, falls back to cue interpolation when `words` is absent, and
    lists the 3 closest lines when a phrase is absent."""
    import pytest
    from mediasift import see

    speech = [
        {
            "start": 3.92,
            "end": 8.88,
            "text": "If you're not sure what you're going to do the next day, then go to sleep on time.",
            "words": [
                {"start": 3.92, "end": 4.2, "text": " If"},
                {"start": 6.24, "end": 6.4, "text": " go"},
                {"start": 6.4, "end": 6.56, "text": " to"},
                {"start": 6.56, "end": 6.8, "text": " sl"},
                {"start": 6.8, "end": 6.96, "text": "eep"},
                {"start": 6.96, "end": 7.12, "text": " on"},
                {"start": 7.12, "end": 7.44, "text": " time."},
            ],
        },
        {
            "start": 10.0,
            "end": 14.0,
            "text": "Hôm nay chúng ta cùng tìm hiểu câu chuyện lịch sử thú vị này.",
            "words": [
                [10.0, 10.5, "Hôm"],
                [10.5, 11.0, "nay"],
                [11.8, 12.2, "câu"],
                [12.2, 12.6, "chuyện"],
                [12.6, 13.0, "lịch"],
                [13.0, 13.4, "sử"],
            ],
        },
        {
            "start": 15.0,
            "end": 21.0,
            "text": "Start building an audience now even if you have no product to sell.",
        },
    ]
    # 1. Subword merge + mid-cue word span ("go to sleep on time" inside 3.92-8.88s cue)
    a, b, (ws, we) = see._match_say(speech, "go to sleep on time", 25.0)
    assert (ws, we) == (6.24, 7.44)
    assert 5.8 <= a <= 6.24 and 7.44 <= b <= 8.0

    # 2. Accent-stripped Vietnamese match ("cau chuyen lich su" -> 11.8..13.4s)
    a_vi, b_vi, (ws_vi, we_vi) = see._match_say(speech, "cau chuyen lich su", 25.0)
    assert (ws_vi, we_vi) == (11.8, 13.4)

    # 3. Cue-level interpolation fallback when `words` is absent
    a_fb, b_fb, (ws_fb, we_fb) = see._match_say(speech, "building an audience", 25.0)
    assert 15.0 <= ws_fb < we_fb <= 18.0

    # 4. Missing phrase raises ValueError with closest transcript lines
    with pytest.raises(ValueError, match="closest lines"):
        see._match_say(speech, "quantum entanglement spacecraft", 25.0)


def test_modes_pass_toggles_adaptive_classification_and_hook(clips, tmp_path, monkeypatch):
    """`--mode`, `--with`, `--without`, `classify_signal`, incremental cache upgrade, and `hook.suggest_mode`."""
    from mediasift import hook, see

    assert set(see.resolve_passes("podcast")) == {"sounds", "speech", "voices", "emotion"}
    assert set(see.resolve_passes("podcast", with_passes="ocr", without_passes="emotion")) == {
        "pic", "ocr", "sounds", "speech", "voices",
    }
    assert set(see.resolve_passes("probe")) == {"pic", "sounds", "speech"}
    assert set(see.resolve_passes("music")) == {"pic", "sounds", "harmony", "speech"}
    assert set(see.resolve_passes("reel")) == {"pic", "ocr", "sounds", "speech"}

    # Signal classification: screen, reel, music, and podcast
    s_screen = see.Sensed("u.mp4", 60.0, 1280, 720, 30.0, 15.0, "screen")
    assert see.classify_signal(s_screen)[0] == "screen"
    s_reel = see.Sensed("r.mp4", 30.0, 720, 1280, 30.0, 15.0, "camera")
    assert see.classify_signal(s_reel)[0] == "reel"
    s_music = see.Sensed("m.mp4", 120.0, 1280, 720, 30.0, 15.0, "camera",
                         sounds=[sound.SoundEvent(0.0, 90.0, "music", "bed", score=0.8)])
    assert see.classify_signal(s_music)[0] == "music"
    s_pod = see.Sensed("p.mp4", 600.0, 1280, 720, 30.0, 15.0, "camera",
                       speech=[{"start": 0.0, "end": 400.0, "text": "long talk"}])
    assert see.classify_signal(s_pod)[0] == "podcast"

    # Hook mode suggestions
    assert hook.suggest_mode("ep12_podcast.mp4", "summarize this interview") == "podcast"
    assert hook.suggest_mode("https://youtube.com/shorts/abc123", "what is this brain rot reel") == "reel"
    assert hook.suggest_mode("theme_song.mp4", "check the music beat sync") == "music"
    assert hook.suggest_mode("bug_repro.mov", "ui walkthrough") == "screen"
    assert hook.suggest_mode("mystery.mp4", "what happens here") == "adaptive"

    # Incremental cache & view filtering on synth count clip
    real_models = sound.cache_root() / "models"
    if real_models.exists():
        (tmp_path / "models").symlink_to(real_models)
    monkeypatch.setenv("MEDIASIFT_CACHE", str(tmp_path))
    out = see.see(clips["count"], asr_mode="off", mode="podcast")
    card_pod = (out / "card.md").read_text()
    assert " pic " not in card_pod
    assert "beep x5" in card_pod
    # Upgrade with `--with pic` reuses audio and adds picture events
    out2 = see.see(clips["count"], asr_mode="off", mode="podcast", with_passes="pic")
    card_upg = (out2 / "card.md").read_text()
    assert "flashed (changed and changed back) 8 times" in card_upg
    assert "beep x5" in card_upg


def test_indra_ocr_fallback_and_box_cropping(clips, tmp_path, monkeypatch):
    """When Apple Vision is absent (Linux), `ocr` falls back to Indra OCR for full frames,
    region crops (with frame-relative coordinates), and `see()` event diffs."""
    from PIL import Image, ImageDraw
    from mediasift import ocr, see
    from mediasift.see import font

    if not ocr.indra_available():
        import pytest
        pytest.skip("indra-ocr not installed")

    monkeypatch.setattr(ocr, "vision_available", lambda: False)
    assert ocr.backend("auto") == "indra"
    assert ocr.available() is True

    img = tmp_path / "code_frame.png"
    im = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(im)
    f = font(28)
    d.text((40, 40), "TOP HEADER 42", fill="black", font=f)
    d.text((40, 240), "def spam():", fill="black", font=f)
    im.save(img)

    full_lines = ocr.read(img)
    texts = [t for t, _ in full_lines]
    assert any("TOP HEADER 42" in t for t in texts), texts
    assert any("def spam():" in t for t in texts), texts

    crop_lines = ocr.read(img, box=(0.0, 0.5, 1.0, 1.0))
    crop_texts = [t for t, _ in crop_lines]
    assert any("def spam():" in t for t in crop_texts), crop_texts
    assert not any("TOP HEADER" in t for t in crop_texts), crop_texts
    for _, (x0, y0, x1, y1) in crop_lines:
        assert 0.5 <= y0 < y1 <= 1.0

    # End-to-end screen clip through Indra fallback (simulating Linux)
    import json
    real_models = sound.cache_root() / "models"
    if real_models.exists():
        (tmp_path / "models").symlink_to(real_models)
    monkeypatch.setenv("MEDIASIFT_CACHE", str(tmp_path))
    out = see.see(clips["ui_flow"], asr_mode="off")
    js = json.loads((out / "sensed.json").read_text())
    card = (out / "card.md").read_text()
    assert js["ocr"] is True
    assert "on-screen text: OCR on changed regions" in card
    assert any(e.get("text_added") for e in js["events"])


def test_packet_activity_and_look_active_crop_row(clips, tmp_path, monkeypatch):
    """`decode.packet_activity` profiles P/B byte ratios without decoding frames, and `look()`
    snaps to cached `frame_peaks` and appends a Codec Canvas active-region zoom row when a
    localized motion region is present."""
    import json
    from mediasift import see, sound
    real_models = sound.cache_root() / "models"
    if real_models.exists():
        (tmp_path / "models").symlink_to(real_models)
    from PIL import Image
    pa = decode.packet_activity(clips["motion"])
    assert pa["duration"] > 0
    assert len(pa["bin_ratios"]) >= 1
    assert pa["median_pb"] > 0

    monkeypatch.setenv("MEDIASIFT_CACHE", str(tmp_path))
    out = see.see(clips["motion"], asr_mode="off")
    js = json.loads((out / "sensed.json").read_text())
    assert len(js.get("frame_peaks") or []) >= 4

    sheet_path = see.look(clips["motion"], span="0.2-3.8", n=8)
    with Image.open(sheet_path) as im:
        # 8 frames in 4 cols -> 2 full-frame rows (2 * 180 = 360) + 1 active crop row (180) + sound strip (120) = 660
        assert im.width == 4 * see.TILE_W
        assert im.height >= 3 * see.TILE_H


def test_scan_persistent_text_and_layout_on_card(tmp_path):
    """`see.card` surfaces `Layout:` when non-full and `Lasting text (persistent banner/overlay):`
    when `s.lasting_text` holds static headline/banner text."""
    from mediasift import see
    s = see.Sensed("reel.mp4", 12.0, 576, 1024, 30.0, 15.0, "camera", ocr=True)
    s.layout = {"kind": "letterbox+split-vertical", "viewport": [0.0, 0.25, 1.0, 0.75], "active_ratio": 0.5, "panels": 2}
    s.lasting_text = [{"text": "QUỐC TẾ ĐÀN ÔNG", "zone": "top", "start": 0.0, "end": 12.0, "box": [0.1, 0.15, 0.9, 0.25]}]
    md = see.card(s, ["flow_01.jpg (00:00.5-00:11.5)"], {}, "reel.mp4")
    assert "Layout: letterbox+split-vertical (active viewport 0%-100% x 25%-75%, 2 panels)." in md
    assert 'Lasting text (persistent banner/overlay): top: "QUỐC TẾ ĐÀN ÔNG" (00:00.0-00:12.0).' in md
