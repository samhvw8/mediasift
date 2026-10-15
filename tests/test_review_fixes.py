"""Regression tests for the code review of v0.3 (VFR, rotation, ASR cue merging, audio-only look).
Fixtures are built with ffmpeg at test time, so nothing binary is committed."""
import subprocess

import numpy as np
import pytest

from framesift import decode, see
from framesift.transcript import Cue


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    # Variable frame rate: black at 0 s, white at 10 s, held to 15 s.
    for c in ("black", "white"):
        ff("-f", "lavfi", "-i", f"color=c={c}:s=160x90:d=1", "-frames:v", "1", str(d / f"{c}.png"))
    (d / "list.txt").write_text(f"file '{d}/black.png'\nduration 10\nfile '{d}/white.png'\nduration 5\nfile '{d}/white.png'\n")
    ff("-f", "concat", "-safe", "0", "-i", str(d / "list.txt"), "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(d / "vfr.mp4"))
    # A landscape stream tagged as rotated 90 degrees (a phone held upright).
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:d=2", "-pix_fmt", "yuv420p", str(d / "land.mp4"))
    ff("-display_rotation", "90", "-i", str(d / "land.mp4"), "-c", "copy", str(d / "rot.mp4"))
    ff("-f", "lavfi", "-i", "sine=f=440:d=3", str(d / "tone.wav"))
    return d


def test_vfr_frames_are_what_is_on_screen(media):
    got = dict(decode.frames_at(str(media / "vfr.mp4"), [5.0, 9.9, 12.0], 160))
    assert got[5.0].mean() < 30 and got[9.9].mean() < 30, "before the switch the screen is black"
    assert got[12.0].mean() > 200


def test_rotation_swaps_the_reported_size(media):
    info = decode.probe(str(media / "rot.mp4"))
    assert (info.width, info.height) == (360, 640)
    t, rgb = next(decode.frames_at(str(media / "rot.mp4"), [1.0], 360))
    assert rgb.shape[0] > rgb.shape[1], "portrait frames stay portrait"


def test_asr_cues_are_not_merged_like_subtitles():
    cues = [Cue(1, 2, "Ready?"), Cue(40, 41, "Ready?"), Cue(50, 52, "Is the answer no"), Cue(53, 54, "no")]
    assert len(see._dedupe(cues, subtitles=False)) == 4
    assert len(see._dedupe([Cue(1, 2, "hello"), Cue(1.5, 3, "hello world")], subtitles=True)) == 2


def test_look_works_on_audio_only(media):
    out = see.look(str(media / "tone.wav"), "1-2")
    assert out.exists()


@pytest.fixture(scope="module")
def portrait(tmp_path_factory):
    d = tmp_path_factory.mktemp("portrait")
    ff("-f", "lavfi", "-i", "testsrc2=s=360x640:d=14:r=15", "-pix_fmt", "yuv420p", str(d / "reel.mp4"))
    return see.see(str(d / "reel.mp4"), force=True, asr_mode="off")


def test_portrait_flow_tiles_are_tall_and_filled(portrait):
    from PIL import Image
    sheet = np.asarray(Image.open(portrait / "flow_01.jpg").convert("L"), np.float32)
    h, w = sheet.shape
    assert h / w > 0.5, f"a 9:16 reel got a landscape sheet layout {w}x{h}"
    dark_cols = (sheet[40:].mean(axis=0) < 20).mean()       # skip the label row
    assert dark_cols < 0.15, f"{dark_cols:.0%} of the sheet's columns are letterbox"


def test_event_sheets_match_the_card(portrait):
    card = (portrait / "card.md").read_text()
    files = sorted(p.name for p in portrait.glob("ev_*.jpg"))
    assert ("ev_NN.jpg" in card) == bool(files)


def test_camera_crops_are_all_referenced(portrait, tmp_path):
    s = _sensed(portrait)
    s.content = "camera"
    refs = see.event_sheets(s.source, s, tmp_path)
    listed_ids = {e.id for grp in see.listed(s)[0] for e in grp}
    assert set(refs) <= listed_ids, "camera footage got crops of events the card does not list"


def _sensed(d):
    import json
    from framesift import events as ev
    js = json.loads((d / "sensed.json").read_text())
    s = see.Sensed(**{k: js[k] for k in see.Sensed.__dataclass_fields__})
    s.events = [ev.Event(**{**e, "region": ev.Region(**e["region"])}) for e in js["events"]]
    return s


def test_flow_tile_ids_are_listed_in_the_card(portrait, monkeypatch):
    import re
    from types import SimpleNamespace
    s = _sensed(portrait)
    s.content = "camera"                      # the card lists no plain pixel events for camera footage
    labels = []
    monkeypatch.setattr(see, "_label", lambda im, text, size=18: labels.append(text))
    see.flow_sheets(s.source, s, SimpleNamespace(), portrait)
    shown = {e.id for grp in see.listed(s)[0] for e in grp}
    tile_ids = {int(m) for t in labels for m in re.findall(r"#(\d+)", t)}
    assert s.events and tile_ids <= shown, f"tiles name events the card does not list: {sorted(tile_ids - shown)}"


def test_one_voice_is_said_once_not_on_every_line(portrait):
    s = _sensed(portrait)
    s.speech = [{"start": i * 3.0, "end": i * 3.0 + 2, "text": f"line {i}", "voice": "male"} for i in range(3)]
    card = see.card(s, [], {}, "x.mp4")
    assert "[voice" not in card
    assert "One voice" in card
    s.speech[1]["voice"] = "B, female"
    s.speech[0]["voice"] = s.speech[2]["voice"] = "A, male"
    assert "[voice A, male]" in see.card(s, [], {}, "x.mp4")


def test_platform_watermark_is_named_once_not_per_jump(portrait):
    """A TikTok watermark jumps corners; each jump read as new on-screen text on the real reels."""
    from framesift import events as ev
    s = _sensed(portrait)
    s.content = "camera"
    r = ev.Region(0.6, 0.8, 0.9, 0.9, 0.02)
    s.events = [ev.Event(1, "change", 4.6, 4.7, 4.5, 4.8, 4.6, r, text_added=["TikTok", "• @tu.phu.ngo triet:"]),
                ev.Event(2, "change", 160.3, 160.4, 160.2, 160.5, 160.3, r, text_added=["@ tu.phu.ngo.trie..."]),
                ev.Event(3, "change", 20.9, 21.0, 20.8, 21.1, 20.9, r, text_added=["vá huhu cứu"])]
    card = see.card(s, [], {}, "x.mp4")
    assert "TikTok" not in card.split("## Timeline")[1], "watermark repeated in the timeline"
    assert "Watermark" in card and "@tu.phu.ngo" in card.split("## Timeline")[0]
    assert "vá huhu cứu" in card, "real on-screen text must stay"


def test_remote_asr_overlaps_the_picture_pass(tmp_path, monkeypatch):
    """The box does the speech work; the Mac only waits on ssh, so that wait runs beside the
    picture pass instead of after it (11.5 s of a 29 s card on a real 2:44 reel)."""
    import time
    from framesift import asr
    ff("-f", "lavfi", "-i", "testsrc2=s=320x180:d=4:r=15", "-f", "lavfi", "-i", "sine=f=300:d=4",
       "-shortest", "-pix_fmt", "yuv420p", str(tmp_path / "a.mp4"))
    marks = {}

    def fake_remote(pcm, host, language=None, connect_s=3):
        marks["asr_start"] = time.perf_counter()
        time.sleep(0.5)
        return asr.AsrResult([Cue(0.5, 2.0, "hello")], "fake", "m", "en", 0.5)

    real_visual = see._visual
    def timed_visual(*a):
        r = real_visual(*a)
        marks["visual_end"] = time.perf_counter()
        return r
    monkeypatch.setattr(asr, "remote", fake_remote)
    monkeypatch.setattr(see, "_visual", timed_visual)
    monkeypatch.setenv("FRAMESIFT_ON", "fakehost")
    s, _ = see.sense(tmp_path / "a.mp4", tmp_path, asr_mode="on")
    assert [c["text"] for c in s.speech] == ["hello"]
    assert marks["asr_start"] < marks["visual_end"], "speech waited for the picture pass"


def test_a_failing_box_falls_back_to_local_speech(tmp_path, monkeypatch):
    """Real case: ComfyUI held 10.6 of 12 GB on the GPU box, whisper hit CUDA out of memory,
    and the whole card crashed. A busy box must cost speed, not the card."""
    from framesift import asr
    ff("-f", "lavfi", "-i", "testsrc2=s=320x180:d=4:r=15", "-f", "lavfi", "-i", "sine=f=300:d=4",
       "-shortest", "-pix_fmt", "yuv420p", str(tmp_path / "a.mp4"))

    def busy_box(pcm, host, language=None, connect_s=3):
        raise RuntimeError("ASR on box failed (1): CUDA failed with error out of memory")

    monkeypatch.setattr(asr, "remote", busy_box)
    monkeypatch.setattr(asr, "transcribe", lambda pcm, *a, **k: asr.AsrResult([Cue(0.5, 2.0, "local")], "local", "m", "en", 0.1))
    monkeypatch.setenv("FRAMESIFT_ON", "fakehost")
    s, _ = see.sense(tmp_path / "a.mp4", tmp_path, asr_mode="on")
    assert [c["text"] for c in s.speech] == ["local"]


def test_long_talk_lists_speaker_letters_not_voice_strings(portrait):
    # A 1h45m podcast read "[voices A, male, B, male, male]": the letters and genders ran together.
    s = _sensed(portrait)
    vs = ["A, male", "B, male", "male", None, "B, male"]
    s.speech = [{"start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"line {i}", "voice": vs[i % 5]} for i in range(200)]
    card = see.card(s, [], {}, "x.mp4")
    assert "[voices A, B]" in card and "male, male" not in card
    assert "Voices: A, male; B, male." in card


def test_transcript_lines_carry_the_speaker_letter(portrait):
    s = _sensed(portrait)
    s.speech = [{"start": 0.0, "end": 2.0, "text": "hello", "voice": "A, male"},
                {"start": 2.0, "end": 4.0, "text": "hi", "voice": "B, female"},
                {"start": 4.0, "end": 4.5, "text": "ok", "voice": "male"}]
    assert see.transcript_text(s).splitlines() == ["[00:00.0-00:02.0] A: hello", "[00:02.0-00:04.0] B: hi",
                                                   "[00:04.0-00:04.5] ok"]
    s.speech = [dict(c, voice="A, male") for c in s.speech]
    assert see.transcript_text(s).splitlines()[0] == "[00:00.0-00:02.0] hello"   # one speaker: no letters
