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
