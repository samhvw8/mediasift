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
