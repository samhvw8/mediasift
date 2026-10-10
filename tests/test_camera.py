"""Camera moves: which way the picture travels, which ~1 fps frames cannot show
(real bench, Big Buck Bunny q02 'pan or tilt?', missed in every round)."""
import subprocess

import pytest

from framesift import decode, events


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("cam")
    # Detail in every direction (testsrc2 has vertical bars: a tilt over them changes nothing).
    ff("-f", "lavfi", "-i", "mandelbrot=s=1600x1000", "-frames:v", "1", str(d / "big.png"))
    # A 640x360 window over a still picture: x grows -> the content slides left -> the camera pans right.
    ff("-loop", "1", "-i", str(d / "big.png"), "-t", "5", "-r", "30",
       "-vf", "crop=640:360:'100+t*120':200", "-pix_fmt", "yuv420p", str(d / "pan_right.mp4"))
    ff("-loop", "1", "-i", str(d / "big.png"), "-t", "5", "-r", "30",
       "-vf", "crop=640:360:300:'50+t*90'", "-pix_fmt", "yuv420p", str(d / "tilt_down.mp4"))
    ff("-loop", "1", "-i", str(d / "big.png"), "-t", "4", "-r", "30",
       "-vf", "crop=640:360:300:200", "-pix_fmt", "yuv420p", str(d / "still.mp4"))
    return d


def moves(path):
    g = events.build_grid(decode.video_frames(str(path), 15, events.DECODE_W, events.DECODE_H))
    return events.camera_moves(g)


def test_pan_right(clips):
    m = moves(clips / "pan_right.mp4")
    assert len(m) == 1 and m[0]["words"] == "pans right", m
    assert m[0]["start"] < 1.0 and m[0]["end"] > 4.0


def test_tilt_down(clips):
    m = moves(clips / "tilt_down.mp4")
    assert len(m) == 1 and m[0]["words"] == "tilts down", m


def test_still_picture_has_no_move(clips):
    assert moves(clips / "still.mp4") == []


def test_card_says_which_way_the_camera_moved(clips, tmp_path):
    from framesift import see
    s, _ = see.sense(clips / "pan_right.mp4", tmp_path, asr_mode="off")
    s.content = "camera"
    card = see.card(s, [], {}, "pan_right.mp4")
    assert "camera pans right" in card, card


def test_one_editing_jump_is_not_a_pan(tmp_path):
    """A punch-in jump (the frame shifts once) looked like a 1.1 s pan on a real reel."""
    ff("-f", "lavfi", "-i", "mandelbrot=s=1600x1000", "-frames:v", "1", str(tmp_path / "big.png"))
    ff("-loop", "1", "-i", str(tmp_path / "big.png"), "-t", "4", "-r", "30",
       "-vf", "crop=640:360:'if(lt(t,2),300,360)':200", "-pix_fmt", "yuv420p", str(tmp_path / "jump.mp4"))
    assert moves(tmp_path / "jump.mp4") == []
