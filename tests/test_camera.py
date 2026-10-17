"""Camera moves: which way the picture travels, which ~1 fps frames cannot show
(real bench, Big Buck Bunny q02 'pan or tilt?', missed in every round)."""
import subprocess

import pytest

from mediasift import decode, events


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
    from mediasift import see
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


def test_shots_that_return_get_the_same_letter(tmp_path):
    """Real bench q07: 'how many times does it cut back to the rodents?' The card listed 34 cut
    times; which shots are the same place has to be in the text."""
    for name, src in (("a", "mandelbrot=s=640x360"), ("b", "testsrc2=s=640x360"), ("c", "smptebars=s=640x360")):
        ff("-f", "lavfi", "-i", src, "-frames:v", "1", str(tmp_path / f"{name}.png"))
    order = "abaca"
    lst = "".join(f"file '{tmp_path / (x + '.png')}'\nduration 2\n" for x in order) + f"file '{tmp_path / 'a.png'}'\n"
    (tmp_path / "list.txt").write_text(lst)
    ff("-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt"), "-r", "30", "-pix_fmt", "yuv420p",
       str(tmp_path / "shots.mp4"))
    g = events.build_grid(decode.video_frames(str(tmp_path / "shots.mp4"), 15, events.DECODE_W, events.DECODE_H))
    cuts = [e for e in events.detect(g) if e.kind == "cut"]
    labels = events.shot_letters(g, [c.start for c in cuts])
    assert "".join(lab for _, lab in labels) == "ABACA", labels


def test_card_cut_line_carries_shot_letters(tmp_path):
    from mediasift import see
    for name, src in (("a", "mandelbrot=s=640x360"), ("b", "testsrc2=s=640x360")):
        ff("-f", "lavfi", "-i", src, "-frames:v", "1", str(tmp_path / f"{name}.png"))
    lst = "".join(f"file '{tmp_path / (x + '.png')}'\nduration 2\n" for x in "aba") + f"file '{tmp_path / 'a.png'}'\n"
    (tmp_path / "list.txt").write_text(lst)
    ff("-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt"), "-r", "30", "-pix_fmt", "yuv420p", str(tmp_path / "s.mp4"))
    s, _ = see.sense(tmp_path / "s.mp4", tmp_path, asr_mode="off")
    card = see.card(s, [], {}, "s.mp4")
    line = next(l for l in card.splitlines() if " pic CUT" in l)
    assert "2.0 B, 4.0 A s" in line, line


def test_a_drawn_in_overlay_is_a_change_with_its_lifetime(tmp_path):
    """python_100s: a red X drawn in over 0.5 s, held 3 s, then removed, came out as 'motion'
    (no lifetime, colour 'black'); the real bench asks how long it stayed."""
    import numpy as np
    from PIL import Image, ImageDraw
    fps, frames = 30, []
    bg = Image.new("RGB", (640, 360), (25, 25, 30))
    for i in range(int(6 * fps)):
        t = i / fps
        im = bg.copy()
        d = ImageDraw.Draw(im)
        if 1.0 <= t < 4.5:
            k = min(1.0, (t - 1.0) / 0.5)          # strokes grow for 0.5 s
            d.line((220, 100, 220 + 200 * k, 100 + 160 * k), fill=(230, 30, 30), width=18)
            if k >= 0.5:
                k2 = (k - 0.5) * 2
                d.line((420, 100, 420 - 200 * k2, 100 + 160 * k2), fill=(230, 30, 30), width=18)
        frames.append(np.asarray(im))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360",
                          "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", str(tmp_path / "x.mp4")], stdin=subprocess.PIPE)
    p.communicate(np.stack(frames).tobytes())
    g = events.build_grid(decode.video_frames(str(tmp_path / "x.mp4"), 15, events.DECODE_W, events.DECODE_H))
    shown = [e for e in events.detect(g) if e.shown_until is not None]
    assert shown, [(e.kind, e.start, e.end) for e in events.detect(g)]
    e = shown[0]
    assert e.kind == "change" and e.region.colour == "red", (e.kind, e.region.colour)
    assert abs((e.shown_until - e.start) - 3.5) < 0.6, (e.start, e.shown_until)


def test_an_overlay_replaced_together_with_what_was_under_it_still_gets_a_lifetime(tmp_path):
    """python_100s again: braces appear, a red X is drawn over them 1.5 s later, then both vanish at
    once. The X's region then shows the page, not the braces: not 'undone', but it is gone."""
    import numpy as np
    from PIL import Image, ImageDraw
    fps, frames = 30, []
    bg = Image.new("RGB", (640, 360), (25, 25, 30))
    for i in range(int(7 * fps)):
        t = i / fps
        im = bg.copy()
        d = ImageDraw.Draw(im)
        if 1.0 <= t < 5.6:
            d.rectangle((180, 80, 460, 280), fill=(235, 235, 235))          # the braces card
        if 2.5 <= t < 5.6:
            d.line((220, 100, 420, 260), fill=(230, 30, 30), width=18)
            d.line((420, 100, 220, 260), fill=(230, 30, 30), width=18)
        frames.append(np.asarray(im))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360",
                          "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", str(tmp_path / "y.mp4")], stdin=subprocess.PIPE)
    p.communicate(np.stack(frames).tobytes())
    g = events.build_grid(decode.video_frames(str(tmp_path / "y.mp4"), 15, events.DECODE_W, events.DECODE_H))
    evs = events.detect(g)
    x = [e for e in evs if e.region.colour == "red" and 2.3 <= e.start <= 2.8]
    assert x and x[0].shown_until is not None, [(e.kind, round(e.start, 2), e.region.colour, e.shown_until) for e in evs]
    assert abs((x[0].shown_until - x[0].start) - 3.1) < 0.4


def test_a_single_cut_back_to_the_opening_place_reads_plainly():
    """The phone reel's one cut is a punch-in on the same scene: 'CUT: new shot A' read as a contradiction."""
    from mediasift import see
    s = see.Sensed("x.mp4", 20.0, 1080, 1920, 30, 15, "camera")
    s.events = [events.Event(1, "cut", 4.3, 4.3, 4.2, 4.4, 4.4, events.Region(0, 0, 1, 1, 1.0))]
    s.shots = [(0.0, "A"), (4.3, "A")]
    assert "CUT: same place again (shot A, as at the start)" in see.card(s, [], {}, "x.mp4")
    s.shots = [(0.0, "A"), (4.3, "B")]
    assert "CUT: new shot B" in see.card(s, [], {}, "x.mp4")
