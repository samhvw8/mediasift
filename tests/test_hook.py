"""The prompt hook names real media files and video URLs, and stays silent otherwise."""
from pathlib import Path

from mediasift.hook import candidates

ROOT = Path(__file__).resolve().parent.parent


def test_finds_paths_urls_and_ignores_the_rest(tmp_path):
    clip = tmp_path / "my clip.mp4"
    clip.write_bytes(b"x")
    other = tmp_path / "rec.mov"
    other.write_bytes(b"x")
    assert candidates(f'what happens in "{clip}" at the end?', "/") == [str(clip.resolve())]
    assert candidates("look at rec.mov please", str(tmp_path)) == [str(other.resolve())]
    assert candidates("what happens in rec.mov?", str(tmp_path)) == [str(other.resolve())]
    assert candidates("see rec.mov. thanks", str(tmp_path)) == [str(other.resolve())]
    assert candidates("summarise https://www.youtube.com/watch?v=abc123.", "/") == ["https://www.youtube.com/watch?v=abc123"]
    assert candidates("fix the bug in parser.py", "/") == []
    assert candidates("the file missing.mp4 is gone", str(tmp_path)) == []


def test_odd_text_never_raises():
    assert candidates("~nobody_here.mp4 and \x00x.mp4", "/") == []
