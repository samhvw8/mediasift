"""`see URL` downloads: a subtitle failure must never cost the video.

Real failures (session logs, 2026-10-13): yt-dlp asked for YouTube's auto-translated captions
('vi' on an English video, 'en' on a Vietnamese one), got HTTP 429 for them, and aborted before
the video. A maintainer confirms only translated tracks get the 429 and `--ignore-errors` keeps the
download going (yt-dlp issue 13831). The fake below behaves that way.
"""
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from mediasift import see

FAKE = r'''#!{python}
import re, sys, json, os
args = sys.argv[1:]
open(os.environ["FAKE_ARGV"], "a").write(json.dumps(args) + "\n")
tracks = json.loads(os.environ.get("FAKE_TRACKS", '{{"en-orig": false, "en": false, "vi": true}}'))  # name -> translated
if "--print-to-file" in args and os.environ.get("FAKE_LANG"):
    i = args.index("--print-to-file")
    open(args[i + 2], "w").write(os.environ["FAKE_LANG"] + "\n")
pats = args[args.index("--sub-langs") + 1].split(",") if "--sub-langs" in args else []
asked = [t for t in tracks if any(re.fullmatch(p, t) for p in pats)]
out = args[args.index("-o") + 1]
ignore = "-i" in args or "--ignore-errors" in args
for t in asked:
    if tracks[t]:
        print(f"ERROR: Unable to download video subtitles for '{{t}}': HTTP Error 429: Too Many Requests", file=sys.stderr)
        if not ignore:
            sys.exit(1)                        # real yt-dlp: the video is never downloaded
    else:
        open(out.replace("%(ext)s", t + ".vtt"), "w").write("WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n")
open(out.replace("%(ext)s", "mp4"), "wb").write(b"\0")
sys.exit(1 if any(tracks[t] for t in asked) else 0)
'''


@pytest.fixture
def fake_ytdlp(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    exe = bin_ / "yt-dlp"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    node = bin_ / "node"
    node.write_text("#!/bin/sh\n")
    node.chmod(node.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("MEDIASIFT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.jsonl"))
    return tmp_path / "argv.jsonl"


def test_a_translated_subtitle_429_does_not_lose_the_video(fake_ytdlp):
    video, subs = see.fetch("https://www.youtube.com/watch?v=abc")
    assert video.name == "video.mp4" and video.exists()
    assert subs is not None and subs.name == "video.en-orig.vtt"     # the original captions, not a translation


def test_the_installed_js_runtime_is_passed_to_ytdlp(fake_ytdlp):
    """Without one, YouTube extraction 'has been deprecated, and some formats may be missing'."""
    see.fetch("https://www.youtube.com/watch?v=abc")
    args = __import__("json").loads(fake_ytdlp.read_text().splitlines()[0])
    pairs = [args[i + 1] for i, a in enumerate(args) if a == "--js-runtimes"]
    assert "node" in pairs


# YouTube auto-dubs: each dubbed audio track brings its own '-orig' captions (hz639bJjCr0, an English
# talk, came with de-DE-orig, es-US-orig, fr-FR-orig ... and the picker took German).
DUBBED = '{"de-DE-orig": false, "en-orig": false, "en": false, "fr-FR-orig": false, "vi": true}'


def test_captions_follow_the_video_language_not_a_dub(fake_ytdlp, monkeypatch):
    monkeypatch.setenv("FAKE_TRACKS", DUBBED)
    monkeypatch.setenv("FAKE_LANG", "en-US")
    _, subs = see.fetch("https://www.youtube.com/watch?v=dub")
    assert subs is not None and subs.name == "video.en-orig.vtt"


def test_unknown_language_with_several_caption_languages_uses_speech_recognition(fake_ytdlp, monkeypatch):
    """Guessing a caption language could print another language's words as the speech."""
    monkeypatch.setenv("FAKE_TRACKS", DUBBED)
    _, subs = see.fetch("https://www.youtube.com/watch?v=nolang")
    assert subs is None
