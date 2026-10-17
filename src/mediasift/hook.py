"""`mediasift hook`: a Claude Code UserPromptSubmit hook that points the agent at a video.

When the prompt names an existing video/audio file or a video URL, it adds one line of context
("run `mediasift see PATH`", and whether a card is already cached). It never senses anything
itself: hooks have a short timeout and run on every prompt. Prompts without a video cost nothing.

settings.json:
  {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "mediasift hook"}]}]}}
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"}
MEDIA = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"} | AUDIO_EXT
# Only links that are videos by construction: x.com status links in past prompts were mostly text posts.
URL = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/|vimeo\.com/\d)\S+")
PATH = re.compile(r"(?:\"([^\"]+)\"|'([^']+)'|((?:~|/|\./|\.\./)?[\w./~\-]+?\.(?:" +
                  "|".join(e[1:] for e in MEDIA) + r")))(?=[\s,;:)\]?!.\'\"]|$)", re.I)

_RE_REEL = re.compile(r"(?:youtube\.com/shorts/|tiktok|\b(?:reel|reels|short|shorts|meme|brain\s*rot)\b)", re.I)
_RE_PODCAST = re.compile(r"\b(?:podcast|interview|lecture|talk|meeting|panel|discussion|radio|voice\s*note)\b", re.I)
_RE_MUSIC = re.compile(r"\b(?:music|song|mv|concert|lyric|karaoke|ost|soundtrack|chord|bpm|beat\s*sync)\b", re.I)
_RE_SCREEN = re.compile(r"\b(?:screen\s*record|screencast|walkthrough|repro|bug|ui\s*flow|terminal)\b", re.I)


def suggest_mode(item: str, prompt: str = "") -> str:
    text = f"{item} {prompt}"
    ext = Path(item).suffix.lower() if not URL.match(item) else ""
    if ext in AUDIO_EXT:
        return "music" if _RE_MUSIC.search(text) else "podcast"
    if _RE_REEL.search(text):
        return "reel"
    if _RE_PODCAST.search(text):
        return "podcast"
    if _RE_MUSIC.search(text):
        return "music"
    if _RE_SCREEN.search(text):
        return "screen"
    return "adaptive"

def candidates(prompt: str, cwd: str) -> list[str]:
    out: list[str] = []
    for m in URL.finditer(prompt):
        out.append(m.group(0).rstrip(".,;)]'\""))
    for m in PATH.finditer(prompt):
        raw = next(g for g in m.groups() if g)
        try:                                    # odd text ("~user.mp4", NUL bytes) is skipped, never fatal
            p = Path(raw.strip()).expanduser()
            p = p if p.is_absolute() else Path(cwd) / p
            if p.suffix.lower() in MEDIA and p.is_file():
                out.append(str(p.resolve()))
        except (OSError, RuntimeError, ValueError):
            continue
    seen: list[str] = []
    for c in out:
        if c not in seen:
            seen.append(c)
    return seen[:3]


def context(items: list[str], prompt: str = "") -> str:
    from .see import URL as IS_URL, out_dir
    lines = []
    for it in items:
        cached = ""
        if not IS_URL.match(it):
            card = out_dir(Path(it)) / "card.md"
            if card.exists():
                cached = f" (already sensed: card at {card})"
        mode = suggest_mode(it, prompt)
        flag = f" --mode {mode}" if mode != "full" else ""
        lines.append(f"Video/audio in the prompt: {it}. You cannot watch or hear it; run "
                     f"`mediasift see {shlex.quote(it)}{flag}` (or `--mode probe` for fast transcript + keyframes, "
                     f"`--with/--without pic,ocr,sounds,harmony,speech,voices,emotion`) and read the card first{cached}.")
    return "\n".join(lines)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return                                  # not a hook payload: say nothing
    prompt = data.get("prompt", "")
    items = candidates(prompt, data.get("cwd", "."))
    if not items:
        return
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                             "additionalContext": context(items, prompt)}}))
