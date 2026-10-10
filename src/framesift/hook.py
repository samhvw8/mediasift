"""`framesift hook`: a Claude Code UserPromptSubmit hook that points the agent at a video.

When the prompt names an existing video/audio file or a video URL, it adds one line of context
("run `framesift see PATH`", and whether a card is already cached). It never senses anything
itself: hooks have a short timeout and run on every prompt. Prompts without a video cost nothing.

settings.json:
  {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "framesift hook"}]}]}}
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

MEDIA = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"}
# Only links that are videos by construction: x.com status links in past prompts were mostly text posts.
URL = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/|vimeo\.com/\d)\S+")
PATH = re.compile(r"(?:\"([^\"]+)\"|'([^']+)'|((?:~|/|\./|\.\./)?[\w./~\-]+?\.(?:" +
                  "|".join(e[1:] for e in MEDIA) + r")))(?=[\s,;:)\]?!.\'\"]|$)", re.I)


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


def context(items: list[str]) -> str:
    from .see import URL as IS_URL, out_dir
    lines = []
    for it in items:
        cached = ""
        if not IS_URL.match(it):
            card = out_dir(Path(it)) / "card.md"
            if card.exists():
                cached = f" (already sensed: card at {card})"
        lines.append(f"Video/audio in the prompt: {it}. You cannot watch or hear it; run "
                     f"`framesift see {shlex.quote(it)}` and read the card first{cached}.")
    return "\n".join(lines)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return                                  # not a hook payload: say nothing
    items = candidates(data.get("prompt", ""), data.get("cwd", "."))
    if not items:
        return
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                             "additionalContext": context(items)}}))
