"""Agent-facing outputs: a markdown storyboard and labelled contact sheets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W = 384
COLS = 3
PER_SHEET = 12


def _font(size: int = 18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def grid(tiles: list[tuple[Path, str]], dest: Path, cols: int = COLS, tile_w: int = TILE_W) -> Path:
    """Tile images into one labelled grid, so an agent reads many frames in
    one image read."""
    font = _font()
    imgs = []
    for f, _ in tiles:
        im = Image.open(f).convert("RGB")
        imgs.append(im.resize((tile_w, max(1, round(im.height * tile_w / im.width)))))
    th = max(i.height for i in imgs)
    cols = min(cols, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * th), "black")
    draw = ImageDraw.Draw(sheet)
    for i, ((_, label), im) in enumerate(zip(tiles, imgs)):
        x, y = (i % cols) * tile_w, (i // cols) * th
        sheet.paste(im, (x, y))
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((x, y, x + box[2] + 10, y + box[3] + 8), fill="black")
        draw.text((x + 5, y + 3), label, fill="yellow", font=font)
    sheet.save(dest, quality=85)
    return dest


def contact_sheets(out: Path, manifest: dict) -> list[str]:
    """Grid many keyframes into one image so an agent looks at 12 frames in one
    read instead of 12 reads. Each tile is labelled with segment and time."""
    tiles = [(out / kf["file"], f"S{seg['id']:02d}  {kf['t']:.1f}s")
             for seg in manifest["segments"] for kf in seg["keyframes"] if "file" in kf]
    names = []
    for n in range(0, len(tiles), PER_SHEET):
        name = f"sheet_{n // PER_SHEET + 1:02d}.jpg"
        grid(tiles[n:n + PER_SHEET], out / name)
        names.append(name)
    return names


def storyboard(m: dict) -> str:
    s = m["stats"]
    lines = [
        f"# Storyboard: {Path(m['source']).name}",
        "",
        f"- duration {m['duration']}s · {s['segments']} segments · {s['keyframe_images']} keyframe images "
        f"(from ~{s['source_frames_approx']} source frames)"
        + (f" · content: {m['content']['type']}" if m.get("content", {}).get("type") else ""),
        f"- evidence: {s['hard_cuts']} hard cuts, {s['codec_scenecuts']} codec scene-cuts, "
        f"{s['pauses']} pauses, {s['transcript_cues']} transcript cues",
    ]
    if m.get("sheets"):
        lines.append(f"- contact sheets (12 labelled frames each): {', '.join(m['sheets'])}")
    lines += ["", "Reasons: visual_cut = hard cut · visual_change = gradual/visual drift · audio_change = new "
              "sound/speaker/music · at_pause = moved into silence · codec_scenecut = encoder put an unscheduled "
              "I-frame here · max_duration = forced split", ""]
    for seg in m["segments"]:
        ev = f"visual {seg['visual_change']} · audio {seg['audio_change']}"
        if seg.get("speech_ratio") is not None:
            ev += f" · sound {int(seg['speech_ratio'] * 100)}%"
        lines.append(f"## S{seg['id']:02d}  {seg['start_ts']} → {seg['end_ts']}  ({', '.join(seg['reasons'])})")
        lines.append(f"_{ev}_")
        for kf in seg["keyframes"]:
            if "file" in kf:
                lines.append(f"- {kf['t']:.1f}s `{kf['file']}` ({kf['why']})")
            else:
                lines.append(f"- {kf['t']:.1f}s same shot as `{kf['same_as']}`")
        if seg["transcript"]:
            lines.append(f"> {seg['transcript']}")
        lines.append("")
    return "\n".join(lines)
