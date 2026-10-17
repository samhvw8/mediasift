"""ffmpeg/ffprobe plumbing. Everything that touches a subprocess lives here."""

from __future__ import annotations

import json
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np


_PTS = re.compile(r"pts_time:\s*(-?[0-9.]+)")


class FfmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise FfmpegError(f"mediasift needs {' and '.join(missing)} on PATH")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    return proc.stdout


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool


def _ratio(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(s)


def probe(path: str) -> MediaInfo:
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path,
    ]))
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if not duration:
        duration = max(float(s.get("duration") or 0.0) for s in streams) if streams else 0.0
    if duration <= 0:
        raise FfmpegError(f"could not determine duration of {path}")
    w, h = (int(video["width"]), int(video["height"])) if video else (0, 0)
    if video and _rotation(video) % 180 == 90:
        w, h = h, w          # ffmpeg auto-rotates on decode; report the size as shown
    return MediaInfo(
        path=path,
        duration=duration,
        width=w,
        height=h,
        fps=_ratio(video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else 0.0,
        has_video=video is not None,
        has_audio=audio is not None,
    )


def _rotation(video: dict) -> int:
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    try:
        return int(video.get("tags", {}).get("rotate", 0)) % 360
    except ValueError:
        return 0


def _stream_raw(cmd: list[str], frame_bytes: int) -> Iterator[bytes]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield buf
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg decode failed ({proc.returncode}): {err.strip()[-2000:]}")


def video_frames(path: str, fps: float, width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
    """Uniformly resampled tiny RGB frames. Frame i sits at t = i / fps."""
    vf = f"fps={fps},scale={width}:{height}:flags=area,format=rgb24"
    # Decode stays in software on purpose: for this tiny-output scan hardware
    # decode measured slower on both targets (VideoToolbox 4-20x on M1 Pro,
    # NVDEC ~1.45x on RTX 3060 vs a 2016 Xeon). Skipping unreferenced
    # B-frames helped on M1 Pro (1.4x) but cost 20% on the Xeon, so it is
    # Apple-Silicon only. A sample moves by at most one source frame.
    skip = ["-skip_frame", "noref"] if sys.platform == "darwin" and platform.machine() == "arm64" else []
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *skip, "-i", path, "-an", "-sn",
           "-vf", vf, "-f", "rawvideo", "-"]
    for i, buf in enumerate(_stream_raw(cmd, width * height * 3)):
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)


def keyframe_frames(path: str, width: int, height: int) -> Iterator[tuple[float, np.ndarray]]:
    """Decode only the codec's intra frames (I-frames), each with the pts the
    decoder reports (showinfo on stderr). Packet flags and decoded frames do
    not always line up one to one, so timestamps must come from the decoder."""
    vf = f"showinfo,scale={width}:{height}:flags=area,format=rgb24"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin", "-skip_frame", "nokey", "-i", path,
           "-an", "-sn", "-fps_mode", "passthrough", "-vf", vf, "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    times: "queue.Queue[float | None]" = queue.Queue()
    tail: list[str] = []

    def read_stderr() -> None:
        for raw in proc.stderr:
            line = raw.decode(errors="replace")
            m = _PTS.search(line) if "showinfo" in line else None
            if m:
                times.put(float(m.group(1)))
            else:
                tail.append(line)
                del tail[:-50]
        times.put(None)

    th = threading.Thread(target=read_stderr, daemon=True)
    th.start()
    n = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            t = times.get(timeout=60)
            if t is None:
                raise FfmpegError("ffmpeg produced a frame without a timestamp")
            yield t, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()
        th.join(timeout=5)
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg keyframe decode failed ({proc.returncode}): {''.join(tail)[-2000:]}")


def audio_pcm(path: str, sr: int) -> np.ndarray:
    """Whole audio track as mono int16. Kept as int16 (half the memory of
    float32: 200 MB for a 1h45m podcast); consumers convert block by block."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-sn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg audio decode failed: {proc.stderr.decode(errors='replace')[-2000:]}")
    return np.frombuffer(proc.stdout, np.int16)


@dataclass
class Packets:
    """Compressed-domain view of the video stream: no decoding needed.

    The encoder already did motion estimation; how many bits it spent on each
    predicted (P/B) frame is a free measure of how much changed.
    """
    times: np.ndarray    # pts of every video packet, sorted
    sizes: np.ndarray    # bytes
    is_key: np.ndarray   # bool: intra (I/IDR) packet


def packets(path: str) -> Packets:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "packet=pts_time,size,flags", "-of", "csv=p=0", path])
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3 or parts[0] in ("", "N/A"):
            continue
        rows.append((float(parts[0]), int(parts[1]), "K" in parts[2]))
    if not rows:
        return Packets(np.zeros(0), np.zeros(0), np.zeros(0, bool))
    rows.sort()
    arr = np.array(rows, dtype=object)
    return Packets(arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(bool))


def frames_at(path: str, times: list[float], width: int) -> Iterator[tuple[float, np.ndarray]]:
    """Many exact moments in one decode pass (one ffmpeg, not one per frame): each requested
    time gets the first frame at or after it. Yields (requested time, RGB) in time order."""
    want = sorted(set(round(t, 3) for t in times))
    if not want:
        return
    info = probe(path)
    w = min(width, info.width) // 2 * 2
    h = round(info.height * w / info.width / 2) * 2
    # Seek near the first moment; keep only frames in a one-frame window after each wanted time
    # (scaling every decoded frame was the cost); showinfo reports each kept frame's pts.
    # `fps=30` first: every tick then holds the frame on screen at that time, also in variable
    # frame rate files (screen recordings), where "the first frame after t" can be the next state.
    start = max(0.0, want[0] - 1.0)
    win = 1.02 / 30
    # Seek without discarding (-noaccurate_seek) and keep absolute timestamps (-copyts): an accurate
    # seek drops the frame that is still on screen when it began before the seek point.
    sel = "+".join(f"between(t,{t - 0.002:.3f},{t + win:.3f})" for t in want)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin", "-noaccurate_seek", "-ss", f"{start:.3f}",
           "-copyts", "-i", path,
           "-an", "-sn", "-vf", f"fps=30,select='{sel}',showinfo,scale={w}:{h}:flags=bicubic,format=rgb24",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    pts: "queue.Queue[float | None]" = queue.Queue()
    tail: list[str] = []

    def read_stderr() -> None:
        for raw in proc.stderr:
            line = raw.decode(errors="replace")
            m = _PTS.search(line) if "showinfo" in line else None
            if m:
                pts.put(float(m.group(1)))
            else:
                tail.append(line)
                del tail[:-50]
        pts.put(None)

    th = threading.Thread(target=read_stderr, daemon=True)
    th.start()
    n, k = w * h * 3, 0
    try:
        while k < len(want):
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            t = pts.get(timeout=60)
            if t is None:
                break
            while k < len(want) and t >= want[k] - 0.002:
                yield want[k], np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                k += 1
    finally:
        proc.kill()
        proc.stdout.close()
        proc.wait()
        th.join(timeout=5)
    if k < len(want):
        # Past the last frame (a time at the very end): fall back to a seek each.
        for t in want[k:]:
            yield t, None


def extract_frame(path: str, t: float, out: str, width: int) -> None:
    _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
          "-vf", f"scale='min({width},iw)':-2", "-q:v", "3", out])


def extract_clip(path: str, start: float, end: float, out: str, reencode: bool) -> None:
    codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"] if reencode else ["-c", "copy"]
    _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", path,
          "-t", f"{end - start:.3f}", *codec, out])
