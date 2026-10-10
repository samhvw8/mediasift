"""Synthetic video with known ground truth.

  0-4   red         440 Hz tone
  4-8   blue        440 Hz tone        <- hard visual cut at 4, audio unchanged
  8-12  blue        pause, then noise  <- audio-only change at ~8
  12-16 test pattern (moving)  noise   <- hard cut at 12
  16-20 red         440 Hz tone        <- cut at 16, same shot as 0-4 (duplicate)
"""
import subprocess
import sys

V = "s=320x180:r=25"
A = "sample_rate=44100"


def make(out: str) -> None:
    inputs = [
        f"color=c=red:{V}:d=4", f"color=c=blue:{V}:d=8", f"testsrc2={V}:d=4", f"color=c=red:{V}:d=4",
        f"sine=f=440:{A}:d=8", f"anullsrc=r=44100:cl=mono:d=0.6", f"anoisesrc=color=pink:{A}:amplitude=0.3:d=7.4",
        f"sine=f=440:{A}:d=4",
    ]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for i in inputs:
        cmd += ["-f", "lavfi", "-i", i]
    fc = ("[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,format=yuv420p[v];"
          "[4:a]aformat=channel_layouts=mono[a0];[6:a]aformat=channel_layouts=mono[a2];"
          "[7:a]aformat=channel_layouts=mono[a3];[a0][5:a][a2][a3]concat=n=4:v=0:a=1[a]")
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-g", "50",
            "-c:a", "aac", "-shortest", out]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "fixture.mp4")
