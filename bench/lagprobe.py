"""How laggy does the machine feel while a command runs?

Runs COMMAND and, alongside it at normal priority, asks for a 5 ms sleep
200 times a second and records how late each wakeup is. Late wakeups are what
a user feels as stutter (a UI thread misses its frame). Prints wall time and
the wakeup lateness percentiles, with an idle baseline first.

    uv run python bench/lagprobe.py -- framesift see VIDEO --force
"""
import subprocess
import sys
import threading
import time

import numpy as np


def probe(stop: threading.Event, out: list) -> None:
    while not stop.is_set():
        t = time.perf_counter()
        time.sleep(0.005)
        out.append((time.perf_counter() - t - 0.005) * 1000)


def measure(cmd: list[str] | None, seconds: float = 5.0) -> tuple[float, np.ndarray]:
    lat: list[float] = []
    stop = threading.Event()
    th = threading.Thread(target=probe, args=(stop, lat), daemon=True)
    th.start()
    t0 = time.perf_counter()
    if cmd:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        time.sleep(seconds)
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    return wall, np.array(lat)


def show(name: str, wall: float, lat: np.ndarray) -> None:
    p = np.percentile(lat, [50, 99, 99.9])
    print(f"{name:10} wall {wall:6.1f}s  wakeup late p50 {p[0]:5.2f} ms  p99 {p[1]:6.2f} ms  "
          f"p99.9 {p[2]:6.2f} ms  >16ms {100 * (lat > 16).mean():.2f}%")


if __name__ == "__main__":
    cmd = sys.argv[sys.argv.index("--") + 1:]
    show("idle", *measure(None))
    show("command", *measure(cmd))
