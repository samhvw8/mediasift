"""How laggy does the machine feel, and what does it cost, while a command runs?

Runs COMMAND and, alongside it at normal priority, asks for a 5 ms sleep
200 times a second and records how late each wakeup is. Late wakeups are what
a user feels as stutter (a UI thread misses its frame). Prints wall time and
the wakeup lateness percentiles, with an idle baseline first.

Every second it also samples the command's process tree (CPU %, physical
footprint), the system's swap use, free-memory level and GPU busy %. With --csv
the samples are written there, and the command's stderr is kept, each line
stamped with seconds since the start. macOS only (ps, sysctl, ioreg).

    uv run python bench/lagprobe.py [--csv FILE] -- mediasift see VIDEO --force
"""
import ctypes
import re
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


LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def _footprint(pid: int) -> float:
    """Physical footprint in MB, the number Activity Monitor shows. Unlike ps RSS it counts
    GPU (Metal) buffers: an MLX process holding 1 GB showed RSS 31 MB, footprint 1040 MB."""
    buf = (ctypes.c_uint64 * 40)()
    if LIBPROC.proc_pid_rusage(pid, 2, buf):     # RUSAGE_INFO_V2
        return 0.0
    return buf[9] / 2**20                         # after the 16-byte uuid: 8th field, ri_phys_footprint


def _tree(root: int) -> tuple[float, float]:
    """CPU % (100 = one core) and footprint MB summed over ROOT and its descendants."""
    rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,%cpu="], capture_output=True, text=True).stdout
    kids: dict[int, list[int]] = {}
    pct: dict[int, float] = {}
    for line in rows.splitlines():
        p = line.split()
        if len(p) == 3:
            pid, ppid = int(p[0]), int(p[1])
            kids.setdefault(ppid, []).append(pid)
            pct[pid] = float(p[2])
    cpu = mb = 0.0
    todo = [root]
    while todo:
        pid = todo.pop()
        cpu, mb = cpu + pct.get(pid, 0.0), mb + _footprint(pid)
        todo += kids.get(pid, [])
    return cpu, mb


def _system() -> tuple[float, int, int]:
    """Swap used MB, kernel free-memory level (%), GPU busy %."""
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", swap)
    level = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout
    gpu = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                         capture_output=True, text=True).stdout
    g = re.search(r'"Device Utilization %"=(\d+)', gpu)
    return float(m.group(1)) if m else 0.0, int(level or 0), int(g.group(1)) if g else -1


def sample(stop: threading.Event, pid: int, t0: float, out: list) -> None:
    while not stop.is_set():
        out.append((time.perf_counter() - t0, *_tree(pid), *_system()))
        stop.wait(1.0)


def measure(cmd: list[str] | None, seconds: float = 5.0, err: list | None = None,
            res: list | None = None) -> tuple[float, np.ndarray]:
    lat: list[float] = []
    stop = threading.Event()
    th = threading.Thread(target=probe, args=(stop, lat), daemon=True)
    th.start()
    t0 = time.perf_counter()
    if cmd:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        sm = threading.Thread(target=sample, args=(stop, p.pid, t0, res if res is not None else []), daemon=True)
        sm.start()
        for line in p.stderr:
            if err is not None:
                err.append(f"{time.perf_counter() - t0:7.1f}  {line.rstrip()}")
        if p.wait():
            raise subprocess.CalledProcessError(p.returncode, cmd)
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


def show_res(res: list) -> None:
    if not res:
        return
    a = np.array(res)
    t, cpu, rss, swap, level, gpu = a.T
    busy = gpu[gpu >= 0]
    print(f"process    CPU mean {cpu.mean():.0f}% max {cpu.max():.0f}% (100 = one core), "
          f"CPU time ~{cpu.sum() / 100:.0f} s; memory max {rss.max():.0f} MB")
    print(f"system     swap {swap[0]:.0f} -> {swap[-1]:.0f} MB (max {swap.max():.0f}); "
          f"free-memory level min {level.min():.0f}%; GPU busy mean {busy.mean():.0f}% max {busy.max():.0f}%")


if __name__ == "__main__":
    i = sys.argv.index("--")
    opts, cmd = sys.argv[1:i], sys.argv[i + 1:]
    csv = opts[opts.index("--csv") + 1] if "--csv" in opts else None
    show("idle", *measure(None))
    err: list[str] = []
    res: list = []
    show("command", *measure(cmd, err=err, res=res))
    show_res(res)
    if csv:
        with open(csv, "w") as f:
            f.write("t_s,cpu_pct,footprint_mb,swap_mb,free_level_pct,gpu_pct\n")
            f.writelines(",".join(f"{v:.1f}" for v in r) + "\n" for r in res)
        with open(csv.rsplit(".", 1)[0] + ".stderr.txt", "w") as f:
            f.write("\n".join(err) + "\n")
    print("\n".join(err[-3:]))
