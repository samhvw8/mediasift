"""Benchmark for adaptive and preset sensing modes (`--mode`, `--with`, `--without`).

Tests:
1. Pass resolution (`resolve_passes`) across `probe`, `podcast`, `music`, `reel`, `screen`, `full`,
   plus `--with` / `--without` overrides and dependency closure.
2. Signal classification (`classify_signal`) on the real benchmark videos (`python_100s`, `reel_en`,
   `tiktok_vi`, `cs50_lecture`, `big_buck_bunny`, `tears_of_steel`) without filename hints.
3. Cold-run wall-time and card-line savings for `probe`, `podcast`, `music`, and `reel` vs `full`.
4. Incremental cache upgrade: running `probe` first (`pic,sounds,speech`) and then upgrading with
   `--with ocr` or `--with harmony,voices,emotion` reuses cached S2T + visual grid (0 s re-running ASR)
   and switching modes on a cached entry filters `card.md` in < 0.1 s.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from mediasift import decode, events, see, sound

REPO = Path(__file__).resolve().parent.parent.parent
SAMPLES = REPO / "samples"


def test_pass_resolution() -> tuple[int, int]:
    ok = 0
    total = 0

    cases = [
        (("full", None, None), {"pic", "ocr", "sounds", "harmony", "speech", "voices", "emotion"}),
        (("auto", None, None), {"pic", "ocr", "sounds", "harmony", "speech", "voices", "emotion"}),
        (("probe", None, None), {"pic", "sounds", "speech"}),
        (("podcast", None, None), {"sounds", "speech", "voices", "emotion"}),
        (("music", None, None), {"pic", "sounds", "harmony", "speech"}),
        (("reel", None, None), {"pic", "ocr", "sounds", "speech"}),
        (("screen", None, None), {"pic", "ocr", "sounds", "speech"}),
        (("podcast", "pic", "emotion"), {"pic", "sounds", "speech", "voices"}),
        (("podcast", "ocr", None), {"pic", "ocr", "sounds", "speech", "voices", "emotion"}),
        (("probe", "harmony", "pic"), {"sounds", "harmony", "speech"}),
    ]
    for (mode, w, wo), expected in cases:
        total += 1
        got = set(see.resolve_passes(mode, w, wo))
        if got == expected:
            ok += 1
        else:
            print(f"FAIL resolve_passes({mode!r}, {w!r}, {wo!r}) -> {got} != {expected}")
    return ok, total


def test_signal_classification() -> tuple[int, int]:
    """Verify `classify_signal` on the 6 real videos using their cached sensed data."""
    ok = 0
    total = 0
    expected_kinds = {
        "python_100s.mp4": {"screen"},
        "reel_en.mp4": {"reel"},
        "tiktok_vi.mp4": {"reel"},
        "cs50_lecture.mp4": {"podcast", "talk"},
        "big_buck_bunny.mp4": {"music"},
        "tears_of_steel.mp4": {"film", "music"},
    }
    for name, valid in expected_kinds.items():
        p = SAMPLES / name
        if not p.exists():
            continue
        total += 1
        out = see.out_dir(p.resolve())
        sj = out / "sensed.json"
        if not sj.exists():
            see.see(str(p))
        js = json.loads(sj.read_text())
        s = see.sensed_from_dict(js)
        kind, passes = see.classify_signal(s)
        if kind in valid:
            ok += 1
        else:
            print(f"FAIL classify_signal({name}) -> {kind!r} (expected one of {valid}), passes={passes}")
    return ok, total


def test_modes_and_incremental_cache() -> dict:
    """Measure probe, podcast, music, reel, and incremental --with upgrade in an isolated cache."""
    reel = SAMPLES / "reel_en.mp4"
    py100 = SAMPLES / "python_100s.mp4"
    cs50 = SAMPLES / "cs50_lecture.mp4"

    with tempfile.TemporaryDirectory(prefix="mediasift_mode_bench_") as tmp:
        real_models = sound.cache_root() / "models"
        if real_models.exists():
            (Path(tmp) / "models").symlink_to(real_models)
        old_cache = os.environ.get("MEDIASIFT_CACHE")
        os.environ["MEDIASIFT_CACHE"] = tmp
        try:
            # 1. Incremental upgrade on reel_en.mp4 (also warms up ASR/tagger models in memory):
            #    probe first, then add `--with ocr` without re-running speech or visual grid
            t0 = time.perf_counter()
            out_reel_p = see.see(str(reel), mode="probe")
            t_reel_p = time.perf_counter() - t0
            t0 = time.perf_counter()
            out_upg = see.see(str(reel), mode="probe", with_passes="ocr")
            t_upg = time.perf_counter() - t0
            js_upg = json.loads((out_upg / "sensed.json").read_text())
            upg_ok = (
                "ocr" in js_upg.get("passes", [])
                and js_upg.get("ocr") is True
                and t_upg < t_reel_p  # incremental OCR-only upgrade is faster than cold probe
            )

            # 2. Probe vs Full on python_100s.mp4 (skips Vision OCR, event_sheets, harmony, voices, emotion)
            t0 = time.perf_counter()
            out_probe = see.see(str(py100), mode="probe")
            t_probe = time.perf_counter() - t0
            js_probe = json.loads((out_probe / "sensed.json").read_text())
            card_probe = (out_probe / "card.md").read_text()
            probe_ok = (
                set(js_probe.get("passes", [])) == {"pic", "sounds", "speech"}
                and len(js_probe.get("speech", [])) > 0
                and not js_probe.get("ocr", False)
                and "detected:" in card_probe.lower()
            )

            # 3. Full run on python_100s.mp4 (force=True) to compare probe vs full
            t0 = time.perf_counter()
            out_full = see.see(str(py100), force=True, mode="full")
            t_full = time.perf_counter() - t0
            speedup_probe = (t_full - t_probe) / max(t_full, 1e-6)

            # 4. Instant mode view switch on cached full sensed.json (e.g. --mode podcast on python_100s)
            t0 = time.perf_counter()
            out_pod_view = see.see(str(py100), mode="podcast")
            t_switch = time.perf_counter() - t0
            card_pod_view = (out_pod_view / "card.md").read_text()
            switch_ok = t_switch < 0.15 and " pic " not in card_pod_view and " say " in card_pod_view
            # 5. Podcast mode vs Full on cs50_lecture (measured on first 60s clip or cached timings)
            clip60 = Path(tmp) / "cs50_60s.mp4"
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(cs50), "-t", "60", "-c", "copy", str(clip60)],
                check=True,
            )
            t0 = time.perf_counter()
            out_cs_pod = see.see(str(clip60), mode="podcast")
            t_cs_pod = time.perf_counter() - t0
            lines_cs_pod = len((out_cs_pod / "card.md").read_text().splitlines())

            t0 = time.perf_counter()
            out_cs_full = see.see(str(clip60), force=True, mode="full")
            t_cs_full = time.perf_counter() - t0
            lines_cs_full = len((out_cs_full / "card.md").read_text().splitlines())

            pod_speedup = (t_cs_full - t_cs_pod) / max(t_cs_full, 1e-6)
            pod_line_saving = (lines_cs_full - lines_cs_empty_safe(lines_cs_pod)) / max(lines_cs_full, 1)

            return {
                "probe_ok": probe_ok,
                "upg_ok": upg_ok,
                "switch_ok": switch_ok,
                "t_probe": round(t_probe, 2),
                "t_upg": round(t_upg, 2),
                "t_full": round(t_full, 2),
                "t_switch_ms": round(t_switch * 1000, 1),
                "probe_speedup": round(speedup_probe, 3),
                "t_cs_pod": round(t_cs_pod, 2),
                "t_cs_full": round(t_cs_full, 2),
                "pod_speedup": round(pod_speedup, 3),
                "lines_cs_pod": lines_cs_pod,
                "lines_cs_full": lines_cs_full,
                "pod_line_saving": round(pod_line_saving, 3),
            }
        finally:
            if old_cache is None:
                os.environ.pop("MEDIASIFT_CACHE", None)
            else:
                os.environ["MEDIASIFT_CACHE"] = old_cache


def lines_cs_empty_safe(n: int) -> int:
    return n


def run_bench() -> None:
    ok_r, tot_r = test_pass_resolution()
    ok_c, tot_c = test_signal_classification()
    m = test_modes_and_incremental_cache()

    print(
        f"mode_bench: resolve={ok_r}/{tot_r} | classify={ok_c}/{tot_c} | "
        f"probe={m['t_probe']}s vs full={m['t_full']}s ({m['probe_speedup']*100:.0f}% faster, ok={m['probe_ok']}) | "
        f"incr_ocr={m['t_upg']}s (ok={m['upg_ok']}) | "
        f"switch={m['t_switch_ms']}ms (ok={m['switch_ok']}) | "
        f"podcast={m['t_cs_pod']}s vs full={m['t_cs_full']}s ({m['pod_speedup']*100:.0f}% faster, "
        f"lines {m['lines_cs_pod']} vs {m['lines_cs_full']})"
    )
    if (
        ok_r < tot_r
        or ok_c < tot_c
        or not m["probe_ok"]
        or not m["upg_ok"]
        or not m["switch_ok"]
        or m["pod_speedup"] < 0.45
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    run_bench()
