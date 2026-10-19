"""Codec-aligned motion & intra-bucket peak benchmark on Hugging Face and Kaggle real video clips.

Evaluates three capabilities introduced from the OneVision-Encoder / CoPE-VideoLM / Mage-VL
codec-native vision literature:
  1. Camera pan/tilt vs foreground motion separation on Hugging Face `zai-org/MotionBench`
     (panning/tilting camera clips vs stationary-camera clips with moving foreground objects)
     plus motion-compensated P-frame residual suppression during camera pans.
  2. Stratified intra-bucket residual peak snapping (`events.select_bucket_peaks` and
     `events.snap_look_times`) vs uniform `(k + 0.5)` clock midpoints across 15 clips from
     Hugging Face (`zai-org/MotionBench`, `varunredd/reels-clips`, `mm-eval/TempCompass`,
     `eshan292/ig-reels`), Kaggle (`unidpro/video-emotion-recognition-dataset`), and
     the real-footage sense bench (`reel_en.mp4`, `tiktok_vi.mp4`).
  3. Compressed-domain packet activity (`decode.packet_activity`) speed and scene-cut preservation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mediasift import decode, events, see, sound


BENCH_CLIPS = [
    ("mb_pan_1.mp4", "zai-org/MotionBench", "f099db892172a015c489507c9abe56b036d960ef", "MotionBench/self-collected/Uw3ZbbULdjP4ccee.mp4", "camera_pan", "right"),
    ("mb_pan_2.mp4", "zai-org/MotionBench", "f099db892172a015c489507c9abe56b036d960ef", "MotionBench/self-collected/8ewZS1KwvFVf7paQ.mp4", "camera_pan", "up"),
    ("mb_pan_3.mp4", "zai-org/MotionBench", "f099db892172a015c489507c9abe56b036d960ef", "MotionBench/self-collected/y6MxNgRYCb4c9iKQ.mp4", "camera_pan", "left"),
    ("mb_still_1.mp4", "zai-org/MotionBench", "f099db892172a015c489507c9abe56b036d960ef", "MotionBench/self-collected/EQEJYaSbuzkmL1TJ.mp4", "camera_still", "still"),
    ("mb_still_2.mp4", "zai-org/MotionBench", "f099db892172a015c489507c9abe56b036d960ef", "MotionBench/self-collected/szyRsTjjV8yji7rV.mp4", "camera_still", "still"),
    ("mb_still_3.mp4", "zai-org/MotionBench", "f099db892172a015c489507c9abe56b036d960ef", "MotionBench/self-collected/73JLhkFliEGLKrCw.mp4", "camera_still", "still"),
    ("reel_clip_1.mp4", "varunredd/reels-clips", "af3c6a787d1a1e537238e48f57bfe2076157f21e", "data/clips_compressed/clip_1.mp4", "reel_short", ""),
    ("reel_clip_2.mp4", "varunredd/reels-clips", "af3c6a787d1a1e537238e48f57bfe2076157f21e", "data/clips_compressed/clip_2.mp4", "reel_short", ""),
    ("reel_clip_3.mp4", "varunredd/reels-clips", "af3c6a787d1a1e537238e48f57bfe2076157f21e", "data/clips_compressed/clip_3.mp4", "reel_short", ""),
    ("reel_clip_4.mp4", "varunredd/reels-clips", "af3c6a787d1a1e537238e48f57bfe2076157f21e", "data/clips_compressed/clip_4.mp4", "reel_short", "letterbox"),
    ("reel_clip_5.mp4", "varunredd/reels-clips", "af3c6a787d1a1e537238e48f57bfe2076157f21e", "data/clips_compressed/clip_5.mp4", "reel_short", ""),
    ("tc_dir_fwd.mp4", "mm-eval/TempCompass", "497fa38efcc9025bfce62491ad252b4be1ea714c", "videos/1012085909.mp4", "action_dir", ""),
    ("tc_dir_rev.mp4", "mm-eval/TempCompass", "497fa38efcc9025bfce62491ad252b4be1ea714c", "videos/1012085909_reverse.mp4", "action_dir", ""),
    ("tc_order_1.mp4", "mm-eval/TempCompass", "497fa38efcc9025bfce62491ad252b4be1ea714c", "videos/1014976360.mp4", "action_order", ""),
    ("ig_reel_1.mp4", "eshan292/ig-reels", "fce0c1fcca2a161c7ffbdcf328311358be144bd1", "reels/002d480d835a/reel.mp4", "ig_reel", ""),
    ("ig_reel_2.mp4", "eshan292/ig-reels", "fce0c1fcca2a161c7ffbdcf328311358be144bd1", "reels/003828af992/reel.mp4", "ig_reel", "static-slide"),
    ("ig_reel_3.mp4", "eshan292/ig-reels", "fce0c1fcca2a161c7ffbdcf328311358be144bd1", "reels/01950906047/reel.mp4", "ig_reel", "static-slide"),
    ("ig_reel_4.mp4", "eshan292/ig-reels", "fce0c1fcca2a161c7ffbdcf328311358be144bd1", "reels/027adc6d74d5/reel.mp4", "ig_reel", "static-slide"),
    ("tt_vi_1.mp4", "funa21/tiktok-data", "ad9225e9483d6aa22fb56e31bed03dd171edbfbd", "6895706751161650433/video.mp4", "tiktok_vi", "letterbox"),
    ("tt_vi_2.mp4", "funa21/tiktok-data", "ad9225e9483d6aa22fb56e31bed03dd171edbfbd", "7148068178705370410/video.mp4", "tiktok_vi", "letterbox"),
    ("tt_vi_3.mp4", "funa21/tiktok-data", "ad9225e9483d6aa22fb56e31bed03dd171edbfbd", "7152778345447542059/video.mp4", "tiktok_vi", "letterbox"),
    ("tt_gen_1.mp4", "Shofo/shofo-tiktok-general-small", "93625fe8042be012276b2d88fd27860c4ee68200", "videos/00/7268931097663245600.mp4", "tiktok_en", ""),
    ("tt_gen_2.mp4", "Shofo/shofo-tiktok-general-small", "93625fe8042be012276b2d88fd27860c4ee68200", "videos/00/7263995194335399200.mp4", "tiktok_en", "letterbox+split-vertical"),
    ("yt_salmonn_0.mp4", "videoSALMONN2/video-SALMONN_2_testset", "2cd0ac902f4d7271f64e91d1e3330337b7c6a521", "video/0.mp4", "youtube_short", ""),
    ("yt_salmonn_1.mp4", "videoSALMONN2/video-SALMONN_2_testset", "2cd0ac902f4d7271f64e91d1e3330337b7c6a521", "video/1.mp4", "youtube_short", ""),
    ("yt_salmonn_10.mp4", "videoSALMONN2/video-SALMONN_2_testset", "2cd0ac902f4d7271f64e91d1e3330337b7c6a521", "video/10.mp4", "youtube_short", ""),
    ("yt_vstat_kb1.mp4", "nyu-visionx/vstat", "21b3198fb46627acea9770216a36651a407394b8", "videos/self_recorded/keyboard/0001_pt1.mp4", "youtube_vstat", ""),
    ("fb_aria_inspect.mp4", "facebook/show3d-dataset", "b198fe6c2935db9c2e23a0cc6eeb365aab542c81", "hand_pose/v2/scenes/ASC023/aria_inspecting_f5e9/viz.mp4", "facebook_meta", "split-vertical"),
    ("fb_aria_move.mp4", "facebook/show3d-dataset", "b198fe6c2935db9c2e23a0cc6eeb365aab542c81", "hand_pose/v2/scenes/ASC023/aria_moving-around_2276/viz.mp4", "facebook_meta", "split-vertical"),
    ("fb_boxing.mp4", "facebook/show3d-dataset", "b198fe6c2935db9c2e23a0cc6eeb365aab542c81", "hand_pose/v2/scenes/AZH822/none_boxing_1dbf/viz.mp4", "facebook_meta", "split-vertical"),
]

PERSISTENT_TEXT_GROUND_TRUTH = [
    ("tt_vi_1.mp4", "quoc te dan ong"),
    ("tt_vi_2.mp4", "cuoi hai vo luon nha lindo"),
    ("tt_vi_3.mp4", "o angola lau toc quan"),
    ("tt_gen_2.mp4", "original"),
    ("yt_salmonn_0.mp4", "things you have to look at"),
    ("yt_salmonn_10.mp4", "coolest truck"),
    ("ig_reel_1.mp4", "simple beauty rule nobody mentions"),
    ("ig_reel_2.mp4", "ways to build stronger relationships"),
    ("ig_reel_3.mp4", "data visualization tools you need"),
    ("ig_reel_4.mp4", "chatgpt prompt engineering templates"),
]


def ensure_bench_clips() -> Path:
    bench_dir = sound.cache_root() / "codec_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    for local_name, repo, rev, rfile, _, _ in BENCH_CLIPS:
        dest = bench_dir / local_name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        from huggingface_hub import hf_hub_download
        import shutil
        src = hf_hub_download(repo_id=repo, filename=rfile, repo_type="dataset", revision=rev)
        shutil.copyfile(src, dest)
    return bench_dir


def run_benchmark() -> dict:
    bench_dir = ensure_bench_clips()
    pan_hits = 0
    pan_total = 0
    still_fp = 0
    still_total = 0
    residual_reductions: list[float] = []

    for local_name, _, _, _, kind, expected_dir in BENCH_CLIPS[:6]:
        p = str(bench_dir / local_name)
        frames = list(decode.video_frames(p, 15.0, events.DECODE_W, events.DECODE_H))
        g = events.build_grid(iter(frames))
        moves = events.camera_moves(g, screen=g.identical > 0.3)
        # Compare unshifted raw block change vs motion-compensated g.diff during camera pans
        bl = g.blocks.astype(np.float32)
        raw_diff = np.abs(bl[1:] - bl[:-1]).max(axis=3)
        raw_chg = float((raw_diff > g.thr).mean())
        mc_chg = float((g.diff[1:] > g.thr).mean())
        if kind == "camera_pan":
            pan_total += 1
            words = " ".join(m["words"] for m in moves)
            if expected_dir in words:
                pan_hits += 1
            if raw_chg > 1e-6:
                residual_reductions.append((raw_chg - mc_chg) / raw_chg)
        elif kind == "camera_still":
            still_total += 1
            if moves:
                still_fp += 1

    # Intra-bucket peak snapping vs uniform clock midpoints across benchmark + sample clips
    peak_gains: list[float] = []
    clip_paths = [bench_dir / c[0] for c in BENCH_CLIPS]
    for sname in ("reel_en.mp4", "tiktok_vi.mp4"):
        sp = ROOT / "samples" / sname
        if sp.exists():
            clip_paths.append(sp)

    for cp in clip_paths:
        if not cp.exists():
            continue
        g = events.build_grid(decode.video_frames(str(cp), 15.0, events.DECODE_W, events.DECODE_H))
        act = events.frame_activity(g)
        dur = float(g.times[-1]) if len(g.times) else 0.0
        n = max(4, min(24, int(dur)))
        if dur < 2.0 or len(act) < 4:
            continue
        unif_t = [(k + 0.5) * dur / n for k in range(n)]
        snap_t = events.select_bucket_peaks(g, 0.0, dur, n, anchor_ends=False)
        unif_scores = [float(act[int(np.argmin(np.abs(g.times - t)))]) for t in unif_t]
        snap_scores = [float(act[int(np.argmin(np.abs(g.times - t)))]) for t in snap_t]
        u_mean = max(float(np.mean(unif_scores)), 1e-4)
        s_mean = float(np.mean(snap_scores))
        peak_gains.append(s_mean / u_mean)

    # Check exact gesture apex capture on reel_en.mp4 (q06 fist at ~9.13-9.27s, q04 sweep at ~15.33-15.80s)
    reel_path = ROOT / "samples" / "reel_en.mp4"
    gesture_hits = 0
    gesture_total = 2
    if reel_path.exists():
        g_reel = events.build_grid(decode.video_frames(str(reel_path), 15.0, events.DECODE_W, events.DECODE_H))
        flow_t = events.select_bucket_peaks(g_reel, 0.0, float(g_reel.times[-1]), 26, anchor_ends=False)
        peaks_reel = events.extract_frame_peaks(g_reel)
        look_q06_t = events.snap_look_times(8.79, 10.99, 8, peaks_reel)
        look_q04_t = events.snap_look_times(14.95, 16.91, 8, peaks_reel)
        if any(9.10 <= t <= 9.35 for t in flow_t) and any(9.10 <= t <= 9.35 for t in look_q06_t):
            gesture_hits += 1
        if any(15.25 <= t <= 15.65 for t in look_q04_t) and any(15.75 <= t <= 16.20 for t in look_q04_t):
            gesture_hits += 1

    # Check compressed-domain packet_activity speed & keyframe discovery
    t0 = time.perf_counter()
    pa_Sample = decode.packet_activity(str(bench_dir / "mb_pan_1.mp4"))
    pa_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Layout detection accuracy across letterbox, split-vertical, and static-slide social clips
    layout_hits = 0
    layout_total = 0
    for local_name, _, _, _, _, expected_meta in BENCH_CLIPS[6:]:
        if not expected_meta:
            continue
        layout_total += 1
        g_lay = events.build_grid(decode.video_frames(str(bench_dir / local_name), 10.0, events.DECODE_W, events.DECODE_H))
        lay = events.detect_layout(g_lay)
        if lay["kind"] == expected_meta:
            layout_hits += 1

    # 5. Persistent headline / static overlay OCR recall (scan_persistent_text vs change-only _ocr_events)
    from mediasift import ocr
    import tempfile
    pt_hits = 0
    pt_total = len(PERSISTENT_TEXT_GROUND_TRUTH)
    if ocr.available():
        for fname, expected_phrase in PERSISTENT_TEXT_GROUND_TRUTH:
            fpath = bench_dir / fname
            info = decode.probe(str(fpath))
            g_pt = events.build_grid(decode.video_frames(str(fpath), 10.0, events.DECODE_W, events.DECODE_H))
            s_pt = see.Sensed(str(fpath), info.duration, info.width, info.height, info.fps, 10.0, "camera", events=events.detect(g_pt))
            with tempfile.TemporaryDirectory(prefix="pt_bench_") as wdir:
                lt = see.scan_persistent_text(str(fpath), info, s_pt, Path(wdir))
            joined = " ".join(see._norm_ocr_key(x["text"]) for x in lt)
            if expected_phrase in joined:
                pt_hits += 1

    res = {
        "total_clips": len(BENCH_CLIPS),
        "motionbench_pan_recall": f"{pan_hits}/{pan_total}",
        "motionbench_still_fp": f"{still_fp}/{still_total}",
        "mean_pan_residual_reduction": round(float(np.mean(residual_reductions)) if residual_reductions else 0.0, 3),
        "mean_intra_bucket_peak_gain": round(float(np.mean(peak_gains)) if peak_gains else 1.0, 3),
        "reel_gesture_apex_hits": f"{gesture_hits}/{gesture_total}",
        "social_layout_accuracy": f"{layout_hits}/{layout_total}",
        "persistent_text_recall": f"{pt_hits}/{pt_total}" if ocr.available() else "skipped",
        "packet_activity_ms": round(pa_ms, 1),
        "packet_activity_keys": len(pa_Sample["key_times"]),
    }
    print(
        f"codec_motion_bench ({len(BENCH_CLIPS)} clips): pan_recall={res['motionbench_pan_recall']} "
        f"still_fp={res['motionbench_still_fp']} "
        f"pan_residual_reduction={res['mean_pan_residual_reduction'] * 100:.1f}% "
        f"peak_gain={res['mean_intra_bucket_peak_gain']:.2f}x "
        f"gesture_apex={res['reel_gesture_apex_hits']} "
        f"layout={res['social_layout_accuracy']} "
        f"persistent_text={res['persistent_text_recall']} "
        f"packet_activity={res['packet_activity_ms']:.1f}ms"
    )
    assert pan_hits == pan_total, f"expected {pan_total}/{pan_total} pan recall, got {pan_hits}/{pan_total}"
    assert still_fp == 0, f"expected 0/{still_total} still false positives, got {still_fp}/{still_total}"
    assert gesture_hits == gesture_total, f"expected {gesture_total}/{gesture_total} gesture apex hits"
    assert layout_hits == layout_total, f"expected {layout_total}/{layout_total} layout accuracy, got {layout_hits}/{layout_total}"
    if ocr.available():
        assert pt_hits == pt_total, f"expected {pt_total}/{pt_total} persistent text recall, got {pt_hits}/{pt_total}"
    assert res["mean_intra_bucket_peak_gain"] >= 1.25, f"expected >=1.25x peak gain, got {res['mean_intra_bucket_peak_gain']}"
    return res


if __name__ == "__main__":
    run_benchmark()
