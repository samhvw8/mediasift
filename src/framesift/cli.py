from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

from . import decode
from .pipeline import Params, analyze, fmt_ts, log


def _see(a: argparse.Namespace) -> None:
    from .see import see
    if a.now:
        os.environ["FRAMESIFT_FULL_SPEED"] = "1"
    if a.on:
        os.environ["FRAMESIFT_ON"] = a.on
    out = see(a.video, force=a.force, asr_mode=a.asr)
    sys.stdout.write((out / "card.md").read_text())
    print(f"\n(card and images in {out})")


def _look_span(a: argparse.Namespace) -> None:
    from .see import look
    print(look(a.video, a.span, a.n))


def _hook(a: argparse.Namespace) -> None:
    from .hook import main as hook_main
    hook_main()


def _asr(a: argparse.Namespace) -> None:
    from .asr import serve_stdin
    if a.audio != "-":
        sys.exit("only `framesift asr -` (audio on stdin) is supported")
    serve_stdin(a.lang)


def _analyze(a: argparse.Namespace) -> None:
    p = Params(fps=a.fps, fast=a.fast, min_seg=a.min_seg, max_seg=a.max_seg, sensitivity=a.sensitivity,
               cut_threshold=a.cut_threshold, drift=a.drift, content=a.content, max_frames=a.max_frames, width=a.width,
               sheets=not a.no_sheets, transcript=a.transcript, asr=a.asr, asr_backend=a.asr_backend, asr_model=a.asr_model, lang=a.lang)
    out = a.out or str(Path(a.video).with_suffix("")) + ".framesift"
    r = analyze(a.video, out, p)
    s = r.manifest["stats"]
    t = r.timings
    log(f"{s['segments']} segments, {s['keyframe_images']} keyframes in {t['total']:.1f}s "
        f"(visual {t['visual']:.1f}s, audio {t['audio']:.1f}s)")
    if a.index:
        _index(argparse.Namespace(out=out, on=a.on))
    print(Path(out) / "storyboard.md")


def _index(a: argparse.Namespace) -> None:
    from .embed import Embedder
    from .index import build
    with Embedder(vision=True, host=a.on) as emb:
        log(f"wrote {build(a.out, emb)}")


def _find(a: argparse.Namespace) -> None:
    from .embed import Embedder
    from .query import find, load, parse_span, render_find
    if (a.query is None) == (a.like is None):
        sys.exit("give a text query or --like SECONDS")
    ix = load(a.out)
    if a.query is None:  # --like compares stored frame vectors; no model needed
        res = find(ix, None, None, a.like, a.k, parse_span(a.between))
    else:
        with Embedder(vision=False) as emb:
            res = find(ix, emb, a.query, None, a.k, parse_span(a.between))
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else render_find(res))


def _look(a: argparse.Namespace) -> None:
    if a.span or not a.at:
        if not a.span:
            sys.exit("give a span: framesift look VIDEO START-END")
        a.video = a.out
        return _look_span(a)
    from .query import load, look, render_look
    times = [float(x) for x in a.at.split(",") if x.strip()]
    res = look(load(a.out), times, a.width)
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else render_look(res))


def _frames(a: argparse.Namespace) -> None:
    """Dense look at a short range, for when the keyframes are not enough."""
    decode.require_ffmpeg()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    span = a.end - a.start
    if span <= 0:
        sys.exit("--end must be after --start")
    n = a.n if a.n else max(1, int(span * a.fps))
    ts = [a.start + span * (i + 0.5) / n for i in range(n)]
    for t in ts:
        f = out / f"t{t:08.2f}s.jpg"
        decode.extract_frame(a.video, t, str(f), a.width)
        print(f)


def _split(a: argparse.Namespace) -> None:
    decode.require_ffmpeg()
    m = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(m["source"]).suffix or ".mp4"
    ids = set(a.ids) if a.ids else None
    for seg in m["segments"]:
        if ids and seg["id"] not in ids:
            continue
        f = out / f"S{seg['id']:02d}_{fmt_ts(seg['start']).replace(':', '-')}{ext if not a.reencode else '.mp4'}"
        decode.extract_clip(m["source"], seg["start"], seg["end"], str(f), a.reencode)
        print(f)


def _meaning(a: argparse.Namespace) -> None:
    from .meaning import build
    build(a.out, a.vlm_url, a.vlm_model, lang=a.lang, workers=a.workers)
    print(Path(a.out) / "meaning.md")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="framesift", description="Change-driven video segmentation for agents.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    se = sub.add_parser("see", help="sense a video or URL: a timeline card of picture, sound and speech (cached)")
    se.add_argument("video", help="a video/audio file or a web URL (yt-dlp)")
    se.add_argument("--asr", choices=["auto", "on", "off"], default="auto",
                    help="transcribe speech: auto = when the sound tagger hears speech (default)")
    se.add_argument("--force", action="store_true", help="recompute even if cached")
    se.add_argument("--now", action="store_true", help="skip the polite settings (nice 10, capped worker pools)")
    se.add_argument("--on", metavar="HOST", help="transcribe on this ssh host (a GPU box) when it answers within 3 s; "
                    "default: $FRAMESIFT_ON")
    se.set_defaults(fn=_see)

    hk = sub.add_parser("hook", help="Claude Code UserPromptSubmit hook: points the agent at videos named in a prompt")
    hk.set_defaults(fn=_hook)

    sa = sub.add_parser("asr", help="speech to timed cues: `framesift asr -` reads audio on stdin, prints JSON "
                                    "(what `see --on HOST` runs on the host)")
    sa.add_argument("audio", help="- for stdin")
    sa.add_argument("--lang", help="ISO 639-1 (default: detect)")
    sa.set_defaults(fn=_asr)

    an = sub.add_parser("analyze", help="segment a video and pick keyframes")
    an.add_argument("video")
    an.add_argument("-o", "--out", help="output dir (default: <video>.framesift/)")
    an.add_argument("--fps", type=float, default=5.0, help="analysis rate of tiny frames (default 5)")
    an.add_argument("--fast", action="store_true", help="decode codec I-frames only; much faster on long videos")
    an.add_argument("--min-seg", type=float, default=2.0, help="shortest segment, seconds (default 2)")
    an.add_argument("--max-seg", type=float, default=90.0, help="longest segment, seconds (default 90)")
    an.add_argument("--sensitivity", type=float, default=4.0, help="soft-boundary threshold; lower = more segments")
    an.add_argument("--cut-threshold", type=float, default=0.12, help="hard-cut frame distance floor (default 0.12)")
    an.add_argument("--drift", type=float, help="change that earns a new keyframe; lower = more frames "
                    "(default 0.06 screen, 0.12 camera)")
    an.add_argument("--content", choices=["auto", "screen", "camera"], default="auto",
                    help="content type for drift defaults (default: auto-detect)")
    an.add_argument("--max-frames", type=int, default=80, help="budget of keyframe images, 0 = unlimited")
    an.add_argument("--width", type=int, default=768, help="keyframe image width (default 768)")
    an.add_argument("--no-sheets", action="store_true", help="skip contact sheets")
    an.add_argument("--transcript", help="existing .srt/.vtt/.json transcript to attach")
    an.add_argument("--asr", action="store_true", help="transcribe locally (needs the [asr] extra; add [mlx] on Apple Silicon)")
    an.add_argument("--asr-backend", default="auto", choices=["auto", "parakeet-mlx", "mlx-whisper", "faster-whisper"],
                    help="auto: Parakeet (MLX) for European languages on Apple Silicon, mlx-whisper for others, "
                         "faster-whisper on CUDA/CPU")
    an.add_argument("--asr-model", help="model id or path (default depends on backend)")
    an.add_argument("--lang", help="spoken language, ISO 639-1 (default: detect)")
    an.add_argument("--index", action="store_true", help="also run `index` (needs llama-server)")
    an.add_argument("--on", metavar="HOST", help="with --index: embed on this ssh host (its llama-server)")
    an.set_defaults(fn=_analyze)

    ix = sub.add_parser("index", help="embed frames and speech so `find` can search them (needs llama-server)")
    ix.add_argument("out", help="an analyze output dir")
    ix.add_argument("--on", metavar="HOST", help="embed on this ssh host (its llama-server, e.g. a GPU box)")
    ix.set_defaults(fn=_index)

    fd = sub.add_parser("find", help="frames that match a text query, as one labelled grid")
    fd.add_argument("out", help="an indexed output dir")
    fd.add_argument("query", nargs="?", help="what you want to see, e.g. 'error dialog' or 'whiteboard diagram'")
    fd.add_argument("--like", type=float, metavar="SECONDS", help="instead of text: frames that look like this moment")
    fd.add_argument("-k", type=int, default=6, help="number of distinct moments (default 6)")
    fd.add_argument("--between", metavar="A-B", help="only search this range, seconds, e.g. 60-180")
    fd.add_argument("--json", action="store_true")
    fd.set_defaults(fn=_find)

    lk = sub.add_parser("look", help="see a span: `look VIDEO 12-15` (frames + sound strip), "
                                     "or exact moments of an indexed dir: `look OUT --at 12.5,40`")
    lk.add_argument("out", help="a video file/URL (with SPAN), or an analyze output dir (with --at)")
    lk.add_argument("span", nargs="?", help="START-END in seconds or m:ss, e.g. 11.5-13 or 1:02-1:10")
    lk.add_argument("-n", type=int, default=8, help="frames across the span (default 8)")
    lk.add_argument("--at", help="comma-separated seconds, e.g. 12.5,40,41.5")
    lk.add_argument("--width", type=int, default=768)
    lk.add_argument("--json", action="store_true")
    lk.set_defaults(fn=_look)

    fr = sub.add_parser("frames", help="extract frames from a time range")
    fr.add_argument("video")
    fr.add_argument("--start", type=float, required=True)
    fr.add_argument("--end", type=float, required=True)
    g = fr.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, help="number of evenly spaced frames")
    g.add_argument("--fps", type=float, default=2.0, help="frames per second (default 2)")
    fr.add_argument("--width", type=int, default=768)
    fr.add_argument("-o", "--out", required=True)
    fr.set_defaults(fn=_frames)

    sp = sub.add_parser("split", help="cut the source into one clip per segment")
    sp.add_argument("manifest")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--ids", type=int, nargs="*", help="only these segment ids")
    sp.add_argument("--reencode", action="store_true", help="frame-accurate cuts (default: stream copy, cuts snap to I-frames)")
    sp.set_defaults(fn=_split)

    me = sub.add_parser("meaning", help="describe keyframes, find topic chapters and summarize (needs a VLM server)")
    me.add_argument("out", help="an analyze output dir (has manifest.json)")
    me.add_argument("--vlm-url", default=os.environ.get("FRAMESIFT_VLM_URL", "http://localhost:8011/v1"),
                    help="OpenAI-compatible endpoint (vLLM, llama.cpp, LM Studio, MLX-VLM)")
    me.add_argument("--vlm-model", default=os.environ.get("FRAMESIFT_VLM_MODEL", "qwen3.5-4b"))
    me.add_argument("--lang", default="English", help="language for descriptions and summaries")
    me.add_argument("--workers", type=int, default=8, help="concurrent requests (the server batches them)")
    me.set_defaults(fn=_meaning)

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in sub.choices and not argv[0].startswith("-"):
        argv = ["see", *argv]           # `framesift VIDEO` is `framesift see VIDEO`
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except decode.FfmpegError as e:
        sys.exit(f"framesift: {e}")


if __name__ == "__main__":
    main()
