"""Meaning layer: turn frames + speech into what an agent actually needs.

Runs on an existing analysis (manifest.json + frames/), so it never re-decodes:

1. describe  every unique keyframe with a vision-language model, given the
             speech around it: what is shown, text on screen, people.
2. chapters  topic segmentation of the transcript (TextTiling on sentence
             embeddings), snapped to framesift segment boundaries.
3. summarize each chapter (title, summary, key points, entities) and the
             whole video, from its speech plus its keyframe descriptions.

The model is reached over an OpenAI-compatible HTTP API (vLLM, llama.cpp
server, LM Studio, MLX-VLM), so the same code runs against a local GPU box
or a Mac.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"framesift: {msg}", file=sys.stderr)


@dataclass
class Client:
    url: str       # e.g. http://localhost:8011/v1
    model: str
    timeout: float = 180.0

    def chat(self, content: list[dict] | str, max_tokens: int = 600) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            # Qwen3.x: answer directly, no thinking trace.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = json.loads(r.read())["choices"][0]["message"]["content"]
                return _json(text)
            except Exception as e:  # network, 5xx, or unparseable output
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"model call failed after 3 attempts: {last}")


def _json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _image_part(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


DESCRIBE = """You are indexing a video for someone who cannot watch it.
Speech around this moment: "{speech}"
Describe this frame as JSON with keys:
  "shows": one or two sentences on what is visible and what it communicates (people, setting, diagrams, slides, code, B-roll), not visual style;
  "text": all readable on-screen text, verbatim, "" if none;
  "people": short labels for people visible (e.g. "host", "guest in suit"), [] if none.
Answer in {lang}. JSON only."""

SUMMARIZE = """You are indexing part of a video ({start}-{end}) for someone who cannot watch it.
Transcript of this part:
\"\"\"{speech}\"\"\"
What is on screen during this part (time: description | on-screen text):
{visuals}
Return JSON with keys:
  "title": a specific title for this part, max 10 words;
  "summary": 2-4 sentences with the concrete content (claims, numbers, names, steps), not "they discuss";
  "key_points": up to 5 short factual points;
  "entities": people, organisations, products, terms named.
Answer in {lang}. JSON only."""

OVERVIEW = """These are the chapters of one video, in order:
{chapters}
Return JSON with keys:
  "title": a specific title for the whole video;
  "summary": 3-5 sentences on what the video is about and its main conclusions;
  "topics": up to 8 topics.
Answer in {lang}. JSON only."""


def fmt(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{int(s):02d}" if h else f"{int(m):02d}:{int(s):02d}"


def speech_between(cues: list, a: float, b: float, limit: int) -> str:
    text = " ".join(c[2] for c in cues if c[1] > a and c[0] < b)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def describe(out: Path, m: dict, client: Client, lang: str, workers: int) -> dict[str, dict]:
    cues = m.get("transcript", [])
    jobs = [(kf["file"], kf["t"]) for s in m["segments"] for kf in s["keyframes"] if "file" in kf]

    def one(job):
        f, t = job
        prompt = DESCRIBE.format(speech=speech_between(cues, t - 8, t + 8, 500) or "(no speech)", lang=lang)
        try:
            r = client.chat([_image_part(out / f), {"type": "text", "text": prompt}], max_tokens=400)
        except Exception as e:
            log(f"describe {f} failed: {e}")
            return f, {"shows": "", "text": "", "people": [], "error": str(e)}
        return f, {"shows": str(r.get("shows", "")).strip(), "text": str(r.get("text", "")).strip(),
                   "people": list(r.get("people", []) or [])}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:  # the server batches concurrent requests
        res = dict(ex.map(one, jobs))
    log(f"described {len(jobs)} keyframes in {time.perf_counter() - t0:.1f}s")
    return res


def vision_selftest(client: Client) -> None:
    """Refuse to run on a model that cannot see. A broken vision path (e.g.
    FP8-quantizing the vision tower) still returns fluent JSON describing
    "blank grey screens"; one known image catches that before it poisons
    every description."""
    import io
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((460, 110, 600, 250), fill=(220, 20, 20))
    try:
        font = ImageFont.load_default(size=110)
    except TypeError:
        font = ImageFont.load_default()
    d.text((40, 110), "427", fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}
    r = client.chat([part, {"type": "text", "text": 'JSON with keys "number" (the number written) and "shape_color" (colour of the square).'}],
                    max_tokens=60)
    ok = "427" in json.dumps(r) and "red" in json.dumps(r).lower()
    if not ok:
        raise SystemExit(f"vision self-test failed: the model at {client.url} cannot read a test image (got {r}). "
                         "Check the vision tower is not quantized and images reach the model.")
    log("vision self-test passed")


def embedder(model: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise SystemExit("chapters need fastembed: `uv tool install 'framesift[meaning]'`") from e
    return TextEmbedding(model)


def chapters(m: dict, emb, min_len: float = 90.0, max_len: float = 600.0, window: float = 60.0) -> list[tuple[float, float]]:
    """TextTiling on sentence embeddings: slide over the transcript comparing
    the previous `window` seconds with the next; topic shifts are similarity
    valleys. Boundaries snap to framesift segment starts so chapters never cut
    a shot or a sentence in half."""
    cues = m.get("transcript", [])
    dur = m["duration"]
    seg_starts = [s["start"] for s in m["segments"][1:]]
    if len(cues) < 8 or dur < 2 * min_len:
        return [(0.0, dur)]
    vecs = np.array(list(emb.embed([c[2] for c in cues])), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    mids = np.array([(c[0] + c[1]) / 2 for c in cues])
    cand = [t for t in seg_starts if min_len / 2 <= t <= dur - min_len / 2]
    if not cand:
        return [(0.0, dur)]
    score = []
    for t in cand:
        left = vecs[(mids >= t - window) & (mids < t)]
        right = vecs[(mids >= t) & (mids < t + window)]
        if len(left) == 0 or len(right) == 0:
            score.append(1.0)
            continue
        a, b = left.mean(0), right.mean(0)
        score.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    score = np.array(score)
    # Depth of each valley relative to the highest similarity on either side.
    depth = np.zeros(len(score))
    for i in range(len(score)):
        lo, hi = max(0, i - 8), min(len(score), i + 9)
        depth[i] = (score[lo:i + 1].max() - score[i]) + (score[i:hi].max() - score[i])
    thr = depth.mean() + 0.5 * depth.std()
    bounds: list[float] = []
    for i in np.argsort(-depth):
        if depth[i] < thr:
            break
        t = cand[i]
        if all(abs(t - b) >= min_len for b in bounds):
            bounds.append(t)
    bounds.sort()
    # Split chapters that are still too long at their deepest interior valley.
    edges = [0.0] + bounds + [dur]
    out: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(a)
        while b - a > max_len:
            inside = [(depth[i], cand[i]) for i in range(len(cand)) if a + min_len <= cand[i] <= min(b - min_len, a + max_len)]
            if not inside:
                break
            a = max(inside)[1]
            out.append(a)
    out.append(dur)
    return [(s, e) for s, e in zip(out[:-1], out[1:]) if e - s > 1]


def build(out_dir: str, url: str, model: str, lang: str = "English", embed_model: str =
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", workers: int = 8) -> dict:
    out = Path(out_dir)
    m = json.loads((out / "manifest.json").read_text())
    client = Client(url, model)
    vision_selftest(client)
    t0 = time.perf_counter()
    desc = describe(out, m, client, lang, workers)
    t1 = time.perf_counter()
    spans = chapters(m, embedder(embed_model))
    t2 = time.perf_counter()
    cues = m.get("transcript", [])
    kfs = [(kf["t"], kf["file"]) for s in m["segments"] for kf in s["keyframes"] if "file" in kf]

    def summarize(span):
        a, b = span
        vis = [f"{fmt(t)}: {desc[f]['shows']} | {desc[f]['text'][:200]}" for t, f in kfs if a <= t < b and f in desc]
        prompt = SUMMARIZE.format(start=fmt(a), end=fmt(b), speech=speech_between(cues, a, b, 12000) or "(no speech)",
                                  visuals="\n".join(vis[:20]) or "(no keyframes)", lang=lang)
        try:
            r = client.chat(prompt, max_tokens=700)
        except Exception as e:
            log(f"summarize {fmt(a)} failed: {e}")
            r = {"title": "", "summary": "", "key_points": [], "entities": [], "error": str(e)}
        return r

    with ThreadPoolExecutor(workers) as ex:
        sums = list(ex.map(summarize, spans))
    chs = []
    for i, ((a, b), r) in enumerate(zip(spans, sums)):
        chs.append({
            "id": i + 1, "start": round(a, 2), "end": round(b, 2), "start_ts": fmt(a), "end_ts": fmt(b),
            "segments": [s["id"] for s in m["segments"] if a <= s["start"] < b],
            "title": str(r.get("title", "")).strip(), "summary": str(r.get("summary", "")).strip(),
            "key_points": [str(x) for x in (r.get("key_points") or [])][:5],
            "entities": [str(x) for x in (r.get("entities") or [])][:12],
            "visuals": [{"t": t, "file": f, **desc[f]} for t, f in kfs if a <= t < b and f in desc],
        })
    listing = "\n".join(f"{c['start_ts']} {c['title']}: {c['summary']}" for c in chs)
    try:
        overview = client.chat(OVERVIEW.format(chapters=listing[:14000], lang=lang), max_tokens=600)
    except Exception as e:
        overview = {"title": "", "summary": "", "topics": [], "error": str(e)}
    t3 = time.perf_counter()
    m["meaning"] = {
        "model": model, "lang": lang, "overview": overview, "chapters": chs,
        "timings": {"describe": round(t1 - t0, 1), "chapters": round(t2 - t1, 1), "summarize": round(t3 - t2, 1)},
    }
    (out / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    (out / "meaning.md").write_text(render(m))
    log(f"{len(chs)} chapters; describe {t1 - t0:.0f}s, chapters {t2 - t1:.0f}s, summarize {t3 - t2:.0f}s")
    return m


def render(m: dict) -> str:
    mm = m["meaning"]
    ov = mm["overview"]
    lines = [f"# {ov.get('title') or Path(m['source']).name}", "",
             f"_{fmt(m['duration'])} · {len(mm['chapters'])} chapters · source `{Path(m['source']).name}` · "
             f"details per segment in storyboard.md_", "", ov.get("summary", ""), ""]
    if ov.get("topics"):
        lines += ["Topics: " + ", ".join(map(str, ov["topics"])), ""]
    for c in mm["chapters"]:
        lines.append(f"## {c['id']}. {c['title'] or 'Untitled'}  ({c['start_ts']}–{c['end_ts']})")
        lines.append(c["summary"])
        for p in c["key_points"]:
            lines.append(f"- {p}")
        if c["entities"]:
            lines.append(f"_Named: {', '.join(c['entities'])}_")
        shown = [v for v in c["visuals"] if v.get("shows")]
        if shown:
            lines.append("")
            lines.append("On screen:")
            for v in shown[:8]:
                txt = f' · text: "{v["text"][:160]}"' if v.get("text") else ""
                lines.append(f"- {fmt(v['t'])} `{v['file']}` {v['shows']}{txt}")
        lines.append(f"_segments S{c['segments'][0]:02d}–S{c['segments'][-1]:02d}_" if c["segments"] else "")
        lines.append("")
    return "\n".join(lines)
