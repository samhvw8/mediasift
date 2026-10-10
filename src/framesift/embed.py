"""EmbeddingGemma 2 in one shared space for frames and speech, run by llama.cpp.

Every command starts its own llama-server on a private Unix socket and stops it
when it is done, so nothing keeps running between calls:
  index  text model + vision projector, ~3 s to start;
  find   text model only (--no-mmproj), ~1.5 s to start plus one query, 0.5 GB.

The model is ggml-org/embeddinggemma-2-GGUF (Q8_0). llama-server downloads it
once (~0.9 GB, Hugging Face cache) and later starts read the cache (--offline).

`index --on HOST` runs the server on HOST over ssh and forwards its socket, so
a GPU box embeds the frames while decoding stays local. Vectors from Metal and
CUDA agree to cosine 0.9998 (research/261008-model-ratio/engine-test.md), so an
index built on one machine is searched from the other.

Measured against the PyTorch reference (same engine-test.md): text vectors
cosine 0.9999; frame vectors 0.97 because the image resize differs, with the
same retrieval (hit@3 0.87 on 46 visual questions). Image budget 70 tokens:
the 280-token default is ~6x slower for a 0.955 pairwise-similarity match.

FRAMESIFT_LLAMA_SERVER  local llama-server binary (default: the llama.cpp that framesift's
                        mise.toml pins, found from any folder; then `llama-server` on PATH)
FRAMESIFT_REMOTE_LLAMA_SERVER  the binary on an --on host (default: llama-server)
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

MODEL = "google/embeddinggemma-2"
GGUF = "ggml-org/embeddinggemma-2-GGUF:Q8_0"
DIM = 768
IMAGE_TOKENS = 70        # vision soft tokens per image: 70 | 140 | 280 | 560 | 1120
QUERY = "task: search result | query: {}"
DOCUMENT = "title: none | text: {}"
CTX = 8192               # one batch; non-causal embedding needs each input inside one ubatch
START_TIMEOUT_S = float(os.environ.get("FRAMESIFT_EMBED_START_TIMEOUT", "900"))  # first start downloads
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def log(msg: str) -> None:
    print(f"framesift: {msg}", file=sys.stderr)


def _llama_build(v: dict) -> int:
    n = v.get("version", "").lstrip("b")
    return int(n) if n.isdigit() else -1


def local_binary() -> str:
    """llama.cpp is pinned in framesift's own mise.toml, not installed globally,
    so its mise shim works only inside the repo. Ask mise for the install
    itself, which works from any folder."""
    if os.environ.get("FRAMESIFT_LLAMA_SERVER"):
        return os.environ["FRAMESIFT_LLAMA_SERVER"]
    try:
        r = subprocess.run(["mise", "ls", "--installed", "--json", "github:ggml-org/llama.cpp"],
                           capture_output=True, text=True, timeout=10)
        installs = json.loads(r.stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        installs = []
    for v in sorted(installs, key=_llama_build, reverse=True):
        p = Path(v.get("install_path", "")) / "llama-server"
        if p.is_file():
            return str(p)
    found = shutil.which("llama-server")
    if found and "/mise/shims/" not in found:  # a shim outside the repo has no version to run
        return found
    raise SystemExit("framesift needs llama.cpp's llama-server for index/find: run `mise install` in the "
                     "framesift repo (it pins llama.cpp), or set FRAMESIFT_LLAMA_SERVER")


class _UnixHTTP(http.client.HTTPConnection):
    def __init__(self, sock_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.sock_path = sock_path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.sock_path)
        self.sock = s


class Embedder:
    """A llama-server for the lifetime of a `with` block.

    The server runs under a small shell that waits on its stdin; when this
    process closes that pipe (normally, or by dying), the shell kills the
    server and removes the socket. That holds over ssh too, so no server is
    left behind on either machine."""

    def __init__(self, vision: bool, host: str | None = None):
        self.vision, self.host = vision, host
        self.local_sock = str(Path(tempfile.gettempdir()) / f"framesift-eg2-{uuid.uuid4().hex[:12]}.sock")
        self.proc: subprocess.Popen | None = None
        self._identity: dict | None = None

    def _server_args(self, binary: str, sock: str) -> str:
        args = [binary, "-hf", GGUF, "--embeddings", "-c", str(CTX), "-b", str(CTX), "-ub", str(CTX),
                "-ngl", "99", "--no-webui", "--host", sock]
        args += ["--image-min-tokens", str(IMAGE_TOKENS), "--image-max-tokens", str(IMAGE_TOKENS)] \
            if self.vision else ["--no-mmproj"]
        cmd = shlex.join(args)
        # The cache first; download only when it is missing. A failed start leaves its socket behind.
        return f"{{ {cmd} --offline || {{ rm -f {shlex.quote(sock)}; {cmd}; }}; }}"

    def _script(self, binary: str, sock: str, logf: str) -> str:
        # Plain sh, no signal traps (zsh skips EXIT traps when killed). When stdin
        # closes (we are done, or died) the foreground removes the socket and log
        # and kills the group. When the server exits by itself (a failed start)
        # the background removes the socket, keeps the log and kills the group,
        # so the caller sees the exit.
        s, lf = shlex.quote(sock), shlex.quote(logf)
        body = (f"rm -f {s}; {{ {self._server_args(binary, sock)} > {lf} 2>&1; rm -f {s}; kill 0; }} & "
                f"cat > /dev/null; rm -f {s} {lf}; kill 0")
        return f"exec sh -c {shlex.quote(body)}"

    def __enter__(self) -> "Embedder":
        if self.host:
            remote_sock = f"/tmp/{Path(self.local_sock).name}"
            self.logf = remote_sock[:-5] + ".log"
            binary = os.environ.get("FRAMESIFT_REMOTE_LLAMA_SERVER", "llama-server")
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=yes",
                   "-o", "LogLevel=ERROR",
                   "-L", f"{self.local_sock}:{remote_sock}", self.host, self._script(binary, remote_sock, self.logf)]
            where = f"{self.host} (log {self.host}:{self.logf})"
        else:
            binary = local_binary()
            self.logf = self.local_sock[:-5] + ".log"
            cmd = ["sh", "-c", self._script(binary, self.local_sock, self.logf)]
            where = f"llama-server (log {self.logf})"
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL if not self.host else None, start_new_session=True)
        try:
            self._wait_ready(where)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        props = self._get("/props")
        self._identity = {"model": MODEL, "gguf": GGUF, "dim": DIM, "image_tokens": IMAGE_TOKENS,
                          "engine": f"llama.cpp {props.get('build_info', '?')}", "host": self.host or "local"}
        log(f"embedder ready on {self.host or 'this machine'} ({'text + vision' if self.vision else 'text'}) "
            f"in {time.perf_counter() - t0:.1f}s")
        return self

    def _wait_ready(self, where: str) -> None:
        start, said = time.time(), False
        while time.time() - start < START_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise SystemExit(f"embedder exited ({self.proc.returncode}) before it was ready:\n{self._log_tail()}")
            try:
                if self._get("/health", timeout=2).get("status") == "ok":
                    return
            except (OSError, http.client.HTTPException, ValueError):
                pass
            if not said and time.time() - start > 10:
                log(f"waiting for {where}; the first start downloads the model (~0.9 GB)")
                said = True
            time.sleep(0.05)
        raise SystemExit(f"embedder not ready after {START_TIMEOUT_S:.0f}s:\n{self._log_tail()}")

    def _log_tail(self) -> str:
        if self.host:
            return f"see {self.host}:{self.logf}"
        try:
            return Path(self.logf).read_text()[-2000:]
        except OSError:
            return f"no log at {self.logf}"

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # the shell's `cat` ends, its trap stops the server
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                os.killpg(self.proc.pid, 15)
        Path(self.local_sock).unlink(missing_ok=True)
        if not self.host:
            Path(self.logf).unlink(missing_ok=True)

    def _get(self, path: str, timeout: float = 30) -> dict:
        c = _UnixHTTP(self.local_sock, timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                return {"status": r.status}
            return json.loads(body)
        finally:
            c.close()

    def _embed(self, inputs: list) -> np.ndarray:
        c = _UnixHTTP(self.local_sock, 600)
        try:
            c.request("POST", "/v1/embeddings", body=json.dumps({"input": inputs}),
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
        finally:
            c.close()
        if r.status != 200:
            raise RuntimeError(f"embedder HTTP {r.status}: {body[:500]!r}")
        data = sorted(json.loads(body)["data"], key=lambda d: d["index"])
        v = np.array([d["embedding"] for d in data], np.float32)
        if v.shape != (len(inputs), DIM):
            raise RuntimeError(f"embedder returned shape {v.shape}, expected ({len(inputs)}, {DIM})")
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def identity(self) -> dict:
        return dict(self._identity)

    def texts(self, texts: list[str], kind: str, batch: int = 32) -> np.ndarray:
        tpl = QUERY if kind == "query" else DOCUMENT
        if not texts:
            return np.zeros((0, DIM), np.float32)
        return np.concatenate([self._embed([tpl.format(t) for t in texts[i:i + batch]])
                               for i in range(0, len(texts), batch)])

    def images(self, paths: list[str], batch: int = 8) -> np.ndarray:
        if not self.vision:
            raise RuntimeError("this embedder was started without the vision projector")
        out = []
        for i in range(0, len(paths), batch):
            parts = []
            for p in paths[i:i + batch]:
                url = f"data:{MIME[Path(p).suffix.lower()]};base64," + base64.b64encode(Path(p).read_bytes()).decode()
                parts.append({"content": [{"type": "image_url", "image_url": {"url": url}}]})
            out.append(self._embed(parts))
        return np.concatenate(out) if out else np.zeros((0, DIM), np.float32)
