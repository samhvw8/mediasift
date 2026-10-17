"""Build the Core ML sound tagger once: CED-small (mispeech/ced-small, pinned) as a batched fp16 model,
log-mel dB (64 windows, 64 mels, 301 frames) -> sigmoid AudioSet probabilities (64, 527).

coremltools has no Python 3.14 build, so this runs in its own throwaway environment (`mise run ced-coreml`):

    uv run --no-project --python 3.13 --with torch==2.7.0 --with torchaudio==2.7.0 \
        --with coremltools==9.0 --with transformers==4.46.3 python scripts/ced_coreml.py OUT_DIR

mediasift reads the result from <cache>/models/ced-small-coreml-<rev>/ (sound.CED_COREML) and falls back to the
sherpa-onnx tagger on the CPU when it is missing. The mel front-end stays in numpy (sound.log_mel): frame power
overflows fp16.
"""
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForAudioClassification

REPO, REV = "mispeech/ced-small", "06bb40c5ec089e96867ebc5246be02441f4a71e4"
B, F, T = 64, 64, 301          # 3 s windows: 1 + 48000 / 160 frames with center=True

out = Path(sys.argv[1])
src = snapshot_download(REPO, revision=REV, allow_patterns=["*.json", "*.py", "model.safetensors"])
model = AutoModelForAudioClassification.from_pretrained(src, trust_remote_code=True).eval()


def _fix_coremltools_cast():
    """coremltools 9.0 casts a folded shape value with dtype(x.val), which numpy 2 refuses for a
    1-element array ('only 0-dimensional arrays can be converted to Python scalars'). CED reads shapes
    in its attention, patch embedding and position embeddings; patch the one line instead of each op."""
    import inspect
    from coremltools.converters.mil.frontend.torch import ops
    src = inspect.getsource(ops._cast)
    assert "dtype(x.val)" in src, "coremltools changed: check whether this patch is still needed"
    exec(src.replace("dtype(x.val)", "dtype(np.asarray(x.val).reshape(-1)[0])"), ops.__dict__)


class Tagger(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel):
        enc = self.m.encoder
        x = torch.unsqueeze(mel, 1).permute(0, 2, 1, 3)
        x = enc.init_bn(x).permute(0, 2, 1, 3)
        return self.m.forward_head(enc.forward_features(x))


ex = torch.randn(B, F, T) * 20 - 30
with torch.no_grad():
    ref = model(ex).logits
tagger = Tagger(model).eval()
with torch.no_grad():
    assert torch.allclose(ref, tagger(ex), atol=1e-5), "the wrapper changed the output"
_fix_coremltools_cast()
ml = ct.convert(torch.jit.trace(tagger, ex), inputs=[ct.TensorType(name="mel", shape=(B, F, T), dtype=np.float32)],
                outputs=[ct.TensorType(name="prob", dtype=np.float32)], convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
ml.short_description = f"{REPO}@{REV[:7]}: log-mel dB ({B}, {F}, {T}) -> AudioSet probabilities ({B}, 527)"
out.mkdir(parents=True, exist_ok=True)
ml.save(str(out / "ced_small_b64.mlpackage"))
err = float(np.abs(ml.predict({"mel": ex.numpy()})["prob"] - ref.numpy()).max())
print(f"saved {out / 'ced_small_b64.mlpackage'}; max |Core ML - PyTorch| on random input {err:.4f}")
assert err < 0.02, "fp16 drifted too far from PyTorch"
