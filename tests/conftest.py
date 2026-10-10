"""Keep test runs out of the real cache: usage.jsonl is the feedback log for real use, and test
cards would sit in ~/.cache/framesift/see forever. Models are shared so tests do not download them."""
import os
import tempfile
from pathlib import Path

import pytest

from framesift import sound


@pytest.fixture(autouse=True, scope="session")
def _private_cache():
    real = sound.cache_root()
    with tempfile.TemporaryDirectory(prefix="framesift-test-") as tmp:
        (real / "models").mkdir(parents=True, exist_ok=True)
        (Path(tmp) / "models").symlink_to(real / "models")
        old = os.environ.get("FRAMESIFT_CACHE")
        os.environ["FRAMESIFT_CACHE"] = tmp
        yield Path(tmp)
        if old is None:
            os.environ.pop("FRAMESIFT_CACHE", None)
        else:
            os.environ["FRAMESIFT_CACHE"] = old
