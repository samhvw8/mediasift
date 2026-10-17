"""Keep test runs out of the real cache: usage.jsonl is the feedback log for real use, and test
cards would sit in ~/.cache/mediasift/see forever. Models are shared so tests do not download them."""
import os
import tempfile
from pathlib import Path

import pytest

from mediasift import sound


@pytest.fixture(autouse=True, scope="session")
def _private_cache():
    real = sound.cache_root()
    with tempfile.TemporaryDirectory(prefix="mediasift-test-") as tmp:
        (real / "models").mkdir(parents=True, exist_ok=True)
        (Path(tmp) / "models").symlink_to(real / "models")
        old = os.environ.get("MEDIASIFT_CACHE")
        os.environ["MEDIASIFT_CACHE"] = tmp
        yield Path(tmp)
        if old is None:
            os.environ.pop("MEDIASIFT_CACHE", None)
        else:
            os.environ["MEDIASIFT_CACHE"] = old
