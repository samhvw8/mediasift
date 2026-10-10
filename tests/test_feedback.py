"""Corrections from agents (`framesift fix`) teach the card what a tagger label usually is."""
import json

import pytest

from framesift import feedback, see


@pytest.fixture(autouse=True)
def _own_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMESIFT_CACHE", str(tmp_path / "cache"))
    feedback.learned.cache_clear()
    yield
    feedback.learned.cache_clear()


def _card_dir(tmp_path, label="Throat clearing"):
    d = tmp_path / "card"
    d.mkdir()
    (d / "sensed.json").write_text(json.dumps({"card": see.CARD_VERSION, "sounds": [
        {"start": 20.5, "end": 22.5, "label": label, "kind": "bed", "times": []},
        {"start": 30.0, "end": 31.0, "label": "Plop", "kind": "hits", "times": [30.2]}]}))
    return d


def test_fix_records_what_the_card_said(tmp_path):
    d = _card_dir(tmp_path)
    row = feedback.record(d, 21.0, "cough")
    assert row["was"] == "Throat clearing" and row["says"] == "cough"
    assert feedback.record(d, 30.25, "cough")["was"] == "Plop"
    assert feedback.record(d, 50.0, "door slam")["was"] is None       # a sound the card missed


def test_three_agreeing_fixes_become_a_learned_name(tmp_path):
    d = _card_dir(tmp_path)
    assert feedback.learned() == {}
    for _ in range(3):
        feedback.record(d, 21.0, "Cough")
    assert feedback.learned() == {"Throat clearing": "cough"}
    feedback.learned.cache_clear()
    assert see._snd_name("Throat clearing") == "throat clearing (usually: cough)"


def test_split_fixes_teach_nothing(tmp_path):
    d = _card_dir(tmp_path)
    for says in ("cough", "cough", "laugh", "sneeze"):
        feedback.record(d, 21.0, says)
    assert "Throat clearing" not in feedback.learned()
