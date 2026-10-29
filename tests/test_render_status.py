"""Human-visible PPE states must preserve unknowns and original head pixels."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.media import _panel_people, _status, _with_head_crops, draw


def test_helmet_and_vest_states_distinguish_unknown_from_missing():
    for value, helmet, vest in [(False, "HELMET: ON", "VEST: ON"),
                                (True, "HELMET: MISSING", "VEST: MISSING"),
                                (None, "HELMET: UNKNOWN", "VEST: UNKNOWN")]:
        assert _status({"no_helmet": value}, "no_helmet") == helmet
        assert _status({"no_vest": value}, "no_vest") == vest
    assert _status({"no_helmet": np.bool_(True)}, "no_helmet") == "HELMET: MISSING"


def test_panel_excludes_unconfirmed_candidates_and_duplicate_persons():
    overlay = {"people": [
        {"tid": 1, "person_id": 1, "confirmed": True},
        {"tid": 9, "person_id": 1, "confirmed": True},
        {"tid": 4, "person_id": 2, "confirmed": False},
        {"tid": 7, "person_id": 3, "hidden": True},
    ]}
    assert [p["tid"] for p in _panel_people(overlay)] == [1]
    assert [p["tid"] for p in _panel_people(overlay, highlight=9)] == [9]


def test_head_crop_is_copied_at_source_resolution():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[100:120, 200:220] = (17, 99, 205)
    overlay = {"ts": 0, "people": [{"tid": 1, "head": [200, 100, 220, 120]}]}
    enriched = _with_head_crops(frame, overlay)
    crop = enriched["people"][0]["head_crop"]
    assert crop.shape == (36, 36, 3)
    frame[:] = 0
    assert tuple(crop[8, 8]) == (17, 99, 205)
    assert "head_crop" not in overlay["people"][0]


def test_annotated_video_labels_display_unknown_and_stable_person_id(monkeypatch):
    frame = np.zeros((720, 1280, 3), np.uint8)
    overlay = {"ts": 1, "people": [{"tid": 9, "person_id": 1, "confirmed": True,
        "box": [100, 100, 160, 280], "head": [110, 110, 130, 130], "foot": None,
        "flags": {"no_helmet": None, "no_vest": False, "zone_entry": None}}]}
    written = []
    real = cv2.putText

    def record(img, text, *args, **kwargs):
        written.append(text)
        return real(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record)
    result = draw(frame, overlay, {})
    assert "P1 / T9" in written
    assert "HELMET: UNKNOWN" in written
    assert "VEST: ON" in written
    assert tuple(result[101, 100]) == (160, 160, 160)  # no green all-clear for an unknown helmet


def test_panel_fits_short_preview_frame():
    image = np.zeros((240, 320, 3), np.uint8)
    people = [{"tid": i, "box": [10, 10, 20, 50], "head": [10, 10, 15, 20],
               "flags": {"no_helmet": None, "no_vest": None}, "foot": None} for i in range(3)]
    assert draw(image, {"ts": 0, "people": people}, {}).shape == image.shape
