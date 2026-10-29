"""The video should represent the same person and observation as event logic."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.pipeline import overlay_people


def test_overlay_uses_identity_and_evidence_representative_without_ghosts():
    class Engine:
        frame_representatives = {1: 1}
        def person_of(self, tid):
            return {1: 1, 9: 1, 12: 2}[tid]

    base = {"flags": {"no_helmet": False}, "foot": None, "head": None, "confirmed": True}
    people = [{**base, "tid": 1, "box": [1, 1, 3, 4]}, {**base, "tid": 9, "box": [1, 1, 4, 4]}]
    hidden = np.array([[1, 1, 4, 4, 9], [4, 4, 6, 8, 12]])
    out = overlay_people(people, hidden, Engine())
    assert [(p["tid"], p["person_id"]) for p in out] == [(1, 1), (12, 2)]
    assert out[0]["confirmed"] and out[1]["hidden"]
