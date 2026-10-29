"""Camera-moved check on a synthetic textured scene: a still camera never trips it, a shift of a few
percent does (once, after persist_s), a passing occluder doesn't, and a moved camera pauses zones.

python -m pytest tests -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.view import ViewCheck  # noqa: E402
from zapdos.zones import ZoneSet  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/rules.yaml").read_text())
SIZE = (1920, 1080)
ZONE = [[700, 600], [1200, 600], [1250, 900], [650, 900]]
FPS = 10


def scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    for _ in range(300):
        x, y = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1])
        w, h = rng.integers(20, 200, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), [int(v) for v in rng.integers(0, 255, 3)], -1)
    return img


def shifted(img, dx):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]), SIZE, borderMode=cv2.BORDER_REFLECT)


def run(frames):
    """frames: function t -> image. Returns [(ts, change)] for every state change over 20 s."""
    view = ViewCheck(CFG["view"], ZONE, SIZE)
    rng = np.random.default_rng(1)
    out = []
    for i in range(20 * FPS):
        ts = i / FPS
        img = cv2.add(frames(ts), rng.integers(0, 6, (SIZE[1], SIZE[0], 3), dtype=np.uint8))  # sensor noise
        change = view.update(ts, img)
        if change is not None:
            out.append((ts, change))
    return out


BASE = scene()


def test_still_camera_never_reads_moved():
    assert run(lambda t: BASE) == []


def test_small_jitter_is_not_a_move():
    assert run(lambda t: shifted(BASE, 8 if int(t) % 2 else 0)) == []  # 0.4% of the width


def test_shift_trips_once_after_persistence_and_recovers():
    persist, every = CFG["view"]["persist_s"], CFG["view"]["check_every_s"]
    moved = shifted(BASE, 0.04 * SIZE[0])
    out = run(lambda t: moved if 5 <= t < 13 else BASE)
    assert [c for _, c in out] == [True, False]
    # decided persist_s after the first moved frame, read one check later (the check runs off-thread)
    assert 5 + persist + every <= out[0][0] <= 5 + persist + 2 * every
    assert 13 + persist + every <= out[1][0] <= 13 + persist + 2 * every


def test_different_view_reads_moved():
    other = scene(seed=7)
    assert [c for _, c in run(lambda t: other if t >= 5 else BASE)] == [True]


def test_passing_occluder_is_not_a_move():
    def frames(t):
        if 6 <= t < 7.5:  # a truck fills most of the view for 1.5 s
            img = BASE.copy()
            img[:, 200:1700] = 40
            return img
        return BASE
    assert run(frames) == []


def test_moved_camera_makes_zones_cant_tell():
    zones = ZoneSet(None, SIZE, CFG["zones"], FPS)
    zones.polygons = {"z": np.array(ZONE, np.float32).reshape(-1, 1, 2)}
    inside = [900, 500, 1000, 800]  # feet at (950, 800), inside the zone
    assert zones.read(1, inside, 0.9)[0]["z"][0] is True
    zones.suspended = True
    assert zones.read(1, inside, 0.9)[0]["z"][0] is None
