"""Person tracking with ByteTrack (ultralytics implementation), plus the per-track state the alert
rules need: whether a track is a confirmed person, and a coarse appearance descriptor.

ByteTrack over DeepSORT: no per-crop ReID network (DeepSORT's cost grows with people on screen and
CCTV crops are too small for good appearance features), and its second association pass keeps
low-score boxes, which carries tracks through partial occlusion behind machines.

Low-score boxes only keep a track alive. Whether a track may *alert* is decided separately by
PersonConfirmation: repeated confident detections with a person-like shape, over some minimum time.
"""
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class PersonTracker:
    def __init__(self, cfg, process_fps):
        if process_fps <= 0 or cfg["track_buffer_s"] < 0:
            raise ValueError("positive process FPS and nonnegative track buffer required")
        args = SimpleNamespace(
            track_high_thresh=cfg["track_high_thresh"],
            track_low_thresh=cfg["track_low_thresh"],
            new_track_thresh=cfg["new_track_thresh"],
            match_thresh=cfg["match_thresh"],
            fuse_score=cfg["fuse_score"],
            track_buffer=int(round(cfg["track_buffer_s"] * process_fps)),
        )
        self.tracker = BYTETracker(args)
        self._none = Boxes(np.zeros((0, 6), np.float32), (1, 1))

    def tick(self):
        """A sample that was skipped: no detections, but one frame of time passes for every track."""
        self.tracker.update(self._none)

    def __call__(self, persons):
        """Returns (N, 6): x1, y1, x2, y2, track_id, conf for tracks matched to a detection in this frame.
        The box is that detection's own box, not the Kalman estimate, so every PPE and floor-point
        reading comes from something the detector actually saw."""
        out = self.tracker.update(persons)
        if len(out) == 0:
            return np.zeros((0, 6))
        out = out[:, :8].astype(np.float64)
        idx = out[:, 7].astype(int)
        raw = np.asarray(persons.xyxy, np.float64).reshape(-1, 4)
        ok = (idx >= 0) & (idx < len(raw))
        out[ok, :4] = raw[idx[ok]]
        return out[:, :6]

    def hidden(self, max_frames):
        """Tracks lost for <= max_frames that the tracker still holds: (N, 5) Kalman-predicted x1, y1, x2,
        y2, track_id. For drawing only; nothing is read from a box nobody can see."""
        t = self.tracker
        rows = [[*s.xyxy, s.track_id] for s in t.lost_stracks if t.frame_id - s.end_frame <= max_frames]
        return np.array(rows, float).reshape(-1, 5)


class PersonConfirmation:
    """A track is a confirmed person once it has had `min_hits` detections at >= `min_conf` with a
    person-like box (height >= min_height_px, height/width within `aspect`), and at least `min_age_s`
    has passed since it first appeared. Confirmation is sticky for the track. Readings from an
    unconfirmed track still gather evidence; they just can't alert yet (events.py)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.first = {}  # track_id -> first ts
        self.hits = {}   # track_id -> qualifying detections so far
        self.done = set()

    def plausible(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        lo, hi = self.cfg["aspect"]
        return h >= self.cfg["min_height_px"] and w > 0 and lo <= h / w <= hi

    def update(self, tid, box, conf, ts):
        if tid in self.done:
            return True
        self.first.setdefault(tid, ts)
        if conf >= self.cfg["min_conf"] and self.plausible(box):
            self.hits[tid] = self.hits.get(tid, 0) + 1
        if self.hits.get(tid, 0) >= self.cfg["min_hits"] and ts - self.first[tid] >= self.cfg["min_age_s"] - 1e-6:
            self.done.add(tid)
            return True
        return False


TORSO_X, TORSO_Y = (0.2, 0.8), (0.2, 0.6)  # clothing region used for appearance, as box fractions


def appearance(frame, box, min_px):
    """Normalised HSV colour histogram of the torso, or None when the person is too small for colour
    to mean much. Used only to veto re-linking two clearly different-looking people."""
    x1, y1, x2, y2 = box[:4]
    h, w = y2 - y1, x2 - x1
    if frame is None or h < min_px or w < 4:
        return None
    H, W = frame.shape[:2]
    a, b = int(max(0, x1 + TORSO_X[0] * w)), int(min(W, x1 + TORSO_X[1] * w))
    c, d = int(max(0, y1 + TORSO_Y[0] * h)), int(min(H, y1 + TORSO_Y[1] * h))
    if b - a < 4 or d - c < 4:
        return None
    hsv = cv2.cvtColor(frame[c:d, a:b], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()


def appearance_distance(a, b):
    """Bhattacharyya distance between two descriptors: 0 identical, 1 nothing in common."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
