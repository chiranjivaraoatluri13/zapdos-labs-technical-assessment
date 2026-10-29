"""Has the camera moved since its zones were drawn?

Zones are pixel polygons, so a bumped, re-angled or re-zoomed camera silently slides them off the
floor they guard: zone alerts turn into false alarms or silent misses. Every check_every_s the frame
is matched to a reference view (ORB features, RANSAC rotation + scale + shift) and the zone corners
are pushed through that transform. If a corner moves more than max_shift of the frame width, or the
view no longer matches at all, for persist_s, the camera has moved: the pipeline emits one
camera_moved event and every zone reads "can't tell" until the view matches again. PPE has no
geometry and carries on. The reference is the image saved with the zone file
(scripts/zone_reference.py), else the run's first frame. Each check (~10 ms) runs on a worker thread
and is read at the next one, so the frame loop never waits for it.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


class ViewCheck:
    def __init__(self, cfg, points, frame_size, reference=None):
        self.cfg = cfg
        self.scale = cfg["width"] / frame_size[0]
        self.points = np.asarray(points, np.float32).reshape(-1, 2) * self.scale  # at check width
        self.orb = cv2.ORB_create(cfg["features"])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref = self.features(cv2.imread(str(reference))) if reference else None
        self.moved = False
        self.since = None      # ts of the first check that disagreed with the current state
        self.next_ts = 0.0
        self.shift = 0.0       # last measured corner shift, share of frame width (inf = no match)
        self.pool = ThreadPoolExecutor(1)
        self.job = None        # (ts, future) of the check in flight

    def features(self, frame):
        h = round(frame.shape[0] * self.cfg["width"] / frame.shape[1])
        gray = cv2.cvtColor(cv2.resize(frame, (self.cfg["width"], h), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def measure(self, feats):
        """Largest zone-corner shift between the reference and these features, as a share of the frame
        width; inf when too few features agree on one transform (a different or unreadable view)."""
        (kp0, des0), (kp1, des1) = self.ref, feats
        n = self.cfg["min_inliers"]
        if not (_enough(kp0, des0, n) and _enough(kp1, des1, n)):
            return math.inf
        matches = self.matcher.match(des0, des1)
        if len(matches) < n:
            return math.inf
        src = np.float32([kp0[m.queryIdx].pt for m in matches])
        dst = np.float32([kp1[m.trainIdx].pt for m in matches])
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=3.0)
        if affine is None or inliers.sum() < n:
            return math.inf
        warped = self.points @ affine[:, :2].T + affine[:, 2]
        return float(np.linalg.norm(warped - self.points, axis=1).max()) / self.cfg["width"]

    def update(self, ts, frame):
        """Call on every processed frame. Returns True when the camera has just been judged moved, False
        when the view has just matched again, None otherwise. Each decision uses the previous check's
        result, so it comes one check_every_s after the check that settled it; `since` is the first
        check of the disagreeing run (the camera_moved event's start_ts)."""
        if ts + 1e-6 < self.next_ts:
            return None
        self.next_ts = ts + self.cfg["check_every_s"]
        if self.ref is None:
            self.ref = self.features(frame)
            return None
        done, self.job = self.job, (ts, self.pool.submit(lambda: self.measure(self.features(frame))))
        if done is None:
            return None
        checked_ts, result = done
        self.shift = result.result()
        if (self.shift > self.cfg["max_shift"]) == self.moved:
            self.since = None
            return None
        if self.since is None:
            self.since = checked_ts
        if checked_ts - self.since + 1e-6 < self.cfg["persist_s"]:
            return None
        self.moved = not self.moved
        return self.moved


def _enough(keypoints, descriptors, n):
    return descriptors is not None and len(keypoints) >= n
