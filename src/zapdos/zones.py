"""Restricted zones: per-camera floor polygons tested against each person's ground point.

The ground point is the bottom-centre of the box, i.e. the feet, because the polygon is drawn on the
floor; a box overlapping the polygon from 1 m in front is not inside it. Each frame the point is one of:
  visible   - the box bottom is plausibly the feet: not at the bottom frame edge, not ending on a
              configured occluder (the zone file's optional "occluders" polygons: carts, rails,
              machine fronts), and not much shorter than the track's recent full height;
  estimated - feet hidden, so top + full height from a full-body sighting of this track within
              zones.feet_history_s (decayed tallest height), when that lands inside the image;
  unknown   - feet hidden with no recent full-body sighting, or the estimate falls off the image.
              The zone reads "can't tell": it can't start an entry and leaves the debounced
              inside/outside state as it was.
Entering needs the point a margin inside the polygon and leaving needs it the same margin outside,
so someone standing on the edge doesn't flicker. While the camera is judged moved (view.py) every
zone reads "can't tell".
"""
import json
from pathlib import Path

import cv2
import numpy as np

BOTTOM_EDGE_PX = 2      # a box ending this close to the bottom of the frame has its feet cut off
HIDDEN_FEET_CONF = 0.7  # an estimated ground point is weaker evidence than visible feet
OCCLUDER_MARGIN = 0.05  # box bottom within this x person height of an occluder = ends on it


class ZoneSet:
    def __init__(self, path, frame_size, cfg, fps):
        self.cfg = cfg
        self.polygons = {}  # zone_id -> polygon in this video's pixels
        self.occluders = []  # polygons in this video's pixels that hide feet and torsos
        self.camera_id = None
        self.reference = None  # image of the view the polygons were drawn on
        self.suspended = False  # camera moved: zones no longer match the floor
        self.frame_w, self.frame_h = frame_size
        self.inside = {}   # (track_id, zone_id) -> debounced inside state
        self.full = {}     # track_id -> (ts of last full-body sighting, decaying tallest full-body height)
        self.ts = 0.0
        if path is None:
            return
        with open(path) as f:
            data = json.load(f)
        self.camera_id = data.get("camera_id")
        if data.get("reference"):
            self.reference = Path(path).parent / data["reference"]
        ref_w, ref_h = data.get("frame_size", frame_size)
        scale = np.array([frame_size[0] / ref_w, frame_size[1] / ref_h], np.float32)
        for z in data["zones"]:
            self.polygons[z["zone_id"]] = (np.array(z["polygon"], np.float32) * scale).reshape(-1, 1, 2)
        for o in data.get("occluders", []):
            self.occluders.append((np.array(o["polygon"] if isinstance(o, dict) else o, np.float32) * scale)
                                  .reshape(-1, 1, 2))

    def at(self, ts):
        """Set the time of the frame about to be read (for the full-body history)."""
        self.ts = ts

    def _on_occluder(self, pt, h):
        return any(cv2.pointPolygonTest(p, (float(pt[0]), float(pt[1])), True) > -OCCLUDER_MARGIN * h
                   for p in self.occluders)

    def ground_point(self, track_id, box):
        """((x, y) or None, source) with source "visible", "estimated" or "unknown"."""
        x1, y1, x2, y2 = box[:4]
        h, cx = y2 - y1, (x1 + x2) / 2
        seen = self.full.get(track_id)
        recent = seen is not None and self.ts - seen[0] <= self.cfg.get("feet_history_s", np.inf) + 1e-6
        full = seen[1] * self.cfg["height_decay_per_s"] ** (self.ts - seen[0]) if recent else 0.0
        cut = y2 >= self.frame_h - BOTTOM_EDGE_PX
        short = recent and h < self.cfg["hidden_feet_ratio"] * full
        if not (cut or short or self._on_occluder((cx, y2), h)):
            self.full[track_id] = (self.ts, max(h, full))
            return (cx, y2), "visible"
        if not recent:
            return None, "unknown"
        foot_y = y1 + max(full, h)
        if foot_y > self.frame_h:
            return None, "unknown"
        return (cx, foot_y), "estimated"

    def read(self, track_id, box, conf):
        """Returns ({zone_id: (inside, conf, reason)}, (ground_point, source)) for this person; ({}, None)
        when the camera has no zones."""
        if not self.polygons:
            return {}, None
        pt, source = self.ground_point(track_id, box)
        if self.suspended:
            return {zid: (None, 0.0, "camera_moved") for zid in self.polygons}, (pt, source)
        if pt is None:
            return {zid: (None, 0.0, "feet_unknown") for zid in self.polygons}, (pt, source)
        margin = self.cfg["hysteresis"] * (box[3] - box[1])
        out = {}
        for zid, poly in self.polygons.items():
            d = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)  # >0 inside
            was = self.inside.get((track_id, zid), False)
            now = d > -margin if was else d > margin
            self.inside[(track_id, zid)] = now
            out[zid] = (now, float(conf) * (HIDDEN_FEET_CONF if source == "estimated" else 1.0), None)
        return out, (pt, source)
