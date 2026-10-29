"""Per-frame PPE reading for each tracked person, in two steps.

measure() collects the per-person evidence: the head box (detected or estimated), the classifier's
p(bare) for it, the assigned vest confidence, which frame edges cut the person, how much of the
torso is visible, which items had an ambiguous owner, and the detector-method helmet reading.
decide() turns that evidence into readings with the classifier threshold, the head gates and the
vest rule. The split lets evaluation cache measure() once per clip and re-run decide() with other
settings in seconds.

A reading is (value, evidence_conf, reason): value True = violation, False = compliant, None = can't
tell, and reason says why it can't tell (None for a decided reading). "Can't tell" never counts as a
violation; that is what keeps tiny, occluded or ambiguous people from alerting.

Head, helmet and vest boxes are matched to people one-to-one per item type (Hungarian on the
distance from each person's head or torso band). When a box could just as well belong to someone
else (the second-best owner is within ppe.ambiguity_margin), both people read "can't tell" for that
item rather than one of them being picked arbitrarily.

Helmet state comes from one of two methods (config ppe.helmet_method):
  classifier - head crop from the native frame -> HelmetClassifier p(bare). Default.
  detector   - SH17 helmet box over the SH17 head box. Kept as the measured baseline: on SteelBench
               it calls 47% of helmeted workers bare, because SH17 rarely fires on 17-42 px helmets.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .helmet import head_from_person

HEAD_BAND_TOP = -0.1    # a head may poke above a tight person box by this share of its height
HELMET_BAND_TOP = -0.2  # a helmet box reaches higher than the head under it
ABSENT_VEST_CONF = 0.5  # no vest detected is weaker evidence than a detected vest
DUPLICATE_IOU = 0.5     # two boxes of one item type overlapping this much are one object
COVER_GRID = 24         # torso visibility is sampled on a grid this many cells a side
BOTTOM_CUT_PX = 3       # a box ending this close to the bottom of the frame is cut there
MIN_BODY_ASPECT = 2.0   # a standing person is at least this many times taller than wide


def unknown(reason):
    return None, 0.0, reason


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def _area(a):
    return max(a[2] - a[0], 0) * max(a[3] - a[1], 0)


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _band_distance(cx, cy, person, y_band):
    """How far the point (cx, cy) is from the centre of the person's band, in person widths and
    heights; None when the point is outside the person box or the band."""
    x1, y1, x2, y2 = person[:4]
    h = y2 - y1
    lo, hi = y1 + y_band[0] * h, y1 + y_band[1] * h
    if not (x1 <= cx <= x2 and lo <= cy <= hi) or x2 <= x1:
        return None
    return abs(cy - (lo + hi) / 2) / h + abs(cx - (x1 + x2) / 2) / (x2 - x1)


def _dedupe(parts):
    """Drop boxes that duplicate a more confident box of the same item."""
    keep = []
    for p in sorted(parts, key=lambda p: -p[4]):
        if all(_iou(p, k) < DUPLICATE_IOU for k in keep):
            keep.append(p)
    return keep


def match_parts(parts, persons, y_band, margin, swaps_matter=True):
    """One-to-one owners for one item type. Returns ({person i: [part]}, {ambiguous person i}).

    A person is ambiguous for the item when a box assigned to someone else is within `margin` of
    being theirs instead and they got none, or (swaps_matter) when exchanging two people's boxes
    costs less than `margin`. Ambiguous people get no box for the item."""
    parts = _dedupe(list(parts))
    n, m = len(persons), len(parts)
    if not n or not m:
        return {}, set()
    d = np.full((n, m), np.inf)
    for j, p in enumerate(parts):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        for i, person in enumerate(persons):
            v = _band_distance(cx, cy, person, y_band)
            if v is not None:
                d[i, j] = v
    finite = np.isfinite(d)
    if not finite.any():
        return {}, set()
    big = (np.nanmax(np.where(finite, d, np.nan)) + 1) * (n + m + 1)
    rows, cols = linear_sum_assignment(np.where(finite, d, big))
    assigned = {i: j for i, j in zip(rows, cols) if finite[i, j]}
    ambiguous = set()
    for i, j in assigned.items():
        for k in range(n):
            if k == i or not finite[k, j]:
                continue
            if k not in assigned:
                if d[k, j] - d[i, j] < margin:
                    ambiguous |= {i, k}
            elif swaps_matter:
                b = assigned[k]
                if finite[i, b] and d[i, b] + d[k, j] - d[i, j] - d[k, b] < margin:
                    ambiguous |= {i, k}
    owner = {i: [parts[j]] for i, j in assigned.items() if i not in ambiguous}
    return owner, ambiguous


def head_candidates(heads, helmets, persons, head_region, margin=0.0):
    """The detected head and helmet boxes matched to each person: ({i: [head]}, {i: [helmet]})."""
    return (match_parts(heads, persons, (HEAD_BAND_TOP, head_region), margin)[0],
            match_parts(helmets, persons, (HELMET_BAND_TOP, head_region), margin)[0])


def cut_edges(box, frame_shape, margin):
    """Which frame edges the box touches: a subset of "ltrb". A person cut by the side of the frame is
    a sliver whose PPE can't be judged; one cut at the top has no visible head."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]
    return "".join(e for e, hit in zip("ltrb", (x1 <= margin, y1 <= margin, x2 >= w - margin, y2 >= h - margin))
                   if hit)


def torso_visibility(i, persons, torso_band, frame_shape=None, occluders=()):
    """(share of person i's torso band inside the frame, share of it covered by occluders or by people
    standing in front). In front = box bottom lower in the image, the usual case for a raised camera.
    A box ending at the bottom edge only shows the top of the body, so its torso band is placed on
    an estimated full height (at least MIN_BODY_ASPECT x its width) rather than on the visible part."""
    x1, y1, x2, y2 = persons[i][:4]
    h = y2 - y1
    if frame_shape is not None and y2 >= frame_shape[0] - BOTTOM_CUT_PX:
        h = max(h, MIN_BODY_ASPECT * (x2 - x1))
    t = (x1, y1 + torso_band[0] * h, x2, y1 + torso_band[1] * h)
    tw, th = t[2] - t[0], t[3] - t[1]
    if tw <= 0 or th <= 0:
        return 0.0, 1.0
    in_frame = 1.0
    if frame_shape is not None:
        fh, fw = frame_shape[:2]
        in_frame = _inter(t, (0, 0, fw, fh)) / (tw * th)
    sx, sy = COVER_GRID / tw, COVER_GRID / th
    mask = np.zeros((COVER_GRID, COVER_GRID), np.uint8)
    for k, other in enumerate(persons):
        if k != i and other[3] > y2 and _inter(other, t) > 0:
            a = (int(np.floor((other[0] - t[0]) * sx)), int(np.floor((other[1] - t[1]) * sy)))
            b = (int(np.ceil((other[2] - t[0]) * sx)) - 1, int(np.ceil((other[3] - t[1]) * sy)) - 1)
            cv2.rectangle(mask, a, b, 1, -1)
    for poly in occluders:
        pts = (np.asarray(poly, np.float32).reshape(-1, 2) - [t[0], t[1]]) * [sx, sy]
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return in_frame, float(mask.mean())


def _head_box(heads, helmets, person_box):
    """The most confident detected head or helmet box, else the head position estimated from the person
    box. Returns (box, source) with source "head", "helmet" or "est"."""
    cands = [(h, "head") for h in heads] + [(h, "helmet") for h in helmets]
    if not cands:
        return head_from_person(person_box), "est"
    hb, src = max(cands, key=lambda c: c[0][4])
    return tuple(float(v) for v in hb[:4]), src


def measure(person_boxes, det, cfg, frame=None, helmet_clf=None, occluders=()):
    """Evidence per person: {"head", "head_src" (head|helmet|est), "p_bare", "rule_helmet", "vest",
    "cut", "ambiguous" (items), "torso_in_frame", "torso_cover"}. p_bare is None without a classifier."""
    margin = cfg.get("ambiguity_margin", 0.0)
    persons = [tuple(float(v) for v in b[:4]) for b in person_boxes]
    heads, amb_head = match_parts(det.heads, persons, (HEAD_BAND_TOP, cfg["head_region"]), margin)
    helmets, amb_helmet = match_parts(det.helmets, persons, (HELMET_BAND_TOP, cfg["head_region"]), margin)
    vcfg = cfg["vest"]
    vests, amb_vest = (match_parts(det.vests, persons, vcfg["torso"], margin, swaps_matter=False)
                       if vcfg["enabled"] else ({}, set()))
    shape = frame.shape if frame is not None else None

    out = []
    for i, box in enumerate(persons):
        hb, src = _head_box(heads.get(i, []), helmets.get(i, []), box)
        ambiguous = {k for k, s in (("head", amb_head), ("helmet", amb_helmet), ("vest", amb_vest)) if i in s}
        m = {"head": hb, "head_src": src, "p_bare": None,
             "rule_helmet": _helmet(heads.get(i, []), helmets.get(i, []), cfg),
             "vest": float(max((v[4] for v in vests.get(i, [])), default=0.0)),
             "cut": cut_edges(box, shape, cfg["edge_margin_px"]) if shape is not None else "",
             "ambiguous": sorted(ambiguous)}
        if vcfg["enabled"] and m["vest"] == 0:
            m["torso_in_frame"], m["torso_cover"] = torso_visibility(i, persons, vcfg["torso"], shape, occluders)
        out.append(m)
    if helmet_clf is not None and out:
        eligible = [m for m in out if _classifier_eligible(m, cfg.get("classifier", {})) is None]
        for m, p in zip(eligible, helmet_clf.p_bare(frame, [m["head"] for m in eligible])):
            m["p_bare"] = float(p)
    return out


def decide(meas, person_boxes, cfg, threshold=None):
    """Readings {"no_helmet": (v, c, why), "no_vest": (v, c, why)} per person from measure()'s output.
    threshold: p(bare) cutoff for the classifier (see helmet_threshold)."""
    readings = []
    for m, box in zip(meas, person_boxes):
        r = {"no_helmet": _helmet_reading(m, cfg, threshold)}
        if cfg["vest"]["enabled"]:
            r["no_vest"] = _vest_reading(m, box, cfg["vest"])
        readings.append(r)
    return readings


def helmet_threshold(cfg, helmet_clf):
    """The p(bare) cutoff: ppe.classifier.threshold if set, else the one shipped with the weights."""
    value = cfg.get("classifier", {}).get("threshold")
    return value if value is not None else (helmet_clf.threshold if helmet_clf else None)


def read_ppe(person_boxes, det, cfg, frame=None, helmet_clf=None):
    """measure + decide. Returns (readings, head_boxes)."""
    meas = measure(person_boxes, det, cfg, frame, helmet_clf)
    return decide(meas, person_boxes, cfg, helmet_threshold(cfg, helmet_clf)), [m["head"] for m in meas]


def _helmet_reading(m, cfg, threshold):
    if any(e in m.get("cut", "") for e in "lrt"):
        return unknown("edge_cut")  # a sliver at the side, or the head is above the frame
    if {"head", "helmet"} & set(m.get("ambiguous", ())):
        return unknown("ambiguous_owner")
    if cfg["helmet_method"] == "classifier":
        why = _classifier_eligible(m, cfg.get("classifier", {}))
        if why is not None:
            return unknown(why)
        if m["p_bare"] is None:
            return unknown("no_classifier")
        return _classifier_helmet(m["p_bare"], cfg.get("classifier", {}), threshold)
    return m["rule_helmet"]


def _classifier_eligible(m, ccfg):
    """None when the head crop is worth classifying, else the reason it is not."""
    hb = m["head"]
    if any(e in m.get("cut", "") for e in "lrt"):
        return "edge_cut"
    if {"head", "helmet"} & set(m.get("ambiguous", ())):
        return "ambiguous_owner"
    # an estimated head position (no head/helmet box) is too often wrong to call "bare":
    # bent-over workers, and false person boxes on machinery
    if ccfg.get("require_head_det") and m["head_src"] == "est":
        return "no_head_box"
    if hb[3] - hb[1] < ccfg.get("min_head_px", 0):
        return "head_small"
    return None


def _classifier_helmet(p, ccfg, threshold):
    band = ccfg.get("unknown_band")
    if band:
        lo, hi = band
        if p >= hi:
            return True, p, None
        if p <= lo:
            return False, 1 - p, None
        return unknown("classifier_uncertain")
    bare = p >= threshold
    return bare, (p if bare else 1 - p), None


def _helmet(heads, helmets, cfg):
    """The detector-method reading: a helmet box over the most confident head box."""
    if not heads:
        # helmet visible but no head box: compliant; nothing visible: can't tell
        return (False, float(max(h[4] for h in helmets)), None) if helmets else unknown("no_head_box")
    head = max(heads, key=lambda h: h[4])
    hh = head[3] - head[1]
    if hh < cfg["min_head_px"]:
        return unknown("head_small")
    grown = (head[0], head[1] - cfg["helmet_expand"] * hh, head[2], head[3])
    for hm in helmets:
        if _inter(hm, grown) >= cfg["helmet_overlap"] * _area(hm):
            return False, float(hm[4]), None
    return True, float(head[4]), None


def _vest_reading(m, box, vcfg):
    cut = m.get("cut", "")
    if "l" in cut or "r" in cut:
        return unknown("edge_cut")
    if m["vest"] > 0:
        return False, m["vest"], None
    if "vest" in m.get("ambiguous", ()):
        return unknown("ambiguous_owner")
    if box[3] - box[1] < vcfg["min_person_px"]:
        return unknown("person_small")
    if m.get("torso_in_frame", 1.0) < vcfg.get("min_torso_in_frame", 0.0):
        return unknown("torso_truncated")
    if m.get("torso_cover", 0.0) > vcfg.get("max_torso_overlap", 1.0):
        return unknown("torso_hidden")
    return True, ABSENT_VEST_CONF, None
