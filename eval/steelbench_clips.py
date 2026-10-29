"""Alert-level evaluation on the 50 SteelBench sample clips against their per-person PPE labels.

SteelBench labels every worker in a clip (when there are <= 5) with helmet and high-vis-vest state
(worn / not_worn / cannot_determine) but gives no boxes or times, so alerts can't be matched to
people one by one. What can be scored exactly:
  - clip level: a clip with >= 1 not_worn worker should get >= 1 alert of that type (recall); a clip
    where every labelled worker is "worn" should get none (any alert there is a certain false alarm)
  - count level: alerts beyond (not_worn + cannot_determine) workers in a clip are certain false
    alarms (duplicates or alerts on compliant people); min(alerts, not_worn) is an optimistic count of
    correct alerts, so person recall from it is an upper bound

Perception (detect, track, measure) is cached per clip; the rules (decide, events) are replayed from
the cache, so thresholds and event settings can be swept in seconds:

python eval/steelbench_clips.py                                  # current config
python eval/steelbench_clips.py --sweep ppe.classifier.threshold=0.3,0.5,0.7,0.78
python eval/steelbench_clips.py --set ppe.classifier.require_head_det=false --set events.default.threshold=0.6
"""
import argparse
import copy
import hashlib
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.detect import detector_model_path, model_fingerprint  # noqa: E402
from zapdos.events import EventEngine  # noqa: E402
from zapdos.pipeline import build_models, observe, open_video, sample_frames  # noqa: E402
from zapdos.ppe import decide, measure  # noqa: E402
from zapdos.track import PersonConfirmation, PersonTracker  # noqa: E402
from zapdos.zones import ZoneSet  # noqa: E402

TYPES = {"no_helmet": "helmet", "no_vest": "high_vis_vest"}


def camera_of(clip):
    """SteelBench clip name -> camera name (the part before the recording date)."""
    return re.sub(r"_\d{8}_\d{6}_\d{4}$", "", clip.removeprefix("clip_"))


def cameras_of(labels_dir):
    """Cameras of the hand-labelled clips in labels_dir (their videos link to SteelBench clips)."""
    return {camera_of((ROOT / json.loads(f.read_text())["video"]).resolve().stem)
            for f in (ROOT / labels_dir).glob("*.json")}


def load_labels(root):
    labels = {}
    for f in sorted((root / "annotations").glob("*.json")):
        a = json.loads(f.read_text())
        persons = a.get("persons") or []
        labels[f.stem] = {t: Counter(p["ppe"].get(k) for p in persons) for t, k in TYPES.items()}
        labels[f.stem]["conditions"] = a.get("visibility_conditions") or []
    return labels


def set_key(cfg, dotted, value):
    *path, last = dotted.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(value)


def perception_key(cfg):
    """Cache key: the settings that detection, tracking and measure() depend on, and the classifier
    files' times and perception source hashes. Decision-only thresholds are left
    out; classifier eligibility gates are included because they affect inference."""
    p = cfg["ppe"]
    w = ROOT / p["classifier"]["weights"]
    mtimes = [f.stat().st_mtime if f.exists() else 0 for f in (w, w.with_suffix(".onnx"))]
    keep = {"detector": cfg["detector"], "tracker": cfg["tracker"], "sampling": cfg["sampling"],
            "ppe": {k: p.get(k) for k in ("head_region", "helmet_expand", "helmet_overlap", "min_head_px",
                                           "edge_margin_px")},
            "vest": {k: p["vest"][k] for k in ("enabled", "torso")}, "clf": [p["helmet_method"], str(w), *mtimes]}
    keep["classifier_gates"] = {k: p.get("classifier", {}).get(k) for k in ("min_head_px", "require_head_det")}
    keep["ambiguity_margin"] = p.get("ambiguity_margin")
    model = ROOT / detector_model_path(cfg["detector"])
    keep["detector_model"] = [str(model.relative_to(ROOT)), model_fingerprint(model)]
    keep["code"] = {name: hashlib.sha256((ROOT / "src/zapdos" / name).read_bytes()).hexdigest()
                    for name in ("detect.py", "track.py", "ppe.py", "helmet.py")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def extract(video, cfg, models):
    cap, src_fps, size = open_video(video)
    fps = min(cfg["sampling"]["process_fps"], src_fps)
    detector, helmet_clf = models
    tracker = PersonTracker(cfg["tracker"], fps)
    frames = []
    for frame_idx, ts, frame, _ in sample_frames(cap, src_fps, fps):
        det = detector(frame)
        tracks = tracker(det.persons)
        frames.append((frame_idx, ts, tracks, measure(tracks[:, :4], det, cfg["ppe"], frame, helmet_clf)))
    cap.release()
    return {"size": size, "fps": fps, "frames": frames}


def replay(clip, cache, cfg, thr):
    zones = ZoneSet(None, cache["size"], cfg["zones"], cache["fps"])
    engine = EventEngine(cfg["events"], clip, cache["fps"])
    confirm = PersonConfirmation(cfg["tracker"]["confirm"])
    for frame_idx, ts, tracks, meas in cache["frames"]:
        readings = decide(meas, tracks[:, :4], cfg["ppe"], thr)
        observations, _ = observe(tracks, meas, readings, zones, ts, confirm)  # no frames: no appearance veto
        engine.step(frame_idx, ts, observations)
    return engine.events


def _tally(s, n, lab, clip_s):
    """Add one clip with `n` alerts of a type to the counts `s`; `lab` is its per-person label counts."""
    s["alerts"] += n
    if not lab:  # no per-person labels (crowded clip or no annotation)
        s["alerts_unlabelled"] += n
        s["s_unlabelled"] += clip_s
        return
    bad, unk = lab["not_worn"], lab["cannot_determine"]
    if bad:
        s["pos_clips"] += 1
        s["tp_clips"] += n > 0
        s["violators"] += bad
        s["matched"] += min(n, bad)
    elif not unk:
        s["neg_clips"] += 1
        s["fp_clips"] += n > 0
    s["certain_fp"] += max(0, n - bad - unk)
    s["alerts_labelled"] += n


def score(events_by_clip, labels, video_s):
    out = {}
    for etype in TYPES:
        s, delays = Counter(), []
        for clip, events in events_by_clip.items():
            of_type = [e for e in events if e.type == etype]
            delays += [e.delay_s for e in of_type]
            _tally(s, len(of_type), labels.get(clip, {}).get(etype), video_s[clip])
        out[etype] = {
            **{k: int(v) for k, v in s.items() if not k.startswith("s_")},
            "clip_recall": _r(s["tp_clips"], s["pos_clips"]),
            "clip_fp_rate": _r(s["fp_clips"], s["neg_clips"]),
            "person_recall_ub": _r(s["matched"], s["violators"]),
            "precision_ub": _r(s["matched"], s["alerts_labelled"]),
            "unlabelled_alerts_per_min": _r(s["alerts_unlabelled"], s["s_unlabelled"] / 60),
            "median_delay_s": round(float(np.median(delays)), 2) if delays else None,
        }
    return out


def _r(a, b):
    return round(a / b, 3) if b else None


def load_perception(videos, cfg, cache_root):
    """Per-clip perception from the cache, extracting the clips that aren't cached yet."""
    cache_dir = cache_root / perception_key(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    caches, models = {}, None
    t0 = time.perf_counter()
    for v in videos:
        f = cache_dir / f"{v.stem}.pkl"
        if not f.exists():
            models = models or build_models(cfg)
            print(f"extract {v.stem}", flush=True)
            f.write_bytes(pickle.dumps(extract(v, cfg, models)))
        caches[v.stem] = pickle.loads(f.read_bytes())
    print(f"perception for {len(caches)} clips ready in {time.perf_counter() - t0:.0f} s ({cache_dir.name})")
    return caches


def sweep_settings(cfg, sweep):
    """[(name, config)]: the config alone, or one copy per value of the swept key."""
    if not sweep:
        return [(None, cfg)]
    key, values = sweep.split("=", 1)
    settings = []
    for val in values.split(","):
        c = copy.deepcopy(cfg)
        set_key(c, key, val)
        settings.append((f"{key}={val}", c))
    return settings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steelbench", default="data/raw/steelbench")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    ap.add_argument("--sweep", metavar="KEY=V1,V2,...", help="replay once per value of one dotted key")
    ap.add_argument("--cache", default="data/cache/steelbench_obs")
    ap.add_argument("--out", default="eval/results/steelbench_clips.json")
    ap.add_argument("--per-clip", action="store_true", help="print alerts per clip next to the labels")
    ap.add_argument("--by-condition", action="store_true", help="also score each SteelBench visibility condition")
    ap.add_argument("--skip-cameras-of", metavar="LABELS_DIR",
                    help="leave out every clip from the cameras of these labelled clips (e.g. labels_heldout)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    for kv in args.set:
        set_key(cfg, *kv.split("=", 1))
    root = ROOT / args.steelbench
    labels = load_labels(root)
    skip = cameras_of(args.skip_cameras_of) if args.skip_cameras_of else set()
    videos = [v for v in sorted((root / "clips").glob("*.mp4")) if camera_of(v.stem) not in skip]
    caches = load_perception(videos, cfg, ROOT / args.cache)
    video_s = {c: len(x["frames"]) / x["fps"] for c, x in caches.items()}

    thr0 = cfg["ppe"]["classifier"].get("threshold") or json.loads(
        (ROOT / cfg["ppe"]["classifier"]["weights"]).with_suffix(".json").read_text())["threshold"]
    results = []
    for name, c in sweep_settings(cfg, args.sweep):
        thr = c["ppe"]["classifier"].get("threshold") or thr0
        events = {clip: replay(clip, cache, c, thr) for clip, cache in caches.items()}
        res = score(events, labels, video_s)
        results.append({"setting": name, "overrides": args.set, "helmet_threshold": thr, **res})
        _print(name or "config", res)
        if args.per_clip:
            _per_clip(events, labels)
        if args.by_condition:
            results[-1]["by_condition"] = _by_condition(events, labels, video_s)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results if args.sweep else results[0], indent=2))


def _print(name, res):
    print(f"\n{name}")
    for etype, r in res.items():
        print(f"  {etype:<9} clips: recall {r['tp_clips']}/{r['pos_clips']}  false-alarm clips {r['fp_clips']}/"
              f"{r['neg_clips']} | alerts {r['alerts']}: certain FP {r['certain_fp']}, "
              f"person recall <= {r['person_recall_ub']}, precision <= {r['precision_ub']} | "
              f"unlabelled {r['unlabelled_alerts_per_min']}/min | median delay {r['median_delay_s']} s")


def _by_condition(events, labels, video_s):
    """The same scores on the clips SteelBench tags with each visibility condition (a clip can have several)."""
    conds = sorted({c for lab in labels.values() for c in lab["conditions"]})
    out = {}
    print(f"\n  {'condition':<10} {'clips':>5} {'min':>5} {'helmet caught':>13} {'helmet FA':>9} {'vest caught':>11} "
          f"{'vest FA':>7} {'certain FP':>10} {'alerts/min':>10}")
    for cond in conds:
        clips = [c for c in events if cond in labels.get(c, {}).get("conditions", [])]
        res = score({c: events[c] for c in clips}, labels, video_s)
        h, v = (Counter(res[t]) for t in ("no_helmet", "no_vest"))  # Counter: absent counts read as 0
        mins = sum(video_s[c] for c in clips) / 60
        row = {"clips": len(clips), "minutes": round(mins, 2), "no_helmet": res["no_helmet"], "no_vest": res["no_vest"],
               "alerts_per_min": _r(h["alerts"] + v["alerts"], mins)}
        out[cond] = row
        print(f"  {cond:<10} {len(clips):>5} {mins:>5.1f} {h['tp_clips']:>9}/{h['pos_clips']:<3} "
              f"{h['fp_clips']:>5}/{h['neg_clips']:<3} {v['tp_clips']:>7}/{v['pos_clips']:<3} "
              f"{v['fp_clips']:>3}/{v['neg_clips']:<3} "
              f"{h['certain_fp'] + v['certain_fp']:>10} {row['alerts_per_min']:>10}")
    return out


def _per_clip(events, labels):
    for clip, evs in sorted(events.items()):
        lab = labels.get(clip, {})
        got = Counter(e.type for e in evs)
        row = "  ".join(f"{t}: {got[t]} vs {dict(lab[t]) if lab.get(t) else '-'}" for t in TYPES)
        print(f"    {clip[-34:]:<34} {row}")


if __name__ == "__main__":
    main()
