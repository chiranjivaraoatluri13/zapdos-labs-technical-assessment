"""Why was a labelled event missed? Frame-by-frame attribution from a run's readings.jsonl.

For each missed labelled event (or every labelled event with --all), each processed frame inside its
labelled interval where the person is present is put in the first stage that lost it:
  no_track           - no tracked person box overlaps the labelled box (IoU >= 0.3): detector or tracker
  unconfirmed        - a track, but not yet a confirmed person (tracker.confirm)
  unknown:<reason>   - the reading was "can't tell": edge_cut, ambiguous_owner, no_head_box (estimated
                       head), head_small (min_head_px), classifier_uncertain, person_small, torso_hidden,
                       torso_truncated, feet_unknown, ...
  compliant          - a decided reading that said no violation (for no_helmet: classifier p(bare) below
                       the threshold; the p(bare) values are listed)
  violating          - the reading was a violation
An event with violating frames but no alert was lost by the temporal rule (persistence, vote, evidence
gap); the longest violating run is shown against min_duration_s.

Needs a run with readings logged:
python run.py --video clips/x.mp4 --zones zones/x.json --out runs/review --log-readings
python eval/miss_review.py --labels labels_heldout --out runs/review [--type no_helmet] [--all]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from events_eval import box_at, iou, load_events, score_clip  # noqa: E402

MIN_IOU = 0.3


def load_readings(path):
    if not path.exists():
        raise SystemExit(f"{path} missing: rerun the clip with --log-readings")
    frames = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows = frames.setdefault(r["ts"], [])
        if r["track_id"] is not None:
            rows.append(r)
    return frames


def stage(rows, gt_box, key):
    """(stage, row) for the labelled person in one frame."""
    best = max(rows, key=lambda r: iou(gt_box, r["box"]), default=None)
    if best is None or iou(gt_box, best["box"]) < MIN_IOU:
        return "no_track", None
    if not best["confirmed"]:
        return "unconfirmed", best
    value, _, why = best["readings"].get(key, [None, 0, "not_read"])
    if value is None:
        return f"unknown:{why}", best
    return ("violating" if value else "compliant"), best


def review(lab, readings, ev, min_duration):
    key = ev["type"] if ev["type"] != "zone_entry" else f"zone_entry:{ev['zone']}"
    person = lab["people"][ev["person"]]
    counts, p_bare, heads, runs, cur, last = Counter(), [], [], [], 0, None
    for ts in sorted(readings):
        if not ev["start"] <= ts <= ev["end"]:
            continue
        gt = box_at(person, ts)
        if gt is None:
            continue
        s, row = stage(readings[ts], gt, key)
        counts[s] += 1
        if row is not None and row.get("p_bare") is not None:
            p_bare.append(row["p_bare"])
        if row is not None:
            heads.append(row["head"][3] - row["head"][1])
        cur = cur + 1 if s == "violating" else 0
        runs.append(cur)
        last = ts
    dt = np.median(np.diff(sorted(readings))) if len(readings) > 1 else 0.1
    longest = max(runs, default=0) * dt
    if not counts:
        verdict = "person never present in the labelled interval"
    elif counts["violating"] and longest + 1e-6 < min_duration:
        verdict = f"temporal rule: longest violating run {longest:.1f}s < min_duration_s {min_duration}"
    elif counts["violating"]:
        verdict = "temporal rule: vote/evidence gap (violating runs long enough, but no alert)"
    else:
        verdict = f"perception: {counts.most_common(1)[0][0]}"
    return {"type": ev["type"], "person": ev["person"], "frames": sum(counts.values()), "stages": dict(counts),
            "p_bare": [round(p, 3) for p in p_bare], "head_px_median": round(float(np.median(heads)), 1) if heads else None,
            "longest_violating_run_s": round(longest, 2), "last_ts": last, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=["labels", "labels_heldout"])
    ap.add_argument("--out", required=True, help="run output root with <clip>/readings.jsonl and events.jsonl")
    ap.add_argument("--type", help="only this event type (e.g. no_helmet)")
    ap.add_argument("--all", action="store_true", help="review caught events too")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--results", default="eval/results/miss_review.json")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())["events"]
    out, results = ROOT / args.out, {}
    for f in [f for d in args.labels for f in sorted((ROOT / d).glob("*.json"))]:
        lab = json.loads(f.read_text())
        if not (out / f.stem).exists():
            continue
        s = score_clip(lab, load_events(out / f.stem))
        missed = {(t, z, p) for t, z, p in s["missed_detail"]}
        readings = load_readings(out / f.stem / "readings.jsonl")
        for ev in lab["events"]:
            if args.type and ev["type"] != args.type:
                continue
            was_missed = (ev["type"], ev.get("zone"), ev["person"]) in missed
            if not (was_missed or args.all):
                continue
            p = {**cfg["default"], **cfg.get(ev["type"], {})}
            r = review(lab, readings, ev, p["min_duration_s"])
            r["missed"] = was_missed
            if not was_missed:
                r["verdict"] = "caught"
            if ev["type"] != "no_helmet":
                r.pop("p_bare")
            results.setdefault(f.stem, []).append(r)
            print(f"{f.stem:24} {'MISSED' if was_missed else 'caught':6} {ev['type']:<10} {ev['person']:<3} "
                  f"{r['frames']:>3} frames {r['stages']}  head px {r['head_px_median']}  -> {r['verdict']}")
            if r.get("p_bare"):
                print(f"{'':31} p(bare) {r['p_bare']}")
    path = ROOT / args.results
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
