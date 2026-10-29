"""Event-level precision / recall of the pipeline's alerts against the hand labels in labels/.

Each alert is first given to one labelled person: among the people present at the alert time whose
labelled box (keyframes interpolated, grown by 30% each side for box jitter) contains the alert box's
centre, the one whose box overlaps the alert box most (IoU). A person is present inside their
"present" intervals when the label gives them, else from half a keyframe step (0.5 s at the 1 fps
labelling) before their first keyframe to half a step after their last: a box is never extrapolated
beyond the period that was annotated.

An alert can stand for a labelled event of its person, type (and zone) when the alert time falls
inside the labelled interval (+- 1 s). Alerts and labelled events are then paired one-to-one by a
maximum matching (preferring each event's earliest alert), so the order alerts arrive in can't turn
a valid pairing into a duplicate plus a miss. Paired alerts are true. An unpaired alert that could
stand for some labelled event is a duplicate; one whose person has an "ignore" entry for that type
(state not judgeable by eye either) is dropped from the counts; anything else, including an alert
on nobody, is false. Labelled events with no paired alert are missed.
Precision counts duplicates as wrong: TP / (TP + FP + duplicates).

A missing labels folder, or a labelled clip without a completed events.jsonl, is an error rather
than zero alerts (--allow-missing scores it as zero alerts, and says so).

python eval/events_eval.py --run          # run the pipeline on each labelled clip, then score
python eval/events_eval.py                # score existing out/<camera>/events.jsonl
"""
import argparse
import json
import sys
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
TIME_TOL, GROW = 1.0, 0.3
PRESENCE_PAD_S = 0.5  # half the 1 s keyframe step


def present_intervals(person):
    if "present" in person:
        return [tuple(map(float, iv)) for iv in person["present"]]
    keys = [float(k) for k in person["boxes"]]
    return [(min(keys) - PRESENCE_PAD_S, max(keys) + PRESENCE_PAD_S)]


def is_present(person, t):
    return any(a <= t <= b for a, b in present_intervals(person))


def box_at(person, t):
    """The labelled box at time t (keyframes interpolated, held at the ends), or None when absent."""
    if not is_present(person, t):
        return None
    keys = sorted((float(k), v) for k, v in person["boxes"].items())
    ts = [k for k, _ in keys]
    boxes = np.array([v for _, v in keys], float)
    return np.array([np.interp(t, ts, boxes[:, i]) for i in range(4)])


def same_person(pred_box, person, t):
    box = box_at(person, t)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    gw, gh = GROW * (x2 - x1), GROW * (y2 - y1)
    cx, cy = (pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2
    return x1 - gw <= cx <= x2 + gw and y1 - gh <= cy <= y2 + gh


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def whose(pred_box, people, t):
    """The labelled person an alert box belongs to, or None."""
    near = [k for k, person in people.items() if same_person(pred_box, person, t)]
    return max(near, key=lambda k: iou(pred_box, box_at(people[k], t)), default=None)


def hits(entries, p, who, timed=True):
    """Indexes of the labelled entries alert `p`, on person `who`, can stand for."""
    t = p["alert_ts"]
    return [i for i, g in enumerate(entries)
            if g["type"] == p["type"] and g.get("zone") == p.get("zone_id") and g["person"] == who
            and (not timed or g["start"] - TIME_TOL <= t <= g["end"] + TIME_TOL)]


Row = namedtuple("Row", "event_id type zone_id alert_ts track_id verdict person")  # saved as a JSON list
COUNTS = ("labelled", "alerts", "true", "false", "duplicates", "ignored", "missed")


def match(preds, cands, n_events):
    """Maximum one-to-one pairing of alerts to labelled events; ties go to each event's earliest alert.
    cands[a] = event indexes alert a can stand for. Returns {alert index: event index}."""
    if not preds or not n_events or not any(cands):
        return {}
    reward = np.zeros((len(preds), n_events))
    for a, (p, cs) in enumerate(zip(preds, cands)):
        for e in cs:
            reward[a, e] = 1e6 - p["alert_ts"]  # any valid pair outweighs every timing preference
    rows, cols = linear_sum_assignment(-reward)
    return {a: e for a, e in zip(rows, cols) if reward[a, e] > 0}


def _verdict(p, who, labels, cands, paired):
    if paired:
        return "true"
    if cands:
        return "duplicate"
    if hits(labels.get("ignore", []), p, who, timed=False):
        return "ignored"
    return "false"


def _by_type(gts, preds, rows, matched):
    out = {}
    for etype in sorted({g["type"] for g in gts} | {p["type"] for p in preds}):
        gi = [k for k, g in enumerate(gts) if g["type"] == etype]
        verdicts = Counter(r.verdict for r in rows if r.type == etype)
        out[etype] = {"labelled": len(gi), "true": verdicts["true"], "false": verdicts["false"],
                      "duplicates": verdicts["duplicate"], "ignored": verdicts["ignored"],
                      "missed": sum(matched[k] is None for k in gi)}
    return out


def score_clip(labels, preds):
    gts = labels["events"]
    preds = sorted(preds, key=lambda e: e["alert_ts"])
    owners = [whose(p["bbox"], labels["people"], p["alert_ts"]) for p in preds]
    cands = [hits(gts, p, who) for p, who in zip(preds, owners)]
    pairs = match(preds, cands, len(gts))
    matched = [None] * len(gts)
    for a, e in pairs.items():
        matched[e] = preds[a]
    rows = []
    for a, (p, who) in enumerate(zip(preds, owners)):
        verdict = _verdict(p, who, labels, cands[a], a in pairs)
        rows.append(Row(p["event_id"], p["type"], p.get("zone_id"), round(p["alert_ts"], 2), p["track_id"],
                        verdict, who))
    out = Counter(r.verdict for r in rows)
    delays = [m["alert_ts"] - g["start"] for g, m in zip(gts, matched) if m is not None]
    missed = [(g["type"], g.get("zone"), g["person"]) for g, m in zip(gts, matched) if m is None]
    return {"labelled": len(gts), "alerts": len(preds), "true": out["true"], "false": out["false"],
            "duplicates": out["duplicate"], "ignored": out["ignored"], "missed": len(missed),
            "median_delay_s": round(float(np.median(delays)), 2) if delays else None,
            "by_type": _by_type(gts, preds, rows, matched), "alerts_detail": rows, "missed_detail": missed}


def pr(s):
    wrong = s["false"] + s["duplicates"]
    return (round(s["true"] / (s["true"] + wrong), 3) if s["true"] + wrong else None,
            round(s["true"] / s["labelled"], 3) if s["labelled"] else None)


def run_pipeline(label_files, config, out):
    sys.path.insert(0, str(ROOT / "src"))
    from zapdos.pipeline import build_models, run
    cfg = yaml.safe_load((ROOT / config).read_text())
    models = build_models(cfg)
    for f in label_files:
        lab = json.loads(f.read_text())
        run(ROOT / lab["video"], ROOT / lab["zones"], ROOT / out, cfg, f.stem, models=models)


def _print_table(results, total):
    print(f"\n{'clip':<26} {'labelled':>8} {'alerts':>6} {'true':>5} {'false':>5} {'dup':>4} {'ignored':>7} "
          f"{'missed':>6} {'median delay':>12}")
    for clip, s in results.items():
        print(f"{clip:<26} {s['labelled']:>8} {s['alerts']:>6} {s['true']:>5} {s['false']:>5} "
              f"{s['duplicates']:>4} {s['ignored']:>7} {s['missed']:>6} {s['median_delay_s'] or '-':>12}")
    print(f"{'total':<26} {total['labelled']:>8} {total['alerts']:>6} {total['true']:>5} {total['false']:>5} "
          f"{total['duplicates']:>4} {total['ignored']:>7} {total['missed']:>6}")


def _print_by_type(results):
    types = Counter()
    for s in results.values():
        for etype, b in s["by_type"].items():
            for k, v in b.items():
                types[(etype, k)] += v
    for etype in sorted({t for t, _ in types}):
        b = {k: types[(etype, k)] for k in COUNTS if k != "alerts"}
        p_, r_ = pr(b)
        print(f"  {etype:<10} labelled {b['labelled']}  true {b['true']}  false {b['false']}  dup {b['duplicates']}  "
              f"ignored {b['ignored']}  missed {b['missed']}  -> precision {p_}  recall {r_}")


def _print_errors(results):
    for clip, s in results.items():
        for r in s["alerts_detail"]:
            if r.verdict in ("false", "duplicate"):
                print(f"  {clip}: {r.event_id} {r.type} {r.zone_id or ''} t={r.alert_ts} track {r.track_id} "
                      f"on {r.person or 'nobody'} -> {r.verdict}")
        for m in s["missed_detail"]:
            print(f"  {clip}: MISSED {m[0]} {m[1] or ''} person {m[2]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--out", default="out", help="pipeline output root (out/<camera_id>/events.jsonl)")
    ap.add_argument("--run", action="store_true", help="run the pipeline on each labelled clip first")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--results", default="eval/results/events.json")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score a labelled clip with no events.jsonl as zero alerts instead of failing")
    args = ap.parse_args()

    label_dir = ROOT / args.labels
    if not label_dir.is_dir():
        raise SystemExit(f"labels folder not found: {label_dir}")
    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise SystemExit(f"no label files in {label_dir}")
    if args.run:
        run_pipeline(label_files, args.config, args.out)

    results, total, missing = {}, Counter(), []
    for f in label_files:
        preds = load_events(ROOT / args.out / f.stem, args.allow_missing, missing)
        results[f.stem] = s = score_clip(json.loads(f.read_text()), preds)
        for k in COUNTS:
            total[k] += s[k]
    p, r = pr(total)
    _print_table(results, total)
    print(f"\nprecision {p}  recall {r}  (duplicates count against precision)")
    if missing:
        print(f"WARNING: scored as zero alerts because events.jsonl is missing: {', '.join(missing)}")
    _print_by_type(results)
    _print_errors(results)

    out = ROOT / args.results
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": {**total, "precision": p, "recall": r}, "missing_outputs": missing,
                               "clips": results}, indent=2))


def load_events(cam_dir, allow_missing=False, missing=None):
    """The finalised alerts of one completed run. A run still in progress (or one that crashed) has
    events.live.jsonl but no events.jsonl, and is never scored as if it had no alerts."""
    ev_file = cam_dir / "events.jsonl"
    if ev_file.exists():
        return [json.loads(line) for line in ev_file.read_text().splitlines() if line.strip()]
    why = "run did not complete" if (cam_dir / "events.live.jsonl").exists() else "no output"
    if not allow_missing:
        raise SystemExit(f"{ev_file} missing ({why}); run with --run, or --allow-missing to score it as zero alerts")
    if missing is not None:
        missing.append(cam_dir.name)
    return []


if __name__ == "__main__":
    main()
