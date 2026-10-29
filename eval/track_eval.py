"""Does one person keep one ID? ByteTrack against the labelled people in labels/.

Every processed frame, the labelled people present (events_eval.present_intervals) are matched
one-to-one to tracks by box overlap (keyframes interpolated; Hungarian on IoU, pairs under 0.3
dropped; label boxes are hand-drawn at 1 s keyframes, so the bar is loose). Reported per person:
share of frames with a track ("tracked"), distinct track IDs, and ID switches (the matched ID changes
from one matched frame to the next; a gap is reported too: switches here are changes between
successive matches, including reacquisition).

"One ID per person" alone does not see one ID being used for two people, so per clip also:
  id_reuse          - track IDs matched to two or more labelled people (an identity merge)
  unmatched_tracks  - track IDs never matched to a labelled person, with their frame counts: false
                      person tracks, or people the labels left out; review before calling them false
  missed_person_frames - labelled person-frames with no matched track
These are diagnostics on approximate boxes, not IDF1/HOTA: that needs complete per-frame identities
and visibility labels, which these labels don't have.

python eval/track_eval.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from events_eval import box_at, iou, is_present  # noqa: E402
from steelbench_clips import extract  # noqa: E402
from zapdos.pipeline import build_models  # noqa: E402

MIN_IOU = 0.3


def score(cache, people):
    return score_full(cache, people)[0]


def score_full(cache, people):
    """(per-person rows, clip diagnostics: id_reuse, unmatched_tracks, missed/total person-frames)."""
    ids = {n: [] for n in people}  # (ts, track id) per matched frame
    frames = dict.fromkeys(people, 0)
    track_frames, matched_ids = {}, set()
    for _, ts, tracks, _ in cache["frames"]:
        for t in tracks:
            track_frames[int(t[4])] = track_frames.get(int(t[4]), 0) + 1
        here = [n for n, p in people.items() if is_present(p, ts)]
        for n in here:
            frames[n] += 1
        if not here or not len(tracks):
            continue
        m = np.array([[iou(box_at(people[n], ts), t[:4]) for t in tracks] for n in here])
        # Maximize valid match count first, then overlap. Ungated assignment can
        # sacrifice a valid pair to an under-threshold pair that is later dropped.
        valid = m >= MIN_IOU
        reward = valid * (min(m.shape) + 1 + m)
        for i, j in zip(*linear_sum_assignment(-reward)):
            if m[i, j] >= MIN_IOU:
                ids[here[i]].append((ts, int(tracks[j, 4])))
                matched_ids.add(int(tracks[j, 4]))
    owners = {}
    for n, seq in ids.items():
        for _, tid in seq:
            owners.setdefault(tid, set()).add(n)
    diag = {"id_reuse": {str(t): sorted(ns) for t, ns in sorted(owners.items()) if len(ns) > 1},
            "unmatched_tracks": {str(t): c for t, c in sorted(track_frames.items()) if t not in matched_ids},
            "person_frames": sum(frames.values()),
            "missed_person_frames": sum(frames.values()) - sum(len(s) for s in ids.values())}
    rows = {}
    for n, seq in ids.items():
        gaps = [round(b[0] - a[0], 1) for a, b in zip(seq, seq[1:]) if a[1] != b[1]]
        rows[n] = {"frames": frames[n], "tracked": round(len(seq) / max(frames[n], 1), 2),
                   "matched_frames": len(seq),
                   "ids": sorted({i for _, i in seq}), "switches": len(gaps), "switch_gaps_s": gaps}
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--results", default="eval/results/tracking.json")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    models = build_models(cfg)

    results, diags = {}, {}
    files = sorted((ROOT / args.labels).glob("*.json"))
    if not files:
        raise SystemExit(f"no label files in {ROOT / args.labels}")
    print(f"{'clip':26} {'person':7} {'tracked':>7} {'IDs':>4} {'switches':>8}  track ids, unseen before each switch")
    for f in files:
        lab = json.loads(f.read_text())
        rows, diag = score_full(extract(ROOT / lab["video"], cfg, models), lab["people"])
        results[f.stem], diags[f.stem] = rows, diag
        for name, r in rows.items():
            print(f"{f.stem:26} {name:7} {r['tracked']:7.0%} {len(r['ids']):4} {r['switches']:8}  {r['ids']} "
                  f"{' '.join(f'{g:g}s' for g in r['switch_gaps_s'])}")
        print(f"{f.stem:26} ID reuse {diag['id_reuse'] or '-'}; unmatched tracks {diag['unmatched_tracks'] or '-'}; "
              f"missed person-frames {diag['missed_person_frames']}/{diag['person_frames']}")
    allr = [r for rows in results.values() for r in rows.values()]
    seen = [r for r in allr if r["ids"]]
    total = {"people": len(allr), "never_tracked": len(allr) - len(seen),
             "one_id": sum(len(r["ids"]) == 1 for r in seen),
             "switches": sum(r["switches"] for r in allr),
             "switches_after_buffer": sum(g > cfg["tracker"]["track_buffer_s"]
                                          for r in allr for g in r["switch_gaps_s"]),
             "tracked": round(sum(r["tracked"] * r["frames"] for r in allr) / sum(r["frames"] for r in allr), 2),
             "reused_ids": sum(len(d["id_reuse"]) for d in diags.values()),
             "unmatched_tracks": sum(len(d["unmatched_tracks"]) for d in diags.values()),
             "missed_person_frames": sum(d["missed_person_frames"] for d in diags.values()),
             "person_frames": sum(d["person_frames"] for d in diags.values())}
    print(f"\n{total['people']} people: {total['one_id']} of {len(seen)} tracked people kept one ID, "
          f"{total['switches']} ID switches ({total['switches_after_buffer']} after the person was unseen longer than "
          f"the {cfg['tracker']['track_buffer_s']:g} s buffer), {total['never_tracked']} never tracked; "
          f"{total['tracked']:.0%} of person-frames tracked; {total['reused_ids']} track IDs used for two or more "
          f"labelled people; {total['unmatched_tracks']} tracks never matched a labelled person")
    Path(ROOT / args.results).write_text(json.dumps({"total": total, "clips": results, "diagnostics": diags}, indent=1))


if __name__ == "__main__":
    main()
