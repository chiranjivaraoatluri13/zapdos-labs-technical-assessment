"""Does the camera-moved check (src/zapdos/view.py) stay quiet on still cameras and trip on moved ones?

Real footage: the 50 SteelBench sample clips (22 cameras; 12 of them with 2-6 clips taken minutes to
weeks apart). Each check matches one frame per second to a reference frame:
- still: every clip against its own first frame (must not trip);
- same camera, another clip: against the first frame of another clip of the same camera, as when the
  zones were drawn on an earlier day (lighting and people differ; the camera should not have moved);
- another camera: each camera's first clip against one clip of every other camera (must trip);
- synthetic moves: every clip against its own first frame with the later frames shifted, rotated or
  zoomed about the centre by a set amount.
A clip trips when the check reads moved for persist_s in a row, as in the pipeline. With no zones on
these clips, the measured points are the corners of the central half of the frame.

python eval/view_eval.py [--set view.max_shift=0.03]
"""
import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from steelbench_clips import camera_of  # noqa: E402
from zapdos.pipeline import open_video, sample_frames  # noqa: E402
from zapdos.view import ViewCheck  # noqa: E402

MOVES = {"shift 1%": ("shift", 0.01), "shift 2%": ("shift", 0.02), "shift 3%": ("shift", 0.03),
         "shift 5%": ("shift", 0.05), "rotate 1°": ("rotate", 1), "rotate 3°": ("rotate", 3),
         "rotate 6°": ("rotate", 6), "zoom 1.05×": ("zoom", 1.05), "zoom 1.10×": ("zoom", 1.10),
         "zoom 1.20×": ("zoom", 1.20)}


def warp(img, kind, amount):
    h, w = img.shape[:2]
    if kind == "shift":
        m = np.float32([[1, 0, amount * w], [0, 1, 0]])
    else:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), amount if kind == "rotate" else 0,
                                    1.0 if kind == "rotate" else amount)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def trips(shifts, cfg):
    """True when the shift stays over max_shift for persist_s of consecutive checks."""
    need = math.ceil(cfg["persist_s"] / cfg["check_every_s"] - 1e-6) + 1
    run = 0
    for s in shifts:
        run = run + 1 if s > cfg["max_shift"] else 0
        if run >= need:
            return True
    return False


def central_quad(size):
    """Corners of the central half of the frame: the measured points when a clip has no zones."""
    w, h = size
    return [[w / 4, h / 4], [3 * w / 4, h / 4], [3 * w / 4, 3 * h / 4], [w / 4, 3 * h / 4]]


def check_clip(clip, cfg):
    """One clip at one frame per check: (features per frame, still shifts, {move: tripped})."""
    cap, src_fps, size = open_video(clip)
    w, h = size
    vc = ViewCheck(cfg, central_quad(size), size)
    frames = [f for _, _, f, _ in sample_frames(cap, src_fps, 1 / cfg["check_every_s"])]
    cap.release()
    small = [cv2.resize(f, (cfg["width"], round(h * cfg["width"] / w)), interpolation=cv2.INTER_AREA)
             for f in frames]
    feats = [vc.features(f) for f in small]
    vc.ref = feats[0]
    still = [vc.measure(f) for f in feats[1:]]
    moved = {name: trips([vc.measure(vc.features(warp(f, kind, amount))) for f in small[1:]], cfg)
             for name, (kind, amount) in MOVES.items()}
    return feats, still, moved, size


def finite_max(d):
    vals = [s for v in d.values() for s in v if math.isfinite(s)]
    return round(max(vals) * 100, 2) if vals else None


def pct(shifts):
    return [round(s * 100, 2) if math.isfinite(s) else None for s in shifts]


def tripped(cases, cfg):
    return [k for k, s in cases.items() if trips(s, cfg)]


def still_report(still, cfg):
    return {"clips": len(still), "tripped": tripped(still, cfg), "max_shift_pct": finite_max(still),
            "unmatched_checks": sum(math.isinf(s) for v in still.values() for s in v),
            "checks": sum(len(v) for v in still.values())}


def report(cfg, still, pairs, synth):
    """The summary written to view_check.json; `pairs` = (same-camera shifts, other-camera shifts)."""
    same, other = pairs
    missed_other = [p for p, s in other.items() if not trips(s, cfg)]
    return {
        "config": cfg, "clips": len(still), "cameras": len({camera_of(c) for c in still}),
        "still": still_report(still, cfg),
        "same_camera": {"pairs": len(same), "tripped": tripped(same, cfg), "max_shift_pct": finite_max(same)},
        "other_camera": {"pairs": len(other), "tripped": len(other) - len(missed_other),
                         "not_tripped": missed_other},
        "synthetic": {k: {"tripped": sum(v.values()), "clips": len(v),
                          "missed": [c for c, t in v.items() if not t]} for k, v in synth.items()},
        "per_clip_still_shift_pct": {c: pct(v) for c, v in still.items()},
        "same_camera_shift_pct": {p: pct(v) for p, v in same.items()},
    }


def print_report(res, cfg):
    print(f"\n{res['clips']} clips, {res['cameras']} cameras; moved = shift > {cfg['max_shift'] * 100:g}% of width "
          f"for {cfg['persist_s']:g} s")
    s = res["still"]
    print(f"still:         {len(s['tripped'])}/{s['clips']} clips tripped; max shift {s['max_shift_pct']}%; "
          f"{s['unmatched_checks']}/{s['checks']} checks unmatched")
    s = res["same_camera"]
    print(f"same camera:   {len(s['tripped'])}/{s['pairs']} clip pairs tripped; max finite shift {s['max_shift_pct']}%")
    for p in s["tripped"]:
        print(f"    {p}: {res['same_camera_shift_pct'][p]}")
    s = res["other_camera"]
    print(f"other camera:  {s['tripped']}/{s['pairs']} pairs tripped")
    for p in s["not_tripped"]:
        print(f"    missed {p}")
    for k, v in res["synthetic"].items():
        print(f"{k:<14} {v['tripped']}/{v['clips']} clips tripped")


def check_all(clip_dir, cfg):
    """check_clip on every clip: ({clip: features}, {clip: still shifts}, {move: {clip: tripped}}, size)."""
    feats, still, synth, size = {}, {}, defaultdict(dict), None
    for clip in sorted(clip_dir.glob("*.mp4")):
        feats[clip.stem], still[clip.stem], moved, size = check_clip(clip, cfg)
        for name, was_tripped in moved.items():
            synth[name][clip.stem] = was_tripped
        print(f"{clip.stem}: {len(feats[clip.stem])} checks, still max shift "
              f"{max(still[clip.stem]) * 100:.2f}%", flush=True)
    return feats, still, synth, size


def clip_pairs(feats, cfg, size):
    """Shifts of every ordered pair of same-camera clips, and of every pair of cameras' first clips."""
    def against(ref_clip, clip):  # all sample clips are 1080p, so one quad fits every pair
        vc = ViewCheck(cfg, central_quad(size), size)
        vc.ref = feats[ref_clip][0]
        return [vc.measure(f) for f in feats[clip]]

    by_cam = defaultdict(list)
    for c in feats:
        by_cam[camera_of(c)].append(c)
    same = {f"{a} -> {b}": against(a, b) for cams in by_cam.values() for a, b in itertools.permutations(cams, 2)}
    firsts = [cams[0] for cams in by_cam.values()]
    other = {f"{a} -> {b}": against(a, b) for a, b in itertools.permutations(firsts, 2)}
    return same, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/raw/steelbench/clips")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="view.KEY=VALUE")
    ap.add_argument("--out", default="eval/results/view_check.json")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())["view"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k.removeprefix("view.")] = yaml.safe_load(v)

    feats, still, synth, size = check_all(ROOT / args.clips, cfg)
    res = report(cfg, still, clip_pairs(feats, cfg, size), synth)
    (ROOT / args.out).write_text(json.dumps(res, indent=2))
    print_report(res, cfg)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
