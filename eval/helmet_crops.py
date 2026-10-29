"""Helmet-state benchmark on SteelBench person crops with weak labels.

Clips where every annotated worker has the same SteelBench helmet label give a label to every person
detected in them. Three helmet decisions are compared: SH17's helmet-over-head rule on the full frame,
the same rule on an upscaled crop of each person, and the head-crop classifier. For each: how often it
says "bare head" on helmeted workers (the false-alarm rate that sets alert precision) and on
bare-headed workers (recall).

python eval/helmet_crops.py                 # writes eval/results/helmet_crops.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultralytics import YOLO  # noqa: E402
from zapdos.detect import Detections  # noqa: E402
from zapdos.helmet import HelmetClassifier, head_from_person  # noqa: E402
from zapdos.ppe import head_candidates, helmet_threshold, read_ppe  # noqa: E402

PAD = (0.15, 0.15, 0.15, 0.05)  # person crop padding: left, top, right, bottom, as a share of the box


def unanimous_clips(root):
    out = {}
    for f in sorted((root / "annotations").glob("*.json")):
        a = json.loads(f.read_text())
        labels = {p.get("ppe", {}).get("helmet") for p in a.get("persons") or []}
        if len(labels) == 1 and labels <= {"worn", "not_worn"}:
            out[f.stem] = (labels.pop(), a.get("visibility_conditions") or [])
    return out


def split(result, cls):
    b = result.boxes.cpu().numpy()
    c = b.cls.astype(int)

    def pick(k, min_conf):
        m = (c == cls[k]) & (b.conf >= min_conf)
        return np.hstack([b.xyxy[m], b.conf[m, None]])
    return pick


def sh17_rule_bare(person_boxes, pick, ppe_cfg, min_conf):
    """SH17's helmet-over-head rule per person: True bare, False helmet, None can't tell."""
    det = Detections(None, pick("head", min_conf["head"]), pick("helmet", min_conf["helmet"]),
                     pick("vest", min_conf["vest"]))
    return [r["no_helmet"][0] for r in read_ppe(person_boxes, det, {**ppe_cfg, "helmet_method": "detector"})[0]]


def sampled_frames(video, every_s):
    """(t, frame) every `every_s` seconds of the video."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps * every_s)))
    idx = 0
    while cap.grab():
        if idx % step == 0:
            _, frame = cap.retrieve()
            yield round(idx / fps, 2), frame
        idx += 1
    cap.release()


def person_crops(frame, persons):
    """Each person box, padded, cut from the native frame; and the box in crop coordinates."""
    h, w = frame.shape[:2]
    crops, local = [], []
    for x1, y1, x2, y2 in persons:
        pw, ph = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - PAD[0] * pw)), int(max(0, y1 - PAD[1] * ph))
        cx2, cy2 = int(min(w, x2 + PAD[2] * pw)), int(min(h, y2 + PAD[3] * ph))
        crops.append(frame[cy1:cy2, cx1:cx2])
        local.append(np.array([[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]]))
    return crops, local


class Bench:
    """Runs every method on each sampled frame and tallies their calls per label and condition."""

    def __init__(self, args, cfg):
        dcfg, self.pcfg = cfg["detector"], cfg["ppe"]
        self.args = args
        self.cls, self.min_conf = dcfg["classes"], dcfg["min_conf"]
        self.imgsz, self.conf, self.dev = dcfg["imgsz"], dcfg["conf"], dcfg["device"]
        self.model = YOLO(str(ROOT / dcfg["weights"]))
        w = ROOT / args.helmet_weights
        self.clf = HelmetClassifier(w, self.dev) if w.exists() else None
        self.thr = helmet_threshold(self.pcfg, self.clf)
        self.full = f"full@{self.imgsz}"
        self.methods = [self.full] + [f"crop@{s}" for s in args.crop_sizes] + (["classifier"] if self.clf else [])
        self.head_source = defaultdict(int)
        self.tally = {m: defaultdict(lambda: defaultdict(int)) for m in self.methods}  # method -> label -> call -> n
        self.by_cond = {m: defaultdict(lambda: [0, 0]) for m in self.methods}  # method -> condition -> [bare, n], worn
        self.ms = defaultdict(list)
        self.samples = []
        self.model.predict(np.zeros((640, 640, 3), np.uint8), device=self.dev, verbose=False)

    def _predict(self, imgs, imgsz):
        return self.model.predict(imgs, imgsz=imgsz, conf=self.conf, device=self.dev, verbose=False,
                                  classes=list(self.cls.values()))

    def add_frame(self, frame, stem, t, label, conds):
        t0 = time.perf_counter()
        r = self._predict(frame, self.imgsz)[0]
        self.ms[self.full].append((time.perf_counter() - t0) * 1000)
        b = r.boxes.cpu().numpy()
        keep = ((b.cls.astype(int) == self.cls["person"]) & (b.conf >= self.args.min_person_conf)
                & (b.xyxy[:, 3] - b.xyxy[:, 1] >= self.args.min_person_px))
        persons = b.xyxy[keep]
        if not len(persons):
            return
        pick = split(r, self.cls)
        calls = {self.full: sh17_rule_bare(persons, pick, self.pcfg, self.min_conf), **self._crop_calls(frame, persons)}
        if self.clf:
            calls["classifier"], p = self._classifier_calls(frame, persons, pick)
        for i, box in enumerate(persons):
            outs = {m: calls[m][i] for m in self.methods}
            if self.clf:
                outs["p_bare"] = round(float(p[i]), 3)
            self._count(outs, label, conds)
            self.samples.append({"clip": stem, "t": t, "label": label, "box": [round(float(v)) for v in box], **outs})

    def _crop_calls(self, frame, persons):
        crops, local = person_crops(frame, persons)
        out = {}
        for s in self.args.crop_sizes:
            t0 = time.perf_counter()
            rs = self._predict(crops, s)
            self.ms[f"crop@{s}"].append((time.perf_counter() - t0) * 1000 / len(crops))
            out[f"crop@{s}"] = [sh17_rule_bare(lb, split(rr, self.cls), self.pcfg, self.min_conf)[0]
                                for rr, lb in zip(rs, local)]
        return out

    def _classifier_calls(self, frame, persons, pick):
        """The classifier's call per person (None below min_head_px) and its p(bare)."""
        heads, helmets = head_candidates(pick("head", self.min_conf["head"]), pick("helmet", self.min_conf["helmet"]),
                                         persons, self.pcfg["head_region"])
        head_boxes = []
        for i, box in enumerate(persons):
            cands = heads.get(i, []) + helmets.get(i, [])
            self.head_source["detected" if cands else "estimated"] += 1
            head_boxes.append(max(cands, key=lambda h: h[4])[:4] if cands else head_from_person(box))
        t0 = time.perf_counter()
        p = self.clf.p_bare(frame, head_boxes)
        self.ms["classifier"].append((time.perf_counter() - t0) * 1000 / len(persons))
        return [None if hb[3] - hb[1] < self.args.min_head_px else bool(pi >= self.thr)
                for hb, pi in zip(head_boxes, p)], p

    def _count(self, outs, label, conds):
        for m in self.methods:
            v = outs[m]
            o = "bare" if v else ("helmet" if v is False else "unknown")
            self.tally[m][label][o] += 1
            if label == "worn":
                for c in conds:
                    self.by_cond[m][c][0] += o == "bare"
                    self.by_cond[m][c][1] += 1

    def report(self, n_clips):
        report = {"clips": n_clips, "persons": len(self.samples), "methods": {}}
        n_worn = sum(s["label"] == "worn" for s in self.samples)
        n_bare = sum(s["label"] == "not_worn" for s in self.samples)
        print(f"{n_clips} clips, {len(self.samples)} person-crops ({n_worn} helmet, {n_bare} bare)\n")
        print(f"{'method':<11} {'FALSE bare on helmeted':>23} {'unknown(helmeted)':>18} {'recall on bare':>15} "
              f"{'unknown(bare)':>14} {'ms/person':>10}")
        for m in self.methods:
            w, nw = self.tally[m]["worn"], self.tally[m]["not_worn"]
            nwn, nnw = sum(w.values()) or 1, sum(nw.values()) or 1
            row = {"fp_rate_on_helmeted": w["bare"] / nwn, "unknown_on_helmeted": w["unknown"] / nwn,
                   "recall_on_bare": nw["bare"] / nnw, "unknown_on_bare": nw["unknown"] / nnw,
                   "ms_per_person" if m != self.full else "ms_per_frame": float(np.mean(self.ms[m])),
                   "fp_rate_by_condition": {c: v[0] / v[1] for c, v in self.by_cond[m].items() if v[1]}}
            report["methods"][m] = row
            print(f"{m:<11} {row['fp_rate_on_helmeted']:>22.1%} {row['unknown_on_helmeted']:>18.1%} "
                  f"{row['recall_on_bare']:>15.1%} {row['unknown_on_bare']:>14.1%} {np.mean(self.ms[m]):>9.1f}"
                  + (" (per frame)" if m == self.full else ""))
        if self.clf:
            print(f"classifier head box: {dict(self.head_source)} "
                  "(estimated = no head/helmet box, placed from the person box)")
        print("\nfalse-bare rate on helmeted workers by condition:")
        for m in self.methods:
            by_cond_m = sorted(report["methods"][m]["fp_rate_by_condition"].items())
            print(f"  {m:<11} " + "  ".join(f"{c} {v:.0%}" for c, v in by_cond_m))
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steelbench", default="data/raw/steelbench")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--min-person-conf", type=float, default=0.5)
    ap.add_argument("--min-person-px", type=int, default=60)
    ap.add_argument("--crop-sizes", type=int, nargs="+", default=[320])
    ap.add_argument("--helmet-weights", help="default: the config's classifier")
    ap.add_argument("--min-head-px", type=float, help="classifier can't tell below this head height (default: config)")
    ap.add_argument("--out", default="eval/results/helmet_crops.json")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    ccfg = cfg["ppe"]["classifier"]
    args.helmet_weights = args.helmet_weights or ccfg["weights"]
    args.min_head_px = ccfg["min_head_px"] if args.min_head_px is None else args.min_head_px
    root = ROOT / args.steelbench
    clips = unanimous_clips(root)
    bench = Bench(args, cfg)
    for stem, (label, conds) in clips.items():
        for t, frame in sampled_frames(root / "clips" / f"{stem}.mp4", args.every_s):
            bench.add_frame(frame, stem, t, label, conds)
    report = bench.report(len(clips))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report, "samples": bench.samples}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
