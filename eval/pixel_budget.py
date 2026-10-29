"""How many pixels a worker's head gets, and what a smaller detector input keeps.

SH17 YOLOv8s (PyTorch, the config's device) on three frames of each of the 50 SteelBench sample clips
(at 20%, 50% and 80% of its length), at the configured 1280 input and at 640. Head heights are
measured at 1280; the box counts show what the 640 input loses.

python eval/pixel_budget.py                 # writes eval/results/pixel_budget.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultralytics import YOLO  # noqa: E402
from zapdos.detect import available_device  # noqa: E402

PERSON_MIN_CONF = 0.25  # the tracker's first-stage threshold; heads, helmets and vests use detector.min_conf


def frames(video, at=(0.2, 0.5, 0.8)):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * k))
        ok, frame = cap.read()
        if ok:
            yield frame
    cap.release()


def box_heights(model, frame, imgsz, dcfg, device):
    """{class name: [box height in px]} above each class's confidence floor."""
    b = model.predict(frame, imgsz=imgsz, conf=dcfg["conf"], device=device, verbose=False,
                      classes=list(dcfg["classes"].values()))[0].boxes.cpu().numpy()
    cls = b.cls.astype(int)
    out = {}
    for name, c in dcfg["classes"].items():
        keep = (cls == c) & (b.conf >= dcfg["min_conf"].get(name, PERSON_MIN_CONF))
        out[name] = (b.xyxy[keep, 3] - b.xyxy[keep, 1]).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steelbench", default="data/raw/steelbench")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1280, 640])
    ap.add_argument("--out", default="eval/results/pixel_budget.json")
    args = ap.parse_args()
    dcfg = yaml.safe_load((ROOT / args.config).read_text())["detector"]
    device = available_device(dcfg["device"])
    model = YOLO(str(ROOT / dcfg["weights"]))

    imgs = [f for v in sorted((ROOT / args.steelbench / "clips").glob("*.mp4")) for f in frames(v)]
    found = {s: {name: [] for name in dcfg["classes"]} for s in args.sizes}
    for img in imgs:
        for s in args.sizes:
            for name, hs in box_heights(model, img, s, dcfg, device).items():
                found[s][name] += hs

    heads = np.array(found[args.sizes[0]]["head"])
    res = {"frames": len(imgs), "sizes": args.sizes,
           "head_px": {f"p{p}": round(float(np.percentile(heads, p)), 1) for p in (10, 25, 50, 75, 90)},
           "boxes": {str(s): {name: len(hs) for name, hs in found[s].items()} for s in args.sizes}}
    out = ROOT / args.out
    out.write_text(json.dumps(res, indent=2))
    print(f"{len(imgs)} frames; head height at {args.sizes[0]} (px): {res['head_px']}")
    for s in args.sizes:
        print(f"  boxes at {s}: {res['boxes'][str(s)]}")


if __name__ == "__main__":
    main()
