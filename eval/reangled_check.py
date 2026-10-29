"""End to end: a camera re-angled after its zones were drawn.

Every frame of a labelled clip is rolled about its centre and panned (default 2 degrees and 5% of the
width), as if the camera had been knocked after its zones were drawn. The pipeline then runs three
times with the clip's original zone file: on the original clip, on the re-angled copy without the
camera-moved check, and on the re-angled copy with it. Reported per run: alerts by type, the zone
alerts, any camera_moved event, and throughput.

python eval/reangled_check.py              # writes eval/results/reangled_check.json (copy in runs/)
"""
import argparse
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.pipeline import build_models, run  # noqa: E402


def reangle(src, dst, roll_deg, pan):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), roll_deg, 1.0)
    m[0, 2] += pan * w
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    cap.release()
    out.release()


def summary(events, perf):
    return {"alerts": dict(Counter(e.type for e in events)),
            "zone_alerts": [{"track_id": e.track_id, "zone_id": e.zone_id, "alert_ts": round(e.alert_ts, 2)}
                            for e in events if e.type == "zone_entry"],
            "camera_moved": [{"start_ts": round(e.start_ts, 2), "alert_ts": round(e.alert_ts, 2), "note": e.note}
                             for e in events if e.type == "camera_moved"],
            "throughput_fps": perf["throughput_fps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_heldout/sb_rers1_0012.json", help="clip, zones and camera id")
    ap.add_argument("--roll", type=float, default=2.0, help="degrees")
    ap.add_argument("--pan", type=float, default=0.05, help="share of the frame width")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--out", default="eval/results/reangled_check.json")
    args = ap.parse_args()

    lab_file = ROOT / args.labels
    lab = json.loads(lab_file.read_text())
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    models = build_models(cfg)
    no_check = copy.deepcopy(cfg)
    no_check["view"]["check_every_s"] = math.inf  # the reference is taken, but never compared again
    work = ROOT / "runs/reangled"
    video = work / f"{lab_file.stem}.mp4"
    reangle(ROOT / lab["video"], video, args.roll, args.pan)

    res = {"clip": lab_file.stem, "roll_deg": args.roll, "pan": args.pan, "runs": {}}
    for name, c, v in (("original", cfg, ROOT / lab["video"]), ("reangled_no_check", no_check, video),
                       ("reangled_check", cfg, video)):
        events, perf = run(v, ROOT / lab["zones"], work / name, c, lab_file.stem, models=models)
        res["runs"][name] = summary(events, perf)
    (ROOT / args.out).write_text(json.dumps(res, indent=2))
    for name, r in res["runs"].items():
        print(f"{name:<18} {r['alerts']}  zone alerts {r['zone_alerts']}  camera_moved {r['camera_moved']}  "
              f"{r['throughput_fps']} fps")


if __name__ == "__main__":
    main()
