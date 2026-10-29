"""Save the view a zone file was drawn on, so the pipeline can tell when the camera has moved
(src/zapdos/view.py). Run it when drawing or redrawing a camera's zones.

python scripts/zone_reference.py --video clips/x.mp4 --zones zones/x.json [--at 0]
"""
import argparse
import json
import re
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"no frame at {args.at} s in {args.video}")
    zones = Path(args.zones)
    ref = zones.with_suffix(".jpg")
    h = round(frame.shape[0] * args.width / frame.shape[1])
    cv2.imwrite(str(ref), cv2.resize(frame, (args.width, h), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    text = zones.read_text()  # edit in place to keep the hand-written layout
    if '"reference"' in text:
        text = re.sub(r'"reference": "[^"]*"', f'"reference": "{ref.name}"', text)
    else:
        text = text.replace('"zones": [', f'"reference": "{ref.name}",\n  "zones": [', 1)
    assert json.loads(text)["reference"] == ref.name
    zones.write_text(text)
    print(f"{ref} ({args.width}x{h}) -> {zones}")


if __name__ == "__main__":
    main()
