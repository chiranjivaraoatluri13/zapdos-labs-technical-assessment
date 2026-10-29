"""In-domain head crops for the helmet classifier, weakly labelled from SteelBench's full release.

SteelBench labels each worker's helmet (worn / not_worn / cannot_determine) when a clip has <= 5
workers, without boxes. In a clip where every worker has the same label, every person the detector
finds gets that label: a free, in-domain training set (same plants, lighting, camera heights, cloth
head-wraps) that Hard Hat Workers can't provide.

Clips that mix helmeted and bare workers can't be labelled that way, but they hold most of the
in-domain bare heads (seen from behind, from above, under cloth wraps). Their crops are labelled by
eye from numbered contact sheets (--sheets) into --hand-labels, one "<key> bare|helmet|reject" per line.

Leakage control: clips from any camera that appears in the 50-clip sample are excluded, so the
sample (the evaluation set) stays a test on cameras the classifier has never seen.

Only heads the detector found (head or helmet box) are kept; estimated head positions are too often
wrong (bent-over workers, false person boxes) to carry a label. Crops listed in --rejected (checked by
eye on a contact sheet: head detections on gears/machinery, or headwear that might be a helmet in an
"all bare" clip) are dropped, since a gear labelled "bare" teaches the very false alarm the classifier
exists to prevent.

python scripts/build_steelbench_heads.py    # selects clips, downloads their frames, writes the crops
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from download_data import fetch, frame_jobs  # noqa: E402
from train_helmet import STORE, STORE_CTX  # noqa: E402
from zapdos.detect import PPEDetector  # noqa: E402
from zapdos.helmet import square_crop  # noqa: E402
from zapdos.ppe import head_candidates  # noqa: E402


SHEET_COLS, SHEET_ROWS, TILE = 12, 8, 112  # contact sheet layout, tile side in px


def _fully_labelled(a):
    persons = a.get("persons") or []
    return bool(persons) and len(persons) == a.get("num_workers")


def select_clips(full, sample_clips):
    """(clip_id, camera_id, label): label 0 = every worker helmeted, 1 = every worker bare,
    None = mixed (>= 1 bare worker; needs hand labels per crop)."""
    man = pd.read_csv(full / "manifests/batch_manifest.csv")
    cam = dict(zip(man.clip_id, man.camera_id))
    held = {cam[c] for c in sample_clips if c in cam}
    picked = []
    for f in sorted((full / "annotations").glob("*.json")):
        a = json.loads(f.read_text())
        if not _fully_labelled(a) or cam.get(a["clip_id"]) in held:
            continue
        labels = {p["ppe"].get("helmet") for p in a["persons"]}
        if labels in ({"worn"}, {"not_worn"}):
            picked.append((a["clip_id"], cam[a["clip_id"]], int(labels == {"not_worn"})))
        elif "not_worn" in labels:
            picked.append((a["clip_id"], cam[a["clip_id"]], None))
    return picked, held


def detected_heads(detector, img, min_person_conf, head_region):
    """The most confident head or helmet box of each confident person that has one."""
    det = detector(img)
    persons = det.persons.xyxy[det.persons.conf >= min_person_conf]
    heads, helmets = head_candidates(det.heads, det.helmets, persons, head_region)
    for i in range(len(persons)):
        cands = heads.get(i, []) + helmets.get(i, [])
        if cands:
            yield max(cands, key=lambda h: h[4])


def crop_label(clip_label, key, hand):
    """0 helmet / 1 bare, "reject", or None for a crop from a mixed clip that has no hand label yet."""
    if clip_label is not None:
        return clip_label
    lab = hand.get(key)
    return lab if lab in (None, "reject") else int(lab == "bare")


def read_hand_labels(path):
    lines = path.read_text().splitlines() if path.exists() else []
    return dict(line.split() for line in lines if line.strip() and not line.startswith("#"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="data/raw/steelbench_full")
    ap.add_argument("--sample", default="data/raw/steelbench")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--min-person-conf", type=float, default=0.4)
    ap.add_argument("--min-head-px", type=float, default=10)
    ap.add_argument("--rejected", default="data/labels/sb_heads_rejected.txt")
    ap.add_argument("--hand-labels", default="data/labels/sb_heads_hand.txt")
    ap.add_argument("--sheets", default="data/cache/sb_heads_sheets")
    ap.add_argument("--out", default="data/cache/sb_heads.npz")
    args = ap.parse_args()

    full = ROOT / args.full
    sample_clips = {p.stem for p in (ROOT / args.sample / "clips").glob("*.mp4")}
    picked, held = select_clips(full, sample_clips)
    kinds = Counter({0: "all helmeted", 1: "all bare", None: "mixed"}[y] for *_, y in picked)
    print(f"{len(picked)} clips on {len({c for _, c, _ in picked})} cameras {dict(kinds)}; "
          f"{len(held)} sample cameras held out")

    jobs = frame_jobs(full, [c for c, _, _ in picked])
    with ThreadPoolExecutor(16) as pool:
        got = Counter(m.split()[0] for m in pool.map(lambda j: fetch(*j), jobs))
    print(f"frames: {dict(got)}")

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    detector = PPEDetector(cfg["detector"])
    rej_file = ROOT / args.rejected
    rejected = set(rej_file.read_text().split()) if rej_file.exists() else set()
    hand = read_hand_labels(ROOT / args.hand_labels)
    rows, unlabelled = [], []  # rows: (crop, label, camera, clip, head height, key)
    for clip, camera, label in picked:
        for img_path in sorted((full / "frames" / clip).glob("frame_*.jpg")):
            img = cv2.imread(str(img_path))
            for hb in detected_heads(detector, img, args.min_person_conf, cfg["ppe"]["head_region"]):
                key = f"{clip}/{img_path.stem}@{int(hb[0])},{int(hb[1])}"
                if hb[3] - hb[1] < args.min_head_px or key in rejected:
                    continue
                crop = square_crop(img, hb, ctx=STORE_CTX, out=STORE)
                y = crop_label(label, key, hand)
                if y is None:
                    unlabelled.append((key, crop))
                elif y != "reject":
                    rows.append((crop, y, camera, clip, float(hb[3] - hb[1]), key))

    xs, ys, cams, clips, hhs, keys = zip(*rows)
    y = np.array(ys, np.int64)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=np.stack(xs), y=y, cam=np.array(cams), clip=np.array(clips), hh=np.array(hhs, np.float32),
             key=np.array(keys))
    print(f"{len(rejected)} crops rejected by review")
    print(f"{len(y)} head crops ({y.sum()} bare) from {len(set(clips))} clips, median head "
          f"{np.median(hhs):.0f} px -> {out}")
    by_cam = Counter(zip(cams, ys))
    for c in sorted({c for c, _ in by_cam}):
        print(f"  {c:<40} helmet {by_cam[(c, 0)]:>4}  bare {by_cam[(c, 1)]:>3}")
    if unlabelled:
        write_sheets(unlabelled, ROOT / args.sheets)
        print(f"{len(unlabelled)} crops from mixed clips have no hand label yet: contact sheets in {args.sheets}; "
              f"add '<key> bare|helmet|reject' lines to {args.hand_labels} and rerun")


def write_sheets(items, out_dir):
    """Numbered contact sheets of the head (centre of the stored crop) + index.txt: number -> key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = int(STORE * (1 - 1.3 / STORE_CTX) / 2)  # show ~1.3x the head box instead of 2x
    tiles = []
    for n, (_, crop) in enumerate(items):
        t = cv2.resize(crop[m:STORE - m, m:STORE - m], (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(t, (0, 0), (34, 15), (0, 0, 0), -1)
        cv2.putText(t, str(n), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(t)
    per = SHEET_COLS * SHEET_ROWS
    for s in range(0, len(tiles), per):
        page = tiles[s:s + per] + [np.zeros_like(tiles[0])] * (-len(tiles[s:s + per]) % SHEET_COLS)
        cv2.imwrite(str(out_dir / f"sheet_{s // per:02d}.jpg"),
                    np.vstack([np.hstack(page[r:r + SHEET_COLS]) for r in range(0, len(page), SHEET_COLS)]))
    (out_dir / "index.txt").write_text("".join(f"{n} {k}\n" for n, (k, _) in enumerate(items)))


if __name__ == "__main__":
    main()
