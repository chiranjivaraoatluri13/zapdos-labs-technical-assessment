"""Cache detector outputs once, then compare ByteTrack settings on identical detections.

python eval/tracker_replay.py --videos-dir PATH --cache-dir PATH --out PATH
Only development cameras select settings; held-out cameras are reported separately.
Caches are JSON, include the detector/source fingerprint, and do not contain inferred labels.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from ultralytics.engine.results import Boxes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from zapdos.detect import PPEDetector, available_device, detector_model_path, model_fingerprint
from zapdos.pipeline import open_video, sample_frames
from zapdos.track import PersonTracker
from track_eval import score_full


def cached_detections(video, cfg, directory, detector=None):
    model = detector_model_path(cfg["detector"])  # the exported graph actually run, not just the .pt
    key = hashlib.sha256(json.dumps({"detector": cfg["detector"], "sampling": cfg["sampling"],
        "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "model": str(model), "model_sha256": model_fingerprint(model),
        "code": (ROOT / "src/zapdos/detect.py").read_text()}, sort_keys=True).encode()).hexdigest()
    path = directory / f"{video.stem}-{key[:12]}.json"
    if path.exists():
        return json.loads(path.read_text())
    if detector is None:
        detector = PPEDetector(cfg["detector"])
    cap, src_fps, size = open_video(video)
    fps = min(cfg["sampling"]["process_fps"], src_fps)
    result = {"fingerprint": key, "backend": detector.backend, "video": str(video), "size": size, "fps": fps, "frames": []}
    try:
        for index, ts, frame, _ in sample_frames(cap, src_fps, fps):
            d = detector(frame)
            result["frames"].append({"index": index, "ts": ts, "persons": d.persons.data.tolist(),
                                     "heads": d.heads.tolist(), "helmets": d.helmets.tolist(), "vests": d.vests.tolist()})
    finally:
        cap.release()
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def replay_tracks(cache, cfg):
    tracker = PersonTracker(cfg, cache["fps"])
    frames = []
    for f in cache["frames"]:
        persons = Boxes(np.asarray(f["persons"], np.float32).reshape(-1, 6), cache["size"][::-1])
        frames.append((f["index"], f["ts"], tracker(persons), None))
    return {"size": cache["size"], "fps": cache["fps"], "frames": frames}


def summarize(rows, diags=None):
    r = [p for clip in rows.values() for p in clip.values()]
    total = sum(p["frames"] for p in r)
    out = {"people": len(r), "never_tracked": sum(not p["ids"] for p in r),
           "one_id": sum(len(p["ids"]) == 1 for p in r), "switches": sum(p["switches"] for p in r),
           "tracked": round(sum(p["matched_frames"] for p in r) / max(1, total), 4)}
    if diags:
        out["reused_ids"] = sum(len(d["id_reuse"]) for d in diags.values())
        out["unmatched_tracks"] = sum(len(d["unmatched_tracks"]) for d in diags.values())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--baseline-config", help="optional baseline config; default reproduces confidence fusion on / 3s memory")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["detector"]["device"] = available_device(cfg["detector"]["device"])
    detector = PPEDetector(cfg["detector"])
    baseline = ({**cfg["tracker"], "fuse_score": True, "track_buffer_s": 3.0}
                if not args.baseline_config else yaml.safe_load(Path(args.baseline_config).read_text())["tracker"])
    variants = {"baseline": copy.deepcopy(baseline)}
    for fuse, match, buffer in [(False, .8, 3), (False, .75, 3), (False, .85, 3), (False, .8, 5)]:
        variants[f"fuse_{fuse}_match_{match}_buffer_{buffer}"] = {
            **cfg["tracker"], "fuse_score": fuse, "match_thresh": match, "track_buffer_s": buffer}
    results = {name: {"settings": tc, "development": {}, "heldout": {}, "diagnostics": {"development": {}, "heldout": {}}}
               for name, tc in variants.items()}
    for split, folder in [("development", "labels"), ("heldout", "labels_heldout")]:
        for path in sorted((ROOT / folder).glob("*.json")):
            labels = json.loads(path.read_text())
            cache = cached_detections(Path(args.videos_dir) / Path(labels["video"]).name,
                                      cfg, Path(args.cache_dir), detector)
            for name, tc in variants.items():
                rows, diag = score_full(replay_tracks(cache, tc), labels["people"])
                results[name][split][path.stem] = rows
                results[name]["diagnostics"][split][path.stem] = diag
            print(f"cached/replayed {path.stem}", flush=True)
    for name, result in results.items():
        result["summary"] = {s: summarize(result[s], result["diagnostics"][s]) for s in ("development", "heldout")}
        print(name, result["summary"], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "Approximate hand-box diagnostics at IoU >= 0.3; not official MOT metrics.",
                               "variants": results}, indent=2))


if __name__ == "__main__":
    main()
