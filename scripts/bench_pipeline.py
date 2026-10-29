"""Whole-pipeline speed and event accuracy per setting, on the same labelled clips, repeated.

Each variant is a set of dotted config overrides. Every variant runs every clip `--repeats` times with
the models loaded once (load and warm-up are excluded), and reports full wall time, end-to-end
throughput, per-stage mean and p95 over all processed frames, and the event scores of the last
repeat against the clip labels (events_eval). A speed candidate is only worth keeping if its event
recall holds; this puts both numbers in one table.

python scripts/bench_pipeline.py --labels labels labels_heldout --repeats 3 \
    --variant base \
    --variant openvino-cpu detector.backend=openvino detector.openvino_device=CPU \
    --variant openvino-gpu detector.backend=openvino detector.openvino_device=GPU \
    --variant fps5 sampling.process_fps=5
"""
import argparse
import copy
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from events_eval import COUNTS, load_events, pr, score_clip  # noqa: E402
from steelbench_clips import set_key  # noqa: E402
from zapdos.pipeline import STAGES, Camera, build_models  # noqa: E402


def run_clip(cam):
    """One full pass; returns (wall_s, processed frames, skipped samples, per-stage ms lists)."""
    wall0 = time.perf_counter()
    try:
        for frame_idx, ts, frame, decode_ms in cam.frames():
            cam.step(frame_idx, ts, frame, decode_ms, quiet=True)
    finally:
        cam.close(wall0)
    return time.perf_counter() - wall0, len(cam.ms["detect"]), cam.skipped, cam.ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=["labels"])
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--variant", nargs="+", action="append", required=True, metavar=("NAME", "KEY=VALUE"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--results", default="eval/results/pipeline_bench.json")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / args.config).read_text())
    base["output"]["overwrite"] = True
    labels = [f for d in args.labels for f in sorted((ROOT / d).glob("*.json"))]
    if not labels:
        raise SystemExit("no label files")
    rows = []
    for name, *sets in args.variant:
        cfg = copy.deepcopy(base)
        for kv in sets:
            set_key(cfg, *kv.split("=", 1))
        models = build_models(cfg)
        out = ROOT / args.out / name
        stage_ms, walls, frames, skipped, video_s = {s: [] for s in STAGES}, [], 0, 0, 0.0
        totals, per_clip = Counter(), {}
        for rep in range(args.repeats):
            for f in labels:
                lab = json.loads(f.read_text())
                cam = Camera(ROOT / lab["video"], ROOT / lab["zones"], out, cfg, models, f.stem)
                wall, n, k, ms = run_clip(cam)
                walls.append(wall)
                if rep == 0:
                    frames += n
                    skipped += k
                    video_s += cam.source_stats["source_frames"] / cam.src_fps
                for s in STAGES:
                    stage_ms[s] += ms[s]
                if rep == args.repeats - 1:
                    s = score_clip(lab, load_events(out / f.stem))
                    per_clip[f.stem] = {k: s[k] for k in (*COUNTS, "median_delay_s")}
                    for k in COUNTS:
                        totals[k] += s[k]
            print(f"{name}: repeat {rep + 1}/{args.repeats} done", flush=True)
        wall_per_rep = np.array(walls).reshape(args.repeats, -1).sum(1)
        p, r = pr(totals)
        row = {"variant": name, "overrides": sets, "detector": f"{models[0].model_path} on {models[0].backend}",
               "process_fps": cfg["sampling"]["process_fps"], "idle_fps": cfg["sampling"].get("idle_fps"),
               "frames": frames, "frames_skipped": skipped, "video_s": round(video_s, 1),
               "wall_s_per_repeat": [round(float(w), 2) for w in wall_per_rep],
               "throughput_fps": round(frames / float(wall_per_rep.mean()), 2),
               # sampled slots (processed + skipped) per wall-second: the camera rate this keeps up with
               "sampled_fps": round((frames + skipped) / float(wall_per_rep.mean()), 2),
               "realtime_x": round(video_s / float(wall_per_rep.mean()), 2),
               "throughput_fps_min": round(frames / float(wall_per_rep.max()), 2),
               "stages": {s: {"mean_ms": round(float(np.mean(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
                          for s, v in stage_ms.items() if v},
               "events": {**totals, "precision": p, "recall": r}, "per_clip": per_clip}
        row["pipeline_ms_per_frame"] = round(sum(row["stages"][s]["mean_ms"] for s in STAGES if s in row["stages"]), 1)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("variant", "throughput_fps", "sampled_fps", "realtime_x",
                                              "frames_skipped", "pipeline_ms_per_frame", "events")}), flush=True)

    print(f"\n{'variant':<16} {'fps':>6} {'sampled':>7} {'rt x':>5} {'ms/frame':>8} {'detect ms':>9} {'p95':>6}"
          "  TP/labelled  FP  dup  precision recall")
    for r in rows:
        e, d = r["events"], r["stages"]["detect"]
        print(f"{r['variant']:<16} {r['throughput_fps']:>6} {r['sampled_fps']:>7} {r['realtime_x']:>5} "
              f"{r['pipeline_ms_per_frame']:>8} {d['mean_ms']:>9} "
              f"{d['p95_ms']:>6}  {e['true']:>3}/{e['labelled']:<8} {e['false']:>3} {e['duplicates']:>4}  "
              f"{e['precision']!s:>9} {e['recall']!s:>6}")
    out = ROOT / args.results
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
                               "repeats": args.repeats, "clips": [f.stem for f in labels], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
