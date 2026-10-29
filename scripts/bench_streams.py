"""How many streams one machine sustains: run N copies of run.py at once and read their perf.json.

Each stream is the busiest dev clip looped to 60 s (needs ffmpeg on PATH), processed as fast as the
machine allows. A stream keeps up with a live camera while its own throughput stays >=
sampling.process_fps. Results merge into eval/results/streams.json per setting (default, cpu_only,
cpu) and stream count, with the one-stream run's perf.json (per-stage times) kept as well.

python scripts/bench_streams.py --streams 1 2 4 6 [--cpu-only | --device cpu]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval/results/streams.json"
CLIP = ROOT / "clips/sb_crm_workshop_0109.mp4"
ZONES = ROOT / "zones/sb_crm_workshop_0109.json"


def looped(clip, seconds, out):
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", str(clip), "-t", str(seconds),
                        "-c", "copy", str(out)], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--device", help="passed to run.py (e.g. cpu: the PyTorch detector on CPU)")
    ap.add_argument("--cpu-only", action="store_true", help="passed to run.py")
    args = ap.parse_args()

    video = looped(CLIP, args.seconds, ROOT / f"runs/bench/loop_{args.seconds}s.mp4")
    tag = "cpu_only" if args.cpu_only else args.device or "default"
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    entry = results.setdefault(tag, {"rows": [], "one_stream_perf": None})
    for n in args.streams:
        out = ROOT / f"runs/bench/{tag}_x{n}"
        cmd = [sys.executable, str(ROOT / "run.py"), "--video", str(video), "--zones", str(ZONES), "--out", str(out)]
        cmd += ["--device", args.device] if args.device else []
        cmd += ["--cpu-only"] if args.cpu_only else []
        t0 = time.perf_counter()
        procs = [subprocess.Popen(cmd + ["--camera-id", f"s{i}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 for i in range(n)]
        if any(p.wait() for p in procs):
            sys.exit(f"a stream failed at n={n}")
        wall = time.perf_counter() - t0
        perfs = [json.loads((out / f"s{i}/perf.json").read_text()) for i in range(n)]
        fps = [p["throughput_fps"] for p in perfs]
        det = [p["stages"]["detect"]["p95_ms"] for p in perfs]
        row = {"streams": n, "min_fps": min(fps), "mean_fps": round(sum(fps) / n, 1),
               "total_fps": round(sum(fps), 1), "detect_p95_ms": max(det), "wall_s": round(wall, 1),
               "keeps_up": min(fps) >= perfs[0]["process_fps"]}
        print(json.dumps(row), flush=True)
        entry["rows"] = sorted([r for r in entry["rows"] if r["streams"] != n] + [row], key=lambda r: r["streams"])
        if n == 1:
            entry["one_stream_perf"] = perfs[0]
        RESULTS.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
