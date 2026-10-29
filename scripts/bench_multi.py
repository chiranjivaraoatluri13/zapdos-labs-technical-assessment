"""How many cameras one machine sustains in a single process (shared models, src/zapdos/multi.py).

Each camera is the busiest dev clip looped to 60 s, all processed as fast as the machine allows.
The cameras keep up with live video while the slowest one's throughput stays >= the processing fps.
Every configuration runs in a fresh process so peak memory is its own.

python scripts/bench_multi.py --cameras 1 2 3 4 5 6 7 8 10 --fps 10 5
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_streams import CLIP, ZONES, looped  # noqa: E402


def worker(n, fps, seconds):
    from zapdos.multi import run_many
    cfg = yaml.safe_load((ROOT / "configs/rules.yaml").read_text())
    cfg["sampling"]["process_fps"] = fps
    video = looped(CLIP, seconds, ROOT / f"runs/bench/loop_{seconds}s.mp4")
    _, summary = run_many([(video, ZONES, f"s{i}") for i in range(n)], ROOT / f"runs/bench/multi_{fps}fps_x{n}",
                          cfg, quiet=True)
    print("RESULT " + json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 10])
    ap.add_argument("--fps", type=float, nargs="+", default=[10, 5])
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--worker", nargs=2, type=float, metavar=("N", "FPS"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(int(args.worker[0]), args.worker[1], args.seconds)
        return

    rows = []
    for fps in args.fps:
        for n in args.cameras:
            p = subprocess.run([sys.executable, __file__, "--worker", str(n), str(fps), "--seconds", str(args.seconds)],
                               capture_output=True, text=True, check=False)  # failures are reported below
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            if p.returncode or not line:
                sys.exit(f"worker failed at {n} cameras, {fps} fps:\n{p.stderr[-2000:]}")
            rows.append(json.loads(line[0][7:]))
            print(json.dumps(rows[-1]), flush=True)
    (ROOT / "eval/results/multi_camera.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
