#!/usr/bin/env bash
# Every number in README.md and docs/ from a fresh clone: data, models, unit tests, alerts on the
# labelled clips (dev and held out) with their event-level scores, the 50-clip PPE evaluation, the
# design-note sweeps, and the speed benchmarks. Results are written to eval/results/.
# Needs ffmpeg (stream benchmark). The helmet classifier is committed; to retrain it, see
# scripts/train_helmet.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
command -v ffmpeg >/dev/null || { echo "ffmpeg is needed by scripts/bench_streams.py" >&2; exit 1; }
mac() { [[ "$(uname)" == "Darwin" ]]; }

$PY scripts/download_data.py                        # SteelBench 50-clip sample, SH17 weights, clips/ links
if mac; then
  $PY scripts/export_models.py --detector           # CoreML fp16 736x1280 (other platforms run the .pt)
else
  $PY scripts/export_models.py --detector --onnx    # same-resolution CPU backend
fi
$PY -m pytest tests -q                              # alert-logic and camera-moved unit tests

# Alerts and scores
$PY eval/events_eval.py --run                       # pipeline on the 3 dev clips -> out/, P/R table
$PY eval/events_eval.py --run --labels labels_heldout --results eval/results/events_heldout_rerun.json
# (eval/results/events_heldout.json is the frozen first run on the held-out clips; see README, Results)
$PY eval/steelbench_clips.py --by-condition         # 50 clips vs SteelBench per-person PPE labels
$PY eval/steelbench_clips.py --set ppe.helmet_method=detector --out eval/results/steelbench_clips_sh17_rule.json
$PY eval/steelbench_clips.py --sweep events.default.min_duration_s=1.0,1.5,2.0,2.5,3.0 \
  --out eval/results/sweep_min_duration.json
$PY eval/steelbench_clips.py --set detector.conf=0.05 --set tracker.track_low_thresh=0.05 \
  --out eval/results/steelbench_clips_floor_0.05.json  # lower detector and ByteTrack floors
# the 18 px head gate, set on the 43 sample clips from cameras outside the held-out set
$PY eval/steelbench_clips.py --skip-cameras-of labels_heldout --sweep ppe.classifier.min_head_px=10,18 \
  --by-condition --out eval/results/sweep_min_head_px.json
$PY eval/helmet_crops.py                            # helmet calls: full frame vs person crops vs classifier
$PY eval/pixel_budget.py                            # head sizes; what a 640 input keeps
$PY eval/track_eval.py                              # ID stability on the labelled people
$PY eval/track_eval.py --labels labels_heldout --results eval/results/tracking_heldout.json
$PY eval/view_eval.py                               # camera-moved check on the 50 clips
$PY eval/reangled_check.py                          # camera-moved check end to end on a re-angled clip

mkdir -p runs
# Person confirmation before any alert: min_hits detections at >= this confidence (0.5 in the config)
for c in 0 0.8; do
  sed "s/^    min_conf: 0.5 /    min_conf: $c /" configs/rules.yaml > runs/rules_confirm_${c}.yaml
  grep -q "^    min_conf: $c " runs/rules_confirm_${c}.yaml
  $PY eval/events_eval.py --run --config runs/rules_confirm_${c}.yaml --out runs/out_confirm_${c} \
    --results eval/results/events_confirm_conf_${c}.json
done

# The same lower floors on the dev clips (design note, occlusion)
sed -e "s/^  conf: 0.10 /  conf: 0.05 /" -e "s/^  track_low_thresh: 0.10$/  track_low_thresh: 0.05/" \
  configs/rules.yaml > runs/rules_floor_0.05.yaml
[[ $(grep -c "^  conf: 0.05 \|^  track_low_thresh: 0.05$" runs/rules_floor_0.05.yaml) == 2 ]]
$PY eval/events_eval.py --run --config runs/rules_floor_0.05.yaml --out runs/out_floor_0.05 \
  --results eval/results/events_floor_0.05.json

# Frame rate vs recall (design note)
for f in 5 3.2; do
  sed "s/^  process_fps: 10 /  process_fps: $f /" configs/rules.yaml > runs/rules_${f}fps.yaml
  grep -q "^  process_fps: $f " runs/rules_${f}fps.yaml
  $PY eval/events_eval.py --run --config runs/rules_${f}fps.yaml --out runs/out_${f}fps \
    --results eval/results/events_${f}fps.json
  $PY eval/steelbench_clips.py --set sampling.process_fps=$f --out eval/results/steelbench_clips_${f}fps.json
done

# The labelled clips with CoreML on the CPU only (run.py --cpu-only)
if mac; then
  sed "s/^  coreml_units: cpu_and_ne/  coreml_units: cpu_only/" configs/rules.yaml > runs/rules_cpu_only.yaml
  grep -q "^  coreml_units: cpu_only" runs/rules_cpu_only.yaml
  $PY eval/events_eval.py --run --config runs/rules_cpu_only.yaml --out runs/out_cpu_only \
    --results eval/results/events_cpu_only.json
  $PY eval/events_eval.py --run --config runs/rules_cpu_only.yaml --labels labels_heldout --out runs/out_cpu_only \
    --results eval/results/events_heldout_cpu_only.json
fi

# Why each labelled event was missed (per-frame stage attribution; needs readings logged)
for f in labels/*.json labels_heldout/*.json; do
  c=$(basename "$f" .json)
  $PY run.py --video "clips/$c.mp4" --zones "zones/$c.json" --camera-id "$c" --out runs/review --log-readings --overwrite
done
$PY eval/miss_review.py --out runs/review --all

# Speed
$PY scripts/bench_pipeline.py --labels labels labels_heldout --repeats 2 --variant torch-cpu sampling.idle_fps=null \
  --variant torch-cpu-5fps sampling.process_fps=5 sampling.idle_fps=null --results eval/results/pipeline_bench_cpu.json
if $PY -c "import openvino" 2>/dev/null; then       # Intel CPU / integrated GPU / NPU
  $PY scripts/export_models.py --detector --openvino --int8   # INT8 calibrated on non-evaluation cameras (nncf)
  $PY scripts/bench_pipeline.py --labels labels labels_heldout --repeats 2 \
    --variant ov-int8-cpu detector.backend=openvino detector.openvino_device=CPU sampling.idle_fps=null \
    --variant ov-int8-cpu-skip detector.backend=openvino detector.openvino_device=CPU \
    --variant ov-int8-cpu-skip-5fps detector.backend=openvino detector.openvino_device=CPU sampling.process_fps=5 \
    --results eval/results/bench_openvino_int8_cpu.json
  $PY scripts/bench_pipeline.py --labels labels labels_heldout --repeats 1 \
    --variant ov-int8-cpu-skip-8fps detector.backend=openvino detector.openvino_device=CPU sampling.process_fps=8 \
    --variant ov-int8-cpu-skip-6fps detector.backend=openvino detector.openvino_device=CPU sampling.process_fps=6 \
    --results eval/results/bench_openvino_int8_cpu_6_8fps.json
  $PY scripts/bench_pipeline.py --labels labels labels_heldout --repeats 3 \
    --variant openvino-gpu detector.backend=openvino detector.openvino_device=GPU sampling.idle_fps=null \
    --variant openvino-gpu-5fps detector.backend=openvino detector.openvino_device=GPU sampling.process_fps=5 sampling.idle_fps=null \
    --results eval/results/bench_openvino_gpu.json
fi
$PY scripts/bench_detector.py                       # detector backends; skips models not exported (see its docstring)
$PY scripts/bench_streams.py --streams 1 2 4 6      # separate processes, one camera each
$PY scripts/bench_streams.py --streams 1 2 --cpu-only
$PY scripts/bench_streams.py --streams 1 2 --device cpu
$PY scripts/bench_multi.py                          # cameras in one process, shared models
