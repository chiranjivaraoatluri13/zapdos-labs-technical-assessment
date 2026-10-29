# CCTV safety events

## Part 0: Cold Open

**Q1. How many frames can you afford?** `8 / 0.025 = 320 FPS`, or **3.2 FPS/camera**, assuming full batches and no other overhead. Inference binds if frames can be skipped before decoding; decoding all 2,500 incoming FPS would also overload the 600-FPS decoder. At 1.4 m/s, movement is **0.44 m/sample**. Assuming a 2 m-wide zone, that gives 4–5 observations; 1.5 seconds of helmet confirmation gives about five, provided the head is readable. Brief crossings remain risky. I would change stream/sampling rates first, prioritize activity, then add boxes if necessary.

**Q2. How many false alarms?** `40 × 3 × 5 × 3,600 × 0.001 = 2,160` false no-helmet frames/hour, or **17,280 per shift**, before filtering. An 8-of-10 vote helps with independent errors; a hood or poor lighting can stay wrong across frames. Eight observations at 5 FPS span at least 1.4 seconds. I would combine temporal confirmation, deduplication, cooldowns, secondary checks and site calibration. Two false alerts per shift means **one per 160 camera-hours**; validation must count missed violations too.

**Q3. Where is the person?** Test a ground-contact point against the floor polygon, using visible feet or the box bottom-centre. Feet are **visible, estimated or unknown**; recent full-body tracking supports short estimates, while unknown feet cannot start an alert. Entry/exit margins and temporal confirmation limit flicker and premature episode endings. My proposed 3D ground mesh is **not implemented**: I chose 2D polygons because a mesh needs additional calibration, reconstruction and validation.

## What I built

I used SH17 YOLOv8s, ByteTrack's five-second lost-track buffer and a MobileNetV3-small head-crop classifier. PPE needs 1.5 seconds of persistence and a 70% vote over two seconds; zones need 0.5 seconds and a 60% vote over one second. A 60-second cooldown limits repeats. [Rules are configurable](configs/rules.yaml); camera drift pauses zone decisions.

## Setup once, then run the selected videos

Use Python 3.12 on Windows. From the repository root:

```powershell
python -m pip install -r requirements-cpu.txt
python scripts/download_data.py
python scripts/export_models.py --detector --openvino --int8
```

After setup, this downloads the five pinned clips, checks their SHA-256 values, runs the whole pipeline on the tested Intel GPU and verifies the evidence:

```powershell
python scripts/run_sample.py --all --backend openvino --openvino-device GPU
```

For CPU, replace the GPU flags with `--cpu-only`; a missing INT8 export falls back to slower PyTorch. [Individual video commands and review notes](TEST_VIDEOS.md).

| Selected video | Emitted alerts | Saved output | Evidence |
|---|---:|---|---|
| 02_single_mill_steam | 2 | [Annotated video](out/demo_samples/02_single_mill_steam/annotated.mp4) | [Clips](out/demo_samples/02_single_mill_steam/clips) |
| 10_crowded_workshop_alt | 7 | [Annotated video](out/demo_samples/10_crowded_workshop_alt/annotated.mp4) | [Clips](out/demo_samples/10_crowded_workshop_alt/clips) |
| 12_tippler_low_light | 2 | [Annotated video](out/demo_samples/12_tippler_low_light/annotated.mp4) | [Clips](out/demo_samples/12_tippler_low_light/clips) |
| 01_single_mill | 4 | [Annotated video](out/demo_samples/01_single_mill/annotated.mp4) | [Clips](out/demo_samples/01_single_mill/clips) |
| 07_conveyor_group | 1 | [Annotated video](out/demo_samples/07_conveyor_group/annotated.mp4) | [Clips](out/demo_samples/07_conveyor_group/clips) |

These are emitted-alert counts. All 16 identities/timestamps reproduced, and every snapshot/clip passed verification; 12 clips have shortened source-boundary windows. The selected clips' hand-labelled accuracy and median-delay table remains pending annotation.

Outputs: `runs/samples/<camera_id>/` contains annotated video, snapshots, clips, finalized/live JSONL, readings, timing and provenance. Earlier runs are archived. [Source manifest](samples/review_manifest.json) · [New-camera setup and full guide](docs/PIPELINE_GUIDE.md). Demonstration polygons are illustrative; real restrictions require site policy.

## Performance and checks

Six-clip Windows benchmark: 81 seconds of video; Intel Core Ultra 7 256V (8 cores/8 threads) and Intel iGPU; detector input 736 × 1280. Mean timings use 10 FPS sampling, no idle skipping and exclude loading/warm-up. These benchmarks predate the identity/display fixes.

| Hardware / backend | Input res. | Decode ms | Detect ms | Track + rules ms | Processed FPS | 10-FPS equivalents* |
|---|---|---:|---:|---:|---:|---:|
| CPU / PyTorch | 1920 × 1080 | 7.1 | 327.5 | 1.2 | 2.95 | 0.30 |
| CPU / OpenVINO INT8 | 1920 × 1080 | 6.5 | 114.5 | 0.8 | 8.02 | 0.80 |
| Intel iGPU / OpenVINO | 1920 × 1080 | 8.2 | 32.0 | 1.3 | 22.32 | 2.23 |

*FPS/10 is a workload equivalent, not measured concurrency. Throughput includes helmet classification and media encoding. INT8 CPU and GPU exceeded their respective 5/10 FPS targets; 5 FPS sampling lost a labelled event.*

Latest full tests: **93 passed, one skipped** (INT8 export unavailable). Run `python -m pytest tests -q` and `python scripts/verify_run.py runs/samples`. [Per-video timings and replay checks](eval/results/sample_review_20261028.json). Media validity does not establish alert correctness.

## Completed stretch objective

**Zone drift detection.** The pipeline compares the camera view with its saved reference, emits `camera_moved` after a persistent mismatch and pauses zone decisions until the view matches again. PPE detection continues. [Implementation](src/zapdos/view.py) and [tests](tests/test_view.py) cover movement, recovery, jitter and temporary occlusion. A [recorded re-angle experiment](eval/results/reangled_check.json) detected a 103-pixel zone-corner shift and alerted at 4.0 seconds; this was an induced camera transform, not a physical camera test.

## Where it is most likely wrong

Small heads, hidden workers, glare and coveralls remain difficult: both additional-camera helmet violations were missed. A separate conveyor test alerted on machinery. In `01_single_mill`, the operator receives a new identity and another alert pair. Crowds and hidden-foot estimates can also give incorrect identities or zone decisions.

I would check alerted and non-alerted tracks against independent labels. IR, full-shift reliability, live ingest and a fresh installation remain unverified. [Design note](docs/DESIGN.md).

## What I used and time spent

Data/models: [SH17](https://github.com/ahmadmughees/SH17dataset), [SteelBench](https://huggingface.co/datasets/ThinkingHub/PP) (CC BY-NC 4.0), [Hard Hat Workers](https://huggingface.co/datasets/keremberke/hard-hat-detection). Training/export scripts and metadata are included. Libraries: Ultralytics, PyTorch/torchvision, OpenCV, OpenVINO, ONNX Runtime, NumPy, SciPy, PyYAML, pytest.

I used **ChatGPT 6.1 Sol Light and Cursor** for coding assistance and review. **Time spent: 6.5 hours.**
