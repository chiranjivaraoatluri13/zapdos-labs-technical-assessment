# CCTV safety events

## Part 0: Cold Open

**Q1. How many frames can you afford?** `8 / 0.025 = 320 FPS`, or **3.2 FPS per camera** across 100 cameras, assuming full batches and ignoring other overhead. Assuming frames can be skipped before decoding, inference is the bottleneck; decoding all 2,500 incoming FPS would exceed the 600 FPS decoder limit. At 1.4 m/s, a worker moves **0.44 m between samples**; assuming a 2 m-wide zone, expect roughly 4–5 observations, but brief crossings and small or occluded heads remain difficult. I would prioritize activity-aware sampling and high-risk cameras, then add edge boxes if more capacity is required.

**Q2. How many false alarms?** `40 × 3 × 5 × 3,600 × 0.001 = 2,160` false no-helmet person-frames/hour, or **17,280 per 8-hour shift**, before temporal filtering. An 8-of-10 vote reduces independent random errors, but hoods, lighting and camera angles cause correlated errors; at 5 FPS, the first eight observations span at least 1.4 s. I would combine temporal confirmation, track-based deduplication, cooldowns, secondary checks and site-calibrated thresholds. Two false alerts per shift means **one per 160 camera-hours**; validate event-level false alerts and missed violations over multiple shifts.

**Q3. Where is the person?** Test an estimated ground-contact point against the floor polygon, using visible feet or the bounding-box bottom-centre. My implementation distinguishes **visible, estimated and unknown** feet; recent full-body tracking supports short-term estimates, while unknown feet cannot start a zone alert. Temporal confirmation and separate entry/exit margins limit boundary flicker and prevent one noisy or occluded frame from ending an episode. My original calibrated 3D ground-mesh idea is **not implemented**: I chose 2D floor polygons for fixed CCTV because a mesh would require additional camera calibration, reconstruction and validation.

Video → SH17 detection → ByteTrack → head-crop helmet classifier → persistent, deduplicated alerts. [Design](docs/DESIGN.md) · [Current audit](docs/ASSESSMENT_AUDIT.md) · [Historical M2 experiments](docs/RESULTS.md).

## Selected demonstration videos

The five selected videos, annotated outputs, evidence links and individual one-line commands are in [TEST_VIDEOS.md](TEST_VIDEOS.md). Their complete sample outputs are under [out/demo_samples](out/demo_samples). After the setup below, run all five with:

```powershell
python scripts/run_sample.py --all --backend openvino --openvino-device GPU
```

The helper downloads each pinned source clip, verifies its checksum, runs the pipeline and checks the saved evidence. The selected examples emitted 16 alerts with 16 readable snapshots and clips. They are demonstrations selected after review; `01_single_mill` retains a known identity-continuity issue and possible duplicate alert pair. See the review notes before interpreting alert counts as accuracy.

## Run

From the project root, Windows CPU / Python 3.12:

```powershell
python -m pip install -r requirements-cpu.txt
python scripts/download_data.py
python scripts/export_models.py --detector --openvino --int8
python run.py --video clips/sb_crm_workshop_0109.mp4 --zones zones/sb_crm_workshop_0109.json --cpu-only --save-video --out runs/demo
```

The downloader fetches SteelBench sample videos and SH17 weights; the helmet classifier is committed. The export writes the OpenVINO detector and an INT8 copy calibrated on 198 frames from 15 SteelBench cameras that are not evaluation cameras. Off macOS, `--cpu-only` uses that INT8 model on every CPU thread (PyTorch CPU if it isn't exported). For a live camera on this CPU add `--process-fps 8`; `--no-idle-skip` turns frame skipping off. Apple’s historical pins are in `requirements.txt`; export CoreML with `python scripts/export_models.py --detector`. Auto otherwise uses PyTorch on an available GPU/CPU.

New camera: draw a zone JSON, save its reference with `scripts/zone_reference.py`, then supply `--zones` in the same run command. Without zones, PPE still runs. Restrictions are site policy, not inferred from video. Use `--config` for camera-specific rules.

Optional CPU export: `python scripts/export_models.py --detector --onnx`, then add `--backend onnx` to the run. ONNX was slower here and remains opt-in.

Intel iGPU: add `--backend openvino --openvino-device GPU` (fp32 model). It gives the same event scores, at 22 FPS.

## Rules and evidence

ByteTrack: low-score recovery, **5s lost buffer**, match threshold 0.8, confidence fusion off. Rules see the detector's own box, never a Kalman prediction, so predicted-only boxes supply no PPE or floor evidence.

**Person confirmation.** No helmet, vest or zone alert until a track has ≥3 detections at ≥0.5 confidence, ≥0.3s age, ≥24px height and a person-like aspect (`tracker.confirm`). On the six clips this removed 4 false alerts (all on the degraded `sb_bf_cdi_cc2_0031`) with no lost event and no change in median alert delay.

**Identity, episode, cooldown.** A new tracker ID joins a lost person only when exactly one candidate fits a short window (walked ≤5s within one person-height of the predicted position, or stayed ≤10s in place), with similar size, and, for crops ≥96px, compatible torso colour. A runner-up within the margin, or two new IDs claiming the same person, gives a new identity marked `identity.how: "ambiguous"` with the candidates. An episode closes when the violation ends; the 60s cooldown suppresses a later episode (listed in `suppressed_episodes`) without moving the earlier `end_ts`.

**PPE.** Helmets and vests are matched one-to-one to people (Hungarian). If another person could own the item, the reading is unknown (`ambiguous_owner`). `no_vest` needs a torso ≥95% in frame and ≤30% covered by people in front or configured `occluders`. Every unknown carries a reason: `edge_cut`, `no_head_box`, `head_small`, `classifier_uncertain`, `person_small`, `torso_truncated`, `torso_hidden`, `feet_unknown`, `camera_moved`. Helmet: native 64px MobileNetV3-small crops, threshold 0.853 unchanged; `ppe.classifier.unknown_band: [lo, hi]` optionally turns mid scores into unknown.

**Floor points.** Feet are `visible`, `estimated` (from a full-body height seen in the last 3s) or `unknown`. Unknown never enters a zone and leaves the hysteresis state unchanged.

PPE: ≥70% violating readable observations in 2s, ≥0.6s readable samples, ≥1.5s persistence. Pending evidence restarts after a gap of more than max(0.6s, two sampled frames), so a single missed sample at 3 FPS no longer restarts the evidence. Zones: 1s/60% vote, 0.5s persistence. All numbers: `configs/rules.yaml`.

**Frame skipping.** Every sampled frame is processed while anyone is in view. Once no person has been detected or tracked for 1s, only 2 FPS is processed until someone appears (`sampling.idle_fps`, `idle_after_s`). Skipped samples are not decoded to images, but they still advance the tracker's clock, so a lost ID expires after 5s of real time. Clips repeat frames across idle gaps so they play at real speed. A worker walking in while the scene is idle is first seen up to 0.5s later.

`<out>/<camera_id>/`: `run.json` (unique `run_id`), `events.live.jsonl` (append-only `alert` → `clip` → `episode_end` → `run_end` records, written as they happen), `events.jsonl` (final, written atomically at the end), snapshots, clips, `perf.json`, optional `readings.jsonl` (`--log-readings`). Each event has `snapshot_status`/`clip_status` (`pending`/`ready`/`failed`, with `clip_error`) and `clip_truncated` (`start`/`end` when the video began or ended inside the clip window). An earlier run's files are moved to `archive/<run_id>/`, never overwritten, unless `--overwrite`. Confidence is not a calibrated safety probability.

## Current Windows results

Intel Core Ultra 7 256V, 8 cores/8 threads. `scripts/bench_pipeline.py`: full inference over all six SteelBench clips (81s of video, three of them low light or degraded), repeated runs (3 for iGPU, 2 for CPU), evidence media on, model load/warm-up excluded. Stage times in milliseconds per processed frame, mean / p95:

| Backend, sampled FPS, skipping | Decode | Detect | Track | Helmet | Rules | Media | Processed FPS | Real time | TP/24 | FP | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CPU OpenVINO INT8, 10, skip (default `--cpu-only`)** | 6.5 / 7.4 | 112.9 / 153.3 | 0.8 / 1.3 | 0.9 / 2.2 | 0.1 / 0.2 | 1.5 / 1.5 | **8.1** | 0.91× | 17 | 1 | .708 |
| CPU OpenVINO INT8, 10, no skip | 6.5 / 7.1 | 114.5 / 150.8 | 0.7 / 1.2 | 0.8 / 2.1 | 0.1 / 0.2 | 1.4 / 1.4 | 8.0 | 0.80× | 17 | 1 | .708 |
| CPU OpenVINO INT8, 8, skip (live) | 6.2 / 6.4 | 111.5 / 155.8 | 0.8 / 1.3 | 0.8 / 2.1 | 0.1 / 0.3 | 1.7 / 1.5 | 8.2 | **1.14×** | 17 | 1 | .708 |
| CPU OpenVINO INT8, 6, skip | 6.6 / 7.6 | 111.7 / 154.7 | 0.8 / 1.2 | 0.8 / 2.1 | 0.1 / 0.2 | 1.6 / 1.4 | 8.2 | 1.50× | 17 | 1 | .708 |
| CPU OpenVINO INT8, 5, skip | 7.2 / 8.7 | 117.0 / 154.2 | 0.8 / 1.3 | 0.9 / 2.4 | 0.1 / 0.3 | 1.8 / 2.0 | 7.8 | 1.70× | 16 | 1 | .667 |
| CPU PyTorch, 10, no skip | 7.1 / 7.9 | 327.5 / 401.9 | 1.0 / 1.6 | 1.0 / 2.8 | 0.2 / 0.3 | 1.7 / 1.6 | 2.95 | 0.29× | 17 | 1 | .708 |
| iGPU OpenVINO fp32, 10, no skip | 8.2 / 10.6 | 32.0 / 40.1 | 1.1 / 2.2 | 1.2 / 3.4 | 0.2 / 0.4 | 1.9 / 2.1 | 22.3 | 2.2× | 17 | 1 | .708 |

**CPU-only, the brief's target is met: 8.1 processed FPS against 5, with the same 17 events and 1 false alert as PyTorch.**

- **INT8 on all 8 threads.** OpenVINO's default on this hybrid CPU uses only the 4 performance cores, and INT8 there takes 160ms (`scripts/bench_detector.py`, `eval/results/detector_bench_win32.json`). On the 30-frame check, INT8 on the CPU kept every reference person, head, helmet and vest.
- **Skipping.** These clips have people in view almost throughout, so skipping saved only 11% of frames here. On the person-free low-light clip it saved 83%.
- **Sampled rate.** At 10 FPS the CPU runs at 0.91× real time, which is fine for files. A live camera needs 8 FPS: real time with no lost event, but one low-light event alerts 1.6s later. 5 FPS loses an event.
- **Rejected:** smaller input (640px finds 8 of 33 helmets) and the nano detector (0/6 vests). No live-latency or multi-camera guarantee; two cameras on one CPU process 9.3 FPS in total.

Events from the default `--cpu-only` run (`out/`, scored by `eval/events_eval.py` with one-to-one matching):

| Set | Labels | Alerts | True | False | Dup | Ignored | Missed | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 14 | 12 | 11 | 0 | 0 | 1 | 3 | 1.00 | .786 |
| Additional cameras | 10 | 7 | 6 | 1 | 0 | 0 | 4 | .857 | .60 |

Before this revision the additional cameras gave 2 false alerts. The R5 vest alert, on a person cut by the bottom edge, is now `torso_truncated`. The remaining false alert, R4, is a reflective coverall the detector never boxes as a vest. For each of the 7 misses, `eval/miss_review.py` (results in `eval/results/miss_review.json`) attributes every labelled frame to a stage:

- **Never tracked:** zone Q2, Q3 and L.
- **Tracked too rarely to alert:** zone C, tracked in 25 of 145 frames.
- **Temporal rule:** vest R3, whose longest violating run is 1.1s, under the 1.5s rule.
- **Helmet:** two misses, covered in the Limitations section.

Tracker change: development one-ID people **5→7/12**; additional cameras **3→6/7**, ID changes **7→0**, coverage **53.6→56.9%**. Four of 19 people are never tracked. The tracking report also counts ID reuse across people, tracks matching no person, and missed person-frames. Approximate hand boxes at IoU ≥0.3 are diagnostics, not official MOT metrics. Historical M2 scores differ.

## Verify and limitations

`python -m pytest tests -q` (83 tests); `python scripts/verify_run.py out` (6 cameras, 19 events, 0 failed media, 13 clips flagged `start`-truncated because those alerts fire within the first seconds of 10–15s clips); `python eval/events_eval.py`, adding `--labels labels_heldout` for additional cameras. The evaluator stops on a missing label folder or camera output; `--allow-missing` counts the latter as all-missed. Speed: `python scripts/bench_pipeline.py --help`. Tracker comparisons: `eval/tracker_replay.py --help`. Detector caches are keyed on the SHA-256 of the actual model file (`.pt`, ONNX, CoreML or OpenVINO).

Small/hidden workers, low light, coveralls and floor estimates remain weak spots. New HME labels are scene-level: no person-level accuracy claim. Full-shift calibration, live ingest/backpressure and bounded long-running history remain deployment work. CMOT is skipped at the owner’s request. The HME sample was not rerun (its source video isn't in this checkout); its earlier output is under `out/hme_new/archive/`.

Helmet generalization remains poor: both labelled helmet violations on the additional cameras were missed (0/2), despite 1/1 caught in development. Per-frame review:

- **W:** no head box in 87 of 141 frames, and p(bare) drifts from 0.8–0.95 to 0.06–0.2, so the longest violating run is 0.2s. This is a detector miss plus a classifier error.
- **R1:** 18 readable frames, all scored compliant (p(bare) mostly 0.05–0.55, against the 0.853 threshold). This is a classifier error.

The threshold was not lowered to catch them. Fixing them needs human-reviewed head labels from more cameras, with training and evaluation split by camera, which this revision did not have. No occluder polygons are configured for the sample cameras yet.

Used: [SH17](https://github.com/ahmadmughees/SH17dataset), [SteelBench](https://huggingface.co/datasets/ThinkingHub/PP) (CC BY-NC 4.0), [Hard Hat Workers](https://huggingface.co/datasets/keremberke/hard-hat-detection); model provenance/training scripts committed. Original rewrite: Claude Code; fixes/tests/validation: Codex. No detector retraining this revision.

**Owner must supply:** actual hours spent and a ≤3-minute screen-recording link showing zones, alerts and one failure.
