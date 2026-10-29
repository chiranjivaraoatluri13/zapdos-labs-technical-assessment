# Pipeline and reproduction guide

## What I built

This pipeline takes a CCTV video and produces alerts for missing helmets, missing high-visibility vests and people inside configured restricted zones.

I used SH17 YOLOv8s for detection, ByteTrack for tracking and a MobileNetV3-small classifier for helmet decisions on head crops. ByteTrack's second association pass can recover weaker detections during partial occlusion without adding a separate appearance network. A five-second track buffer and conservative identity matching help connect short gaps, although identity continuity is not guaranteed in crowded scenes.

The rules combine several observations before creating an alert. PPE violations need 1.5 seconds of persistence and a 70% violating vote among readable observations in a two-second window. Zone entries use a one-second window, a 60% vote and 0.5 seconds of persistence. A 60-second cooldown limits repeated alerts for the same person and violation. These settings are in [configs/rules.yaml](../configs/rules.yaml).

For zones, I test the estimated ground-contact point against a floor polygon. Feet can be visible, estimated from a recent full-body observation or unknown. Unknown feet cannot start a zone alert. When the camera view changes enough to invalidate its polygons, the pipeline emits `camera_moved` and pauses zone decisions.

## Setup and the five selected videos

Use Python 3.12 on Windows. From the repository root:

```powershell
python -m pip install -r requirements-cpu.txt
python scripts/download_data.py
python scripts/export_models.py --detector --openvino --int8
```

The downloader fetches the sample videos and SH17 weights. The helmet classifier is included in the repository. The export creates the OpenVINO detector models; it is a setup step rather than something required before every video.

After setup, this runs all five selected videos sequentially using one model load on the tested Intel GPU:

```powershell
python scripts/run_sample.py --all --backend openvino --openvino-device GPU
```

It downloads the exact source videos when they are not already cached, verifies their SHA-256 values, loads the committed zone files and reference images, and runs the existing pipeline with detailed readings and annotated output enabled. It then checks that the resulting event records are complete and their evidence is readable. See [TEST_VIDEOS.md](../TEST_VIDEOS.md) for the five individual commands and known issues. To use the CPU, replace the GPU flags with `--cpu-only`.

Generated results go under `runs/samples/<camera_id>/`. Committed sample outputs are under [out/demo_samples](../out/demo_samples). The original dataset URLs and checksums are in [samples/review_manifest.json](../samples/review_manifest.json). New predictions on another backend or at another sampling rate can differ; the saved outputs are not hard-coded into the runner.

## Any other video or new camera

For an existing configured example:

```powershell
python run.py --video clips/sb_crm_workshop_0109.mp4 --zones zones/sb_crm_workshop_0109.json --cpu-only --save-video --log-readings --out runs/demo
```

On the tested Intel integrated GPU, use `--backend openvino --openvino-device GPU` instead of `--cpu-only`.

For a new camera, create a zone JSON using the supplied files as examples, then save the view against which the zones were drawn:

```powershell
python scripts/zone_reference.py --video clips/new.mp4 --zones zones/new.json
```

Supply those paths to the same `run.py` command. Restricted areas must come from the site's safety policy; the pipeline does not decide which areas should be restricted. The five demonstration polygons are illustrative. Omitting `--zones` runs PPE detection alone.

The default sampling rate is 10 FPS. `--cpu-only` uses the INT8 export when available; otherwise it falls back to PyTorch CPU, which was substantially slower in the recorded tests. Sampling fewer frames is a tradeoff: the recorded five-FPS CPU experiment lost a labelled event. Adaptive idle skipping is configurable; the selected-video runner disables it to reproduce the review settings.

## Outputs and verification

Each camera output contains:

- `events.jsonl`: finalized alerts, timestamps, identities, confidence and evidence paths.
- `events.live.jsonl`: records written as alerts occur and clips become available.
- `snapshots/` and `clips/`: annotated evidence for each person-level alert.
- `annotated.mp4`: full annotated video when `--save-video` is enabled; the sample runner enables it.
- `readings.jsonl`: observations and unknown-state reasons when `--log-readings` is enabled; the sample runner enables it.
- `perf.json` and `run.json`: timing, input settings and the run identifier.
- `sample_provenance.json`: source checksum, URL, dataset revision and verification results when using the sample runner.

Evidence clips normally start three seconds before the recorded violation onset and finish three seconds after the alert. Short source videos can limit this window; affected clips are marked as truncated. An earlier run is archived by default. The committed annotated videos and clips use H.264 review copies; their frame counts match the original pipeline outputs.

```powershell
python scripts/verify_run.py runs/samples
python -m pytest tests -q
```

The latest full test run recorded 93 passes and one skip. The skipped parity test requires the local INT8 detector export. The five selected clips were regenerated through the sample runner and reproduced all 16 alert identities and timestamps. Their 16 snapshots and clips passed verification; 12 clips are marked truncated. Media verification checks records and readability; it does not establish that an alert is correct.

## Performance

The README table comes from the committed six-clip Windows benchmarks, covering 81 seconds of video. Hardware: Intel Core Ultra 7 256V, eight CPU cores/eight threads, with its Intel integrated GPU. Input video was 1920 × 1080; detector input was 736 × 1280. The three listed rows use 10 FPS sampling with idle skipping disabled.

Stage times are mean milliseconds per processed frame. Model loading and warm-up are excluded; evidence generation is included in throughput. These measurements precede the identity and display fixes. The derived ten-FPS stream equivalents are throughput divided by ten, not measured simultaneous-camera capacity. Helmet classification and encoding are additional stages included in throughput.

INT8 CPU inference reached 8.02 processed FPS against 2.95 for PyTorch, retaining the same event totals in that comparison. The integrated GPU reached 22.32 FPS. Those results exceed the brief's five-FPS CPU and ten-FPS GPU targets on the named hardware, but do not establish live ingest latency or capacity for many simultaneous cameras.

A separate 15-second BF Conveyor run on the submission source reached 21.9 FPS with annotated video and evidence enabled. Timing varied across later sample runs; both original and repeated measurements are preserved in [the twelve-candidate review](../eval/results/sample_review_20261028.json). Execution speed is separate from detection correctness.

## Where it is most likely wrong

Small heads, partial workers and machinery occlusion remain the main weaknesses. Both labelled missing-helmet events on the additional evaluation cameras were missed. A worker can also disappear before enough readable evidence accumulates to satisfy the persistence rule. None of the twelve selected-video candidates emitted a missing-helmet alert, so that review does not establish missing-helmet recall.

Vest detection is less reliable on reflective coveralls and unusual clothing. A separate BF Conveyor run produced a likely missing-vest alert on machinery that had been detected as a person. Repeated detections can pass confirmation even when the underlying object is wrong.

Nearby workers can merge or split identities. In the selected `01_single_mill`, the operator appears under a new identity late in the clip and receives another zone/vest pair. Hidden-foot estimates can place someone incorrectly relative to a polygon when machinery hides most of the body. The enlarged head panel makes source pixels easier to inspect; it cannot recover missing detail.

I would check these failures by reviewing alert evidence alongside non-alerted tracks, using detailed readings to identify whether detection, association, classification or persistence failed. Improvements need independently reviewed examples from separate cameras and another event-level comparison. A lower threshold alone would not demonstrate progress.

The footage tested so far is too short to establish the site's two-false-alerts-per-shift target. Real IR performance, long-running live ingest and full-shift reliability are not established. The documented installation steps have not been independently reproduced in a completely fresh environment.

The selected-video table reports emitted alerts. The hand-labelled event table requested by the assessment, including true/false/duplicate/missed counts and median onset-to-alert delay, remains pending annotation of these five videos. A source-time `delay_s` in an event is not itself a ground-truth delay measurement. The separate screen recording must show a run, the alert log filling and one failure in no more than three minutes; annotated output videos do not replace it.

## Models, libraries and assistance

Models and data: [SH17 YOLOv8s](https://github.com/ahmadmughees/SH17dataset); MobileNetV3-small trained on [Hard Hat Workers](https://huggingface.co/datasets/keremberke/hard-hat-detection) and additional SteelBench head crops; [SteelBench CCTV clips](https://huggingface.co/datasets/ThinkingHub/PP), CC BY-NC 4.0. Training/export scripts and model metadata are committed.

Main libraries: Ultralytics, PyTorch, torchvision, OpenCV, OpenVINO, ONNX Runtime, NumPy, SciPy, PyYAML and pytest.

I used ChatGPT 6.1 Sol Light and Cursor for coding assistance and review.

Time spent: 6.5 hours.
