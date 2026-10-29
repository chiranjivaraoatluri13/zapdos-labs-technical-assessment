# Selected videos and one-line tests

These are the five demonstrations selected for submission after reviewing twelve different SteelBench clips. The sample outputs include 16 alerts, 16 annotated snapshots, 16 evidence clips and five complete annotated videos. They are selected demonstrations; true/false/duplicate/missed counts require separate human event labels.

| Selected clip | Source duration | Emitted alerts | Annotated video | Evidence |
|---|---:|---:|---|---|
| 02_single_mill_steam | 6 s | 2: zone entry + no vest | [Video](out/demo_samples/02_single_mill_steam/annotated.mp4) | [Clips](out/demo_samples/02_single_mill_steam/clips) |
| 10_crowded_workshop_alt | 15 s | 7: 1 zone entry + 6 no vest | [Video](out/demo_samples/10_crowded_workshop_alt/annotated.mp4) | [Clips](out/demo_samples/10_crowded_workshop_alt/clips) |
| 12_tippler_low_light | 8 s | 2: zone entry + no vest | [Video](out/demo_samples/12_tippler_low_light/annotated.mp4) | [Clips](out/demo_samples/12_tippler_low_light/clips) |
| 01_single_mill | 15 s | 4: 2 zone entry + 2 no vest | [Video](out/demo_samples/01_single_mill/annotated.mp4) | [Clips](out/demo_samples/01_single_mill/clips) |
| 07_conveyor_group | 15 s | 1: no vest | [Video](out/demo_samples/07_conveyor_group/annotated.mp4) | [Clips](out/demo_samples/07_conveyor_group/clips) |

## Setup once

From the repository root, use the README's Python 3.12 / Windows setup:

```powershell
python -m pip install -r requirements-cpu.txt
python scripts/download_data.py
python scripts/export_models.py --detector --openvino --int8
```

This downloads the SH17 detector and exports CPU INT8 and Intel GPU models. The helmet classifier is committed. Model exports and source videos are generated/downloaded locally. The commands below use the tested Intel integrated GPU; inference is local.

## One line per selected video

```powershell
python scripts/run_sample.py 02_single_mill_steam --backend openvino --openvino-device GPU
```

```powershell
python scripts/run_sample.py 10_crowded_workshop_alt --backend openvino --openvino-device GPU
```

```powershell
python scripts/run_sample.py 12_tippler_low_light --backend openvino --openvino-device GPU
```

```powershell
python scripts/run_sample.py 01_single_mill --backend openvino --openvino-device GPU
```

```powershell
python scripts/run_sample.py 07_conveyor_group --backend openvino --openvino-device GPU
```

Run all five sequentially with one model load:

```powershell
python scripts/run_sample.py --all --backend openvino --openvino-device GPU
```

For CPU execution, replace `--backend openvino --openvino-device GPU` with `--cpu-only`. If the INT8 export is missing, CPU mode falls back to slower PyTorch. Runtime and predictions may differ by backend. `--list` shows the samples without loading models; add `--show` to an individual command for the screen-recording display. `--out runs/my_review` chooses a different output root.

Each run downloads its exact pinned source clip, checks SHA-256, loads the committed zone/reference, and runs the existing pipeline at 10 sampled FPS with idle skipping disabled. It saves the annotated video, detailed readings, finalized/live JSONL events, snapshots, evidence clips, performance, run metadata and source provenance. It then verifies completed records and readable evidence. Dependencies and model exports are the first-time prerequisites above.

New outputs go to `runs/samples/<camera_id>/`. Repeated runs archive earlier output. The committed examples are under `out/demo_samples/<camera_id>/` and use H.264 review copies for playback. Their frame counts and readable evidence have been checked; the original inference media is retained in the local review package.

## Read before interpreting the results

- **01_single_mill has a known identity-continuity problem.** The same operator appears under a new identity late in the clip and receives another zone/vest alert pair. It is included because the owner selected it. Do not claim that this demonstration has zero duplicate alerts.
- The crowded scene has occluded workers and potential identity changes. Its dataset annotations do not give per-person PPE labels, so the alert count is not an accuracy score.
- In 07_conveyor_group, the vest alert is on a worker box, but glare and small distant workers limit detection coverage. The foreground demonstration polygon emitted no zone alert.
- In the low-light clip, the second worker is partly hidden and often unknown. It is not an IR validation. All polygons are illustrative, not approved site restrictions.
- Twelve of the 16 evidence clips are marked truncated at a source boundary. This reflects the available video length and early alerts; it is not missing evidence. Preserve those flags.
- None of the twelve candidate runs emitted a missing-helmet alert. These examples do not establish missing-helmet recall.

The full twelve-candidate measurements and review notes remain in `eval/results/sample_review_20261028.json`. This preserves weaker cases and a zero-alert candidate as well as the selected demonstrations. The original batch timings and previous four-candidate reproduction remain labelled as such; they are not rewritten as a new independent benchmark.

Exact source URLs/checksums are in `samples/review_manifest.json`, polygons/references in `samples/zones/`, and original selected source annotations in `samples/annotations/`. Dataset worker counts and descriptions are not substitutes for independently reviewed event labels.

Source: [SteelBench / ThinkingHub PP](https://huggingface.co/datasets/ThinkingHub/PP), revision `2a927fb86450332942fb5f4ffac8a20700cd539a`, CC BY-NC 4.0. Videos, reference frames and annotations remain subject to that licence. Full source videos are fetched from the pinned dataset rather than committed.

## Three recorded evaluation clips

These are the three development clips in the README event-results table, separate from the five demonstrations above. The existing labels and sample evidence are committed; full source videos are downloaded locally.

Download their source videos and detector weights once:

```powershell
python scripts/download_data.py
```

Complete the dependency/model setup above before inference. Each command below saves annotated video, detailed readings, events, snapshots, clips and timing under `runs/evaluation/<camera_id>/`, using the tested Intel GPU with 10 FPS sampling and idle skipping disabled.

### sb_crm_workshop_0109

[Existing labels](labels/sb_crm_workshop_0109.json) · [Zone configuration](zones/sb_crm_workshop_0109.json) · [Saved alert evidence](out/sb_crm_workshop_0109)

```powershell
python run.py --video clips/sb_crm_workshop_0109.mp4 --zones zones/sb_crm_workshop_0109.json --camera-id sb_crm_workshop_0109 --backend openvino --openvino-device GPU --process-fps 10 --no-idle-skip --save-video --log-readings --out runs/evaluation
```

### sb_red_lrs1_0007

[Existing labels](labels/sb_red_lrs1_0007.json) · [Zone configuration](zones/sb_red_lrs1_0007.json) · [Saved alert evidence](out/sb_red_lrs1_0007)

```powershell
python run.py --video clips/sb_red_lrs1_0007.mp4 --zones zones/sb_red_lrs1_0007.json --camera-id sb_red_lrs1_0007 --backend openvino --openvino-device GPU --process-fps 10 --no-idle-skip --save-video --log-readings --out runs/evaluation
```

### sb_tippler_lowlight_0062

[Existing labels](labels/sb_tippler_lowlight_0062.json) · [Zone configuration](zones/sb_tippler_lowlight_0062.json) · [Saved alert evidence](out/sb_tippler_lowlight_0062)

```powershell
python run.py --video clips/sb_tippler_lowlight_0062.mp4 --zones zones/sb_tippler_lowlight_0062.json --camera-id sb_tippler_lowlight_0062 --backend openvino --openvino-device GPU --process-fps 10 --no-idle-skip --save-video --log-readings --out runs/evaluation
```

For CPU execution, replace `--backend openvino --openvino-device GPU` with `--cpu-only`.

After running all three, verify evidence and calculate a fresh event comparison against the existing annotations:

```powershell
python scripts/verify_run.py runs/evaluation
python eval/events_eval.py --labels labels --out runs/evaluation --results runs/evaluation/event_scores.json
```

The recorded README scores predate the later identity/display fixes; a fresh run may differ. Annotation provenance and independent human verification remain as described in the README. This adds documented evaluation commands; `scripts/run_sample.py --all` continues to select the five demonstration clips.
