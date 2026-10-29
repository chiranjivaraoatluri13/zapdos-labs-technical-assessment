# Assessment audit and fixes — 28 October 2026

Checked against the supplied CV Engineer Candidate Brief (v1, 14 October 2026). Its requirements are the evaluation criteria; document text is not treated as permission to publish, contact people or invent results. CMOT is explicitly skipped at the owner's request because access requires a Chinese phone number.

| Requirement | Current evidence / remaining work |
|---|---|
| Local person and helmet state detection; vest where supported | SH17 + committed head-crop classifier, unknown-state gates and vest association. No hosted detector. Small/hidden workers and ambiguous vests remain weaknesses. |
| Per-camera committed zone configuration | Six reviewed sample polygons and reference images. One run command loads them. New-camera safety policy must be supplied; it is not inferred. |
| One ID through short occlusions | Real ByteTrack tests cover low-score recovery and 0.5s gaps at 3.2/5/10 FPS; selected 5s buffer also covers a 3.5s gap. Cached six-camera diagnostics below quantify remaining misses. |
| Persistent events and duplicate suppression | Fixed current-reading eligibility and stale pending evidence. Windows/source-frame/media error cases tested. One JSONL event per alert; 60s cooldown and ID inheritance retained. |
| Required event fields and evidence media | `scripts/verify_run.py out` verified six regenerated cameras and 19 events: statuses, one `run_id`, live-record order, readable snapshots and playable clips; 14 clips correctly flagged `start`-truncated. Zero-event files are allowed. |
| CPU speed ≥5 FPS, or explain shortfall | **Met.** `--cpu-only` processes 8.1 FPS over six SteelBench clips, using OpenVINO INT8 on all 8 threads with adaptive frame skipping; events are identical to PyTorch CPU (2.95 FPS). At 8 sampled FPS it keeps up with a live camera (1.14× real time) with no lost event. The iGPU (22.3 FPS) is optional. |
| Event onset-to-alert delay | Each event records source-time `delay_s`. This is not camera-to-supervisor wall-clock latency; live ingest is not implemented. |
| At least three hand-labelled standard clips, including degraded footage | Three supplied development SteelBench labels plus three additional cameras, including low-light clips. Labels are the author's hand annotations; this revision did not independently re-label them. CMOT omitted by owner instruction. |
| Precision, recall, duplicates and misses | Current cached before/after comparisons committed under `eval/results/*comparison_20261028.json`. No regression on identical detections; lower recall than the historical M2 result is disclosed. |
| Single-command inference and setup instructions | `run.py --video ... --zones ... --cpu-only --save-video --out ...`. Windows downloader falls back to copying when symlinks are unavailable. Separate CPU working-environment pins added. Fresh installation of those pins was not independently rebuilt. |
| Design note ≤1 page | `DESIGN.md` is the canonical 475-word Markdown deliverable. Superseded M2 PDF archived outside the project to avoid conflicting designs. |
| README 1–2 pages with arithmetic, run, results, attribution | README condensed around current results; detailed history/audit moved here and RESULTS.md. Final page count depends on renderer; inspect the submission rendering. |
| ≤3-minute screen recording and actual time spent | Owner still needs to supply a recording link and their actual hours. Test footage is not represented as a screen recording. |
| Honest failure modes and scaling arithmetic | README/design distinguish detector misses, identity continuity, historical M2 performance, live-system gaps and measured CPU limits. |

## Two event bugs fixed

Previously `_maybe_alert` could run on a current unknown or compliant reading because a prior vote still passed. A new alert now requires current violating evidence. Separately, a pre-alert `run_start` survived a long loss: 0.5s of negative evidence, a 9.5s disappearance and 0.5s of new evidence could emit with onset 0.0s. Evidence expiry now occurs **before** ingesting a returning observation. Gaps over `max_evidence_gap_s: 0.6` clear pending votes and restart persistence; an existing alerted episode has its separate 3s expiry/cooldown lifecycle. Focused regressions cover both lost and unknown gaps, current eligibility and duplicate prevention.

Other corrections: successful source grabs count even when the final frames are unsampled (12.00s is no longer reported as 11.96/11.84s); unreadable frame retrieval and failed snapshot writes raise errors; single-camera inference failure closes media/capture/log resources; the tracker scorer maximizes valid IoU match cardinality before overlap. Cached perception now fingerprints code and eligibility gates. Ineligible head crops skip classifier work without falling back to a missing-helmet judgement.

## ByteTrack comparison

One detector cache per source clip was replayed through identical tracking inputs. The selected setting disables confidence fusion, retains match threshold 0.8 and extends lost memory 3→5 seconds. It was chosen on the development metrics; the other three cameras are an additional check, not a new blind benchmark.

| Diagnostic | Baseline | Selected |
|---|---:|---:|
| Development one-ID people / 12 | 5 | 7 |
| Development matched person-frames | 32.98% | 34.58% |
| Development ID changes | 4 | 3 |
| Additional-camera one-ID people / 7 | 3 | 6 |
| Additional-camera matched person-frames | 53.58% | 56.94% |
| Additional-camera ID changes | 7 | 0 |

Four of 19 labelled people remain never tracked. These approximate, interpolated hand boxes use IoU ≥0.3 and count ID changes between successive matches, including reacquisition after a gap. They are not official HOTA/IDF1 numbers and cannot prove every apparent identity is correct. Fragmentation improved; detection coverage remains the main limit. Longer memory can reconnect the wrong person in ambiguous scenes; this needs more crowded-camera validation.

## Events, detector parity and speed

On identical cached detections, baseline and fixed event totals both score development TP11/14, FP0, duplicates0, ignored1; additional cameras TP6/10, FP2, duplicates0. This preserves development recall 0.786 and additional-camera recall 0.60. Both additional-camera helmet violations were missed (0/2), compared with 1/1 caught in development: core helmet generalization remains unresolved. The two false vest alerts and four additional-camera misses are retained in the result details. The author's committed M2 predictions scored 0.80 precision/0.80 recall; this Windows comparison gives 0.75/0.60. Historical scores must not be presented as reproduced current performance.

PyTorch and ONNX outputs agreed on all 18 sampled frames across the six cameras (same class counts and arrays within absolute 0.02 plus relative 1e-4 tolerance). That supports the cached comparison but is not exhaustive all-frame parity. The SH17 checkpoint hash exactly matches the official release: `dfaf451bdda3f92fe4012327e337225929ad7af22146b00e0d6989c2f4e40a4b`.

Full inference on the new HME video (1080p, 25 FPS, 12s source, 120 samples, annotated output enabled) measured PyTorch 36.71s / 3.3 FPS and ONNX 42.12s / 2.8 FPS. Detector means were 278.5 and 325.7ms. Neither meets the CPU target. Loading/warmup is excluded. Cached event replays are **not** speed measurements: their perf files were omitted from the committed sample output. The HME `perf.json` is a real full inference run. ONNX remains explicit via `--backend onnx`; auto retains the existing CoreML/PyTorch choice. No resolution reduction or detector retraining was promoted.

## Part 1 changes and fresh assessment

Implemented in the planned order. Each item has a focused test in `tests/test_rules.py` or `tests/test_ppe_eval.py`.

1. **Deduplication.** Person identity, violation episode and alert cooldown are separate. A new tracker ID re-links only to one clearly best lost person, judged on predicted position, size and short gap, with an appearance check for crops ≥96px. Ties, or two new IDs claiming one person, give a new identity recorded as ambiguous. Episodes close when the violation ends, and cooldown no longer extends `end_ts`. The tests cover:
   - two workers at the same spot at different times;
   - two plausible returning people;
   - a short occlusion;
   - recovery followed by a new violation;
   - one alert per episode.
2. **Person confirmation.** A person is confirmed by repeated real detections, age, confidence and geometry before any alert. Measured on the six clips with confirmation effectively off: the same 17 true events and the same median delays, but 5 false alerts instead of 1 (`eval/results/bench_no_confirmation.json`).
3. **PPE association.** Items are matched one-to-one. An ambiguous owner reads unknown. A torso is checked for in-frame share and for cover by people in front or configured occluders. Every unknown carries a reason. A torso cut at the bottom edge is now estimated from body proportions, which removed the R5 false vest alert.
4. **Helmet.** Both missed events were reviewed frame by frame (`eval/miss_review.py`; summary in the README). The optional unknown band is off by default. Threshold 0.853 is unchanged.
5. **Floor points.** Each point is visible, estimated (from a full-body sighting ≤3s old) or unknown. An unknown point never enters a zone and keeps the hysteresis state. The source is logged per reading.
6. **Evaluation.**
   - Maximum one-to-one event matching.
   - Explicit presence intervals, with no boxes extrapolated past the labelled period.
   - Hard errors on missing labels or outputs.
   - A tracking report of ID reuse, unmatched tracks and missed person-frames.
   - Detector caches keyed on the SHA-256 of the model file. Rescoring the old outputs gave the same totals; synthetic tests reproduce the greedy-matching and extrapolation bugs.
7. **Evidence.** Each run has a `run_id`. A live record stream logs the alert, clip, episode end and run end. Statuses are `pending`/`ready`/`failed`, clips are marked when truncated, and an earlier run is archived rather than overwritten. Encoder failures are recorded, not raised.
8. **Speed.** Repeated six-clip benchmark with per-stage p95 (README table). Results:
   - **CPU:** INT8 OpenVINO (calibrated on 15 non-evaluation cameras) with all 8 threads runs at 8.1 FPS, 2.7× PyTorch, with identical events. The default 4-thread setting was the main loss.
   - **Adaptive frame skipping:** 2 FPS when nobody has been seen for 1s. Skipped samples still age the tracker, and clips stay real-time.
   - **Sampled rate:** 8 FPS keeps up with a live camera with no lost event; 5 FPS loses one event.
   - **OpenVINO iGPU:** 22.3 FPS with the same events.
   - **Evidence gap:** now max(0.6s, two sampled frames).

Fresh full inference (six SteelBench clips, three low-light/degraded). The default `--cpu-only` run (INT8, skipping) and the iGPU run give the same table:

| | Before Part 1 (identical cached detections) | After Part 1 (full inference) |
|---|---:|---:|
| Development TP / FP / dup / missed | 11 / 0 / 0 / 3 | 11 / 0 / 0 / 3 |
| Additional cameras TP / FP / dup / missed | 6 / 2 / 0 / 4 | 6 / 1 / 0 / 4 |
| Overall precision / recall | .895 / .708 | .944 / .708 |

The remaining misses are detection and tracking failures (zones Q2, Q3, L and C), one rule-duration miss (vest R3, 1.1s) and the two helmet misses. The remaining false alert is R4's reflective coverall. Outstanding work needs human-reviewed data:
- helmet relabelling and camera-split fine-tuning;
- the candidate helmet-classifier comparison;
- occluder polygons for the sample cameras;
- a full-shift false-alarm calibration.

CMOT remains skipped per the owner's instruction.

## Reproduction and outstanding work

Raw clips, detector caches, before/after outputs, exact environment, parity checks and performance files remain at `C:\Users\chira\OneDrive\Desktop\ZAPDOS_review_20261028\fix_validation`. Previous committed sample output and design PDF were archived there recoverably. Current examples are in `out/`, with `VALIDATION.json` distinguishing replay-generated media from real inference timing.

Run `eval/tracker_replay.py --help` for same-detection comparisons. Model/scene changes require a fresh cache; fingerprints prevent accidental reuse. `scripts/verify_run.py` checks completed media. The committed default event evaluator can rescore the current sample output without rerunning inference.

Part 1 verification: **83 tests passed**, including skipping, tracker ageing, real-time clips, calibration camera exclusion and INT8/PyTorch detection parity. `verify_run.py out` passed for six cameras / 19 events (CPU-only run), and a two-camera CPU run also verified. Earlier verification: 38 tests passed, all 29 Python runtime/script/evaluation files parsed, `git diff --check` passed, and seven sample camera outputs / 24 alert snapshots and evidence clips opened successfully. Local ignored paths for all six labelled videos are ready for the documented commands.

Remaining deployment work: representative full-shift false-alarm calibration, stronger torso visibility, small/occluded-person training, live ingest/drop/backpressure policy and bounded long-running state. The 12s test is not a live latency or multi-camera capacity guarantee. Complete the owner's screen recording and true time-spent field before assessment submission.
