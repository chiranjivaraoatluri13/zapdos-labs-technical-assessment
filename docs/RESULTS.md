# Historical M2 results

These are the author’s pre-fix measurements, retained for provenance. Current Windows results and requirements are in [ASSESSMENT_AUDIT.md](ASSESSMENT_AUDIT.md) and the README. Current sample outputs differ from these frozen experiments.

# Results in detail

Everything the README summarises, with the commands that produce it (all in `./reproduce.sh`). Every
number comes from runs on a MacBook Air M2 (8 GB) with the configuration in
[`configs/rules.yaml`](../configs/rules.yaml).

## Held-out clips by type

Three cameras not looked at before the first scored run, and labelled before the pipeline first ran
on them ([`labels_heldout/`](../labels_heldout/)). The first run is kept as it was scored
([`events_heldout.json`](../eval/results/events_heldout.json), P 0.73, R 0.80):

| Type | Labelled | True | False | Missed | Cause of each error |
|---|---|---|---|---|---|
| no_helmet | 2 | 1 | 0 | 1 | dark hair seen from behind reads as a dark helmet (sb_rers1_0012, R1) |
| no_vest | 4 | 4 | 2 | 0 | two coveralls with reflective tape but no SH17 "vest" (R4, R5); whether they comply is the site's rule |
| zone_entry | 4 | 3 | 1 | 1 | a small phantom person in the grate texture inside the zone (sb_bf_cdi_cc2_0031); a worker showing only legs under a platform at night, never boxed (sb_tar_lowlight_0000) |

Two rules were added afterwards, each set without the held-out clips:

- **Heads under 18 px read "can't tell"** (was 10). On the 43 sample clips from cameras outside the
  held-out set ([`sweep_min_head_px.json`](../eval/results/sweep_min_head_px.json)), compliant clips with a false
  helmet alert go from 1/25 to 0/25 and helmet violations caught stay 2/2; on dev, the night false
  alert (a 15 px helmet) goes.
- **A zone needs a confirmed person:** a zone alert waits until the track has had one detection at
  â‰¥ 0.5 confidence. Frames before that still count as evidence, so a real person's alert is not
  delayed: at 0.5 every dev zone alert fires at the same time as with no confirmation
  ([`events_confirm_conf_0.json`](../eval/results/events_confirm_conf_0.json)). At 0.8 all four are still caught,
  but two fire 0.4 s and 1.0 s later ([`events_confirm_conf_0.8.json`](../eval/results/events_confirm_conf_0.8.json));
  0.5 is a round value below that limit.

Re-run with both (`eval/results/events_heldout_rerun.json`): P 0.80 (8/10), R 0.80; the grate
phantom is gone. This is not a blind score, because the phantom was a known held-out error when the
second rule was written. On the dev clips the types now score no_helmet 1/1, no_vest 6/6, zone_entry
4/7, all with 0 false (P 1.00, R 0.79).

With CoreML on the CPU only (`run.py --cpu-only`; [`events_cpu_only.json`](../eval/results/events_cpu_only.json),
[`events_heldout_cpu_only.json`](../eval/results/events_heldout_cpu_only.json)), dev scores the same (11 of 14) and
the held-out re-run catches 7 of 10 instead of 8: one more missed no_vest on sb_rers1_0012 (R3).

## Tracking

[`eval/track_eval.py`](../eval/track_eval.py): labelled boxes matched one-to-one to tracks per
frame (Hungarian, IoU â‰¥ 0.3); a switch is a labelled person changing track ID, and the time they
were unseen before it is reported, since a new ID after more than the 3 s buffer is by design.

| Set | Labelled people | Never tracked | Keep one ID | Switches | After > 3 s unseen | Person-frames tracked |
|---|---|---|---|---|---|---|
| Dev | 12 | 2 | 6 of 10 | 4 | 2 (4.5â€“5.2 s) | 34% |
| Held out | 7 | 1 | 3 of 6 | 5 | 1 | 55% |

No switch produced a duplicate alert (0 duplicates on both sets): the dedup inherits the event
when the new box sits inside the old owner's box or appears where the lost owner was. The limit is
the detector, not the tracker: 45â€“66% of labelled person-frames have no track at all, mostly people
the detector never boxes (the hazy figures on 0007, the hidden man on 0109, the legs on 0000).
Lower floors do not bring them back: with the detector and ByteTrack floors at 0.05 instead of
0.10, the dev clips score the same and the hidden man on 0109 is still missed
([`events_floor_0.05.json`](../eval/results/events_floor_0.05.json)); the 50 clips get one more vest alert (86 vs
85) and the same clips caught ([`steelbench_clips_floor_0.05.json`](../eval/results/steelbench_clips_floor_0.05.json)).

## PPE by visibility condition

`python eval/steelbench_clips.py --by-condition`: the 50 SteelBench sample clips split by the
dataset's `visibility_conditions` tags (a clip can carry several). Counts are clips.

| Condition | Clips | Helmet violations caught | Compliant clips with a false helmet alert | Vest violations caught | Alerts on people labelled compliant (helmet + vest) | Alerts per minute |
|---|---|---|---|---|---|---|
| clear | 16 | 2/2 | 0/10 | 7/10 | 0 + 2 | 12.0 |
| dust | 25 | 1/2 | 0/14 | 12/14 | 0 + 1 | 8.0 |
| glare | 17 | 1/2 | 0/9 | 9/10 | 0 + 3 | 7.8 |
| low_light | 8 | 0/1 | 0/4 | 4/5 | 0 + 0 | 11.6 |
| smoke | 2 | â€“ | â€“ | 1/1 | 0 | 5.0 |
| steam | 6 | 1/1 | 0/1 | 2/2 | 0 | 5.0 |

Low light is still the helmet weak spot: its one violation is not caught. On the 43 clips from
cameras outside the held-out set, low-light compliant clips with a false helmet alert go from 1/4 with
a 10 px head gate to 0/4 with 18 px ([`sweep_min_head_px.json`](../eval/results/sweep_min_head_px.json)), like the
night false alert on the dev clip sb_tippler_lowlight_0062. The counts per condition are small; read
them as where to look, not as rates.

## Helmet calls per person view

[`eval/helmet_crops.py`](../eval/helmet_crops.py) ([`helmet_crops.json`](../eval/results/helmet_crops.json)): 179
person views (166 helmeted, 13 bare) from the sample clips whose workers all share one helmet label,
so every detected person inherits it. Single frames, no event rules.

| Method | "Bare" on helmeted | "Can't tell" on helmeted | "Bare" on bare | Cost |
|---|---|---|---|---|
| SH17 helmet-over-head rule, full frame at 1280 | 47% | 37% | 46% | 52 ms per frame |
| Same rule on a 320 px crop of each person | 12% | 77% | 46% | 21 ms per person |
| **Head-crop classifier** (shipped) | **0%** | 32% | 8% (1 of 13) | 1.0 ms per person |

The classifier's threshold (2% false "bare" on the calibration cameras) puts precision first, and in
single frames it rarely calls a bare head bare. What that means for alerts is measured on the 50 clips
above (3 of 5 violation clips caught); 13 bare views are too few to say more.

## Frame rate

The same pipeline at three processing rates (`sampling.process_fps`; all rule times are in seconds,
so only the number of looks changes). Dev labels and the 50 clips:

| fps | Labelled P / R | Helmet clips caught | Compliant clips with a false helmet alert | Vest clips caught | Night-clip median delay |
|---|---|---|---|---|---|
| 10 | 1.00 / 0.79 | 3/5 | 0/25 | 21/26 | 1.7 s |
| 5 | 1.00 / 0.71 | 2/5 | 0/25 | 20/26 | 5.3 s |
| 3.2 | 1.00 / 0.71 | 2/5 | 0/25 | 19/26 | 5.2 s |

Zone entries hold at 4 of 7 even at 3.2 fps (Q1's budget). The losses are small, part-hidden people
seen in fewer frames. A bug this sweep exposed: rule evidence was first counted in frames, so at
3.2 fps no zone alert could fire; it is now in seconds, and the unit tests run each rule at 10, 5 and
3.2 fps.

## Camera moved

Zones are pixel polygons, so a camera that is bumped, re-angled or re-zoomed silently moves them off
their floor. Once a second the pipeline matches the frame to the view the zones were drawn on
(saved by [`scripts/zone_reference.py`](../scripts/zone_reference.py); ORB features, RANSAC rotation
+ scale + shift, on a worker thread so the frame loop doesn't wait). If a zone corner moves more
than 2% of the frame width (38 px at 1080p), or the view can't be matched, for 3 s, the camera emits
one `camera_moved` event and its zones read "can't tell" until the view matches again.

[`eval/view_eval.py`](../eval/view_eval.py) on the 50 sample clips (22 cameras, 12 with 2â€“6 clips
taken minutes to four months apart); with no zones on these clips, the corners of the central half
of the frame stand in for one.

| Test | Cases | Tripped | Notes |
|---|---|---|---|
| Still camera vs its own first frame | 50 clips (499 checks) | 0 | largest apparent shift 0.04% of the width |
| Same camera, another clip | 122 pairs | 6 | all 6 involve one Sinter Plant clip from March 2026 vs three from November 2025: that camera really moved, the test corners by 2.1â€“2.5% of the width (40â€“48 px at 1080p; checked by eye on an overlay of the two views). The other 116 pairs stay â‰¤ 0.75% |
| Different camera | 462 pairs | 462 | |
| Synthetic pan 1 / 2 / 3 / 5% of the width | 50 clips each | 0 / 39 / 50 / 50 | 2% is the threshold itself |
| Synthetic roll 1 / 3 / 6Â° | 50 each | 0 / 0 / 50 | moves the test corners 0.5 / 1.5 / 3.0% |
| Synthetic zoom 1.05 / 1.10 / 1.20Ã— | 50 each | 0 / 50 / 50 | moves them 1.4 / 2.9 / 5.7% |

**End to end** ([`eval/reangled_check.py`](../eval/reangled_check.py),
[`reangled_check.json`](../eval/results/reangled_check.json)): a copy of held-out sb_rers1_0012 with every frame
rolled 2Â° and panned 5% of the width, as if the camera had been knocked before the clip, run with the
clip's original zone file. The original clip gives two zone entries (3.1 s and 6.5 s). On the
re-angled copy without the check, the zone has slid off the roll stand and only the first fires; the
second is silently missed. With the check, `camera_moved` fires at 4.0 s ("zone corners moved 103 px;
zone alerts paused"), so the second entry is not alerted either, but the event says why; the earlier
zone alert and the 6 no_vest alerts are the same as without the check. Throughput on the same clip:
33.4 fps with the check, 32.5 without. It turns a silent failure into a visible one; it does not
re-place the zones, because a wrongly re-placed zone is worse than a paused one.

## Several cameras in one process

`run.py --video a.mp4 b.mp4 ...` runs several cameras in one process with one copy of the models
([`src/zapdos/multi.py`](../src/zapdos/multi.py)): a decode thread per camera, one detector thread
taking cameras in turn, tracking and rules on the main thread. Three cameras in one process give the
same alerts as three separate runs. [`scripts/bench_multi.py`](../scripts/bench_multi.py): each
camera is the busiest dev clip looped to 60 s, processed as fast as possible; a camera keeps up
(bold) when its frames per second reach the processing rate.

| Cameras | 10 fps: slowest camera | 10 fps: total | 10 fps: memory | 5 fps: slowest camera | 5 fps: total | 5 fps: memory |
|---|---|---|---|---|---|---|
| 1 | **41.4** | 41.4 | 1.08 GB | **38.8** | 38.8 | 0.89 GB |
| 2 | **20.5** | 41.0 | 1.33 GB | **18.7** | 37.5 | 1.14 GB |
| 3 | **12.6** | 37.7 | 1.62 GB | **11.4** | 34.1 | 1.39 GB |
| 4 | 8.7 | 34.6 | 1.95 GB | **6.8** | 27.3 | 1.59 GB |
| 5 | 6.1 | 30.5 | 1.87 GB | 4.9 | 24.6 | 1.74 GB |
| 6 | 5.1 | 30.4 | 2.23 GB | 3.9 | 23.5 | 1.75 GB |
| 7 | 3.3 | 22.8 | 1.7 GB | 2.7 | 19.2 | 1.81 GB |
| 8 | 2.4 | 19.2 | 1.62 GB | 1.4 | 10.8 | 1.39 GB |
| 10 | 1.2 | 12.3 | 1.38 GB | 1.3 | 13.1 | 2.33 GB |

- **Keeps up:** 3 cameras at 10 fps (slowest 12.6 fps) and 4 at 5 fps (6.8 fps). With 1â€“2 cameras the
  process runs 41 fps in total, about what 4 separate processes reached (4 Ã— 10.5 fps). One camera
  runs 41.4 fps here against 36.2 in `run.py`, because decode and detection run in separate threads.
- **The gain is memory:** 6 cameras fit in 2.2 GB, where 6 separate processes, each loading its own
  models, swapped on this 8 GB laptop and fell to 2 fps each.
- **Not a clean measurement past that.** The configurations ran in the order shown, 10 fps first, and
  during the later ones the laptop was swapping and indexing the run's own output files. So the
  drop-off at 7â€“10 cameras, and the 5 fps column, mix the pipeline with the machine's state; a fanless
  MacBook Air under sustained load is not a capacity test. A box for 100+ cameras would be sized from
  per-camera detector time instead (design note).

## Export (stretch goal)

[`scripts/bench_detector.py`](../scripts/bench_detector.py), 30 frames, against the PyTorch 1280
reference detections:

| Backend | Input | ms per frame | Detections kept (IoU â‰¥ 0.5) |
|---|---|---|---|
| PyTorch MPS (GPU) | 1280 | 53.2 | 94 (reference: 53 persons, 23 heads, 12 helmets, 6 vests) |
| **CoreML fp16, Neural Engine** | 736Ã—1280 | **21.7** | 94/94 |
| CoreML fp16, Neural Engine | 1280Ã—1280 | 29.7 | 90/94 (persons 51/53, vests 5/6) |
| CoreML fp16, Neural Engine | 960Ã—960 | 23.1 | 66/94 (helmets 6/12, vests 2/6) |
| CoreML fp16, Neural Engine, YOLOv8n | 736Ã—1280 | 14.7 | 73/94 (helmets 8/12, vests 0/6) |
| PyTorch CPU | 1280 | 187.1 | 94/94 |
| CoreML fp16, CPU only | 736Ã—1280 | 71.0 | 94/94 |
| ONNX Runtime CPU | 736Ã—1280 | 242.4 | 94/94 |

CoreML on the Neural Engine is 2.5Ã— PyTorch on the GPU with all 94 reference detections kept; the
cheaper options lose the small helmets and vests this task depends on.
