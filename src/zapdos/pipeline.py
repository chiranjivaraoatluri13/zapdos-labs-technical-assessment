"""decode -> detect -> track -> helmet -> rules (PPE + zones + events) -> media, with per-stage timing.

Outputs per camera, under out/<camera_id>/:
  events.live.jsonl - append-only, written as things happen: an "alert" record the moment an alert
                      fires (clip_status "pending", end_ts provisional), a "clip" record when its clip
                      is ready or failed, an "episode_end" record when the violation ends, "run_end".
  events.jsonl      - written once, atomically, when the run closes: one finalised line per alert.
  run.json, perf.json, snapshots/, clips/, and readings.jsonl when output.readings_log is on.
An earlier run in the same folder is moved to archive/<its run_id>/ unless output.overwrite is set.
"""
import json
import math
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .detect import PPEDetector, available_device, detector_model_path
from .events import EventEngine, Observation
from .helmet import HelmetClassifier
from .media import MediaRecorder, draw
from .ppe import decide, helmet_threshold, measure
from .track import PersonConfirmation, PersonTracker, appearance
from .view import ViewCheck
from .zones import ZoneSet

STAGES = ("decode", "detect", "track", "helmet", "rules", "media")
RUN_FILES = ("events.jsonl", "events.live.jsonl", "readings.jsonl", "perf.json", "run.json", "annotated.mp4")


def open_video(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, src_fps, size


def sample_frames(cap, src_fps, fps, stats=None, want=None):
    """Yields (frame_idx, ts, frame, decode_ms) at `fps`. Every source frame is grabbed (H.264 needs
    them to decode the next one) but only the sampled ones are converted to images; decode_ms covers
    both. want(ts) -> False skips a sample: it is yielded with frame None and never converted."""
    if not math.isfinite(src_fps) or not math.isfinite(fps) or src_fps <= 0 or fps <= 0:
        raise ValueError("source and processing FPS must be positive and finite")
    frame_idx, next_ts, acc = -1, 0.0, 0.0
    while True:
        t0 = time.perf_counter()
        if not cap.grab():
            return
        frame_idx += 1
        if stats is not None:
            stats["source_frames"] = frame_idx + 1
        ts = frame_idx / src_fps
        if ts + 1e-6 < next_ts:
            acc += time.perf_counter() - t0
            continue
        next_ts += 1.0 / fps
        if want is not None and not want(ts):
            ms = (acc + time.perf_counter() - t0) * 1000
            acc = 0.0
            yield frame_idx, ts, None, ms
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            raise OSError(f"cannot retrieve source frame {frame_idx}")
        ms = (acc + time.perf_counter() - t0) * 1000
        acc = 0.0
        yield frame_idx, ts, frame, ms


def build_models(cfg):
    pcfg = cfg["ppe"]
    cfg["detector"]["device"] = available_device(cfg["detector"]["device"])
    detector = PPEDetector(cfg["detector"])
    helmet_clf = (HelmetClassifier(pcfg["classifier"]["weights"], cfg["detector"]["device"])
                  if pcfg["helmet_method"] == "classifier" else None)
    return detector, helmet_clf


def observe(tracks, meas, readings, zones, ts=0.0, confirm=None, descs=None):
    """Per-person observations for the event engine, plus the overlay records for drawing.
    confirm: track.PersonConfirmation (None = every track counts as confirmed); descs: appearance
    descriptors per track or None."""
    zones.at(ts)
    observations, people = [], []
    descs = descs if descs is not None else [None] * len(tracks)
    for (x1, y1, x2, y2, tid, conf), m, reading, desc in zip(tracks, meas, readings, descs):
        tid, box = int(tid), [float(x1), float(y1), float(x2), float(y2)]
        confirmed = confirm.update(tid, box, float(conf), ts) if confirm is not None else True
        zone_read, foot = zones.read(tid, box, conf)
        r = {(k, None): v for k, v in reading.items()}
        r.update({("zone_entry", z): v for z, v in zone_read.items()})
        observations.append(Observation(tid, box, r, confirmed=confirmed, desc=desc))
        flags = {k: v[0] for k, v in reading.items()}
        inside = [v[0] for v in zone_read.values() if v[0] is not None]
        flags["zone_entry"] = any(inside) if inside else None
        people.append({"tid": tid, "box": box, "flags": flags, "foot": foot, "head": m["head"],
                       "confirmed": confirmed, "conf": float(conf), "readings": r})
    return observations, people


def _archive_previous(out, overwrite):
    """Move (or with overwrite, delete) an earlier run's files so nothing is silently replaced."""
    old = [p for p in (*(out / f for f in RUN_FILES), *out.glob("snapshots/*.jpg"), *out.glob("clips/*.mp4"))
           if p.exists()]
    if not old:
        return None
    if overwrite:
        for p in old:
            p.unlink()
        return None
    try:
        old_id = json.loads((out / "run.json").read_text())["run_id"]
    except (OSError, ValueError, KeyError):
        newest = max(p.stat().st_mtime for p in old)
        old_id = "untracked-" + datetime.fromtimestamp(newest).strftime("%Y%m%dT%H%M%S")
    dest = out / "archive" / old_id
    for p in old:
        target = dest / p.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))
    return dest


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


class Camera:
    """One camera's state: tracker, zones, event engine, media and stage timings. The models are passed
    in so that several cameras can share one copy (multi.py)."""

    def __init__(self, video, zones_path, out_root, cfg, models, camera_id=None, save_video=False):
        self.cap, self.src_fps, self.size = open_video(video)
        self.fps = min(cfg["sampling"]["process_fps"], self.src_fps)
        # adaptive sampling: with nobody seen for idle_after_s, process only idle_fps of the samples
        idle = cfg["sampling"].get("idle_fps")
        self.idle_fps = min(idle, self.fps) if idle else None
        self.idle_after_s = cfg["sampling"].get("idle_after_s", 1.0)
        self.person_conf = cfg["tracker"]["track_high_thresh"]
        self.last_person_ts, self.next_idle_ts, self.skipped = -math.inf, 0.0, 0
        self.zones = ZoneSet(zones_path, self.size, cfg["zones"], self.fps)
        self.camera_id = camera_id or self.zones.camera_id or Path(video).stem
        self.out = Path(out_root) / self.camera_id
        self.out.mkdir(parents=True, exist_ok=True)
        archived = _archive_previous(self.out, cfg["output"].get("overwrite", False))
        if archived:
            print(f"{self.camera_id}: previous run moved to {archived}")
        (self.out / "snapshots").mkdir(exist_ok=True)
        (self.out / "clips").mkdir(exist_ok=True)
        self.run_id = new_run_id()

        self.cfg, self.pcfg = cfg, cfg["ppe"]
        self.detector, self.helmet_clf = models
        self.tracker = PersonTracker(cfg["tracker"], self.fps)
        self.confirm = PersonConfirmation(cfg["tracker"]["confirm"])
        self.app_px = cfg["events"]["identity"]["appearance_min_px"]
        self.thr = helmet_threshold(self.pcfg, self.helmet_clf)
        self.engine = EventEngine(cfg["events"], self.camera_id, self.fps)
        self._lock = threading.Lock()  # the clip encoder thread writes "clip" records too
        self.live = open(self.out / "events.live.jsonl", "w")
        self.readings_log = open(self.out / "readings.jsonl", "w") if cfg["output"].get("readings_log") else None
        self.media = MediaRecorder(self.out, cfg["output"], self.fps, self.zones.polygons, self.size, save_video,
                                   on_clip=self._clip_done, min_fps=self.idle_fps)
        corners = [p for poly in self.zones.polygons.values() for p in poly.reshape(-1, 2)]
        self.view = ViewCheck(cfg["view"], corners, self.size, self.zones.reference) if corners else None
        self.moved_ev = None
        self.ms = {s: [] for s in STAGES}
        self.n_src = 0
        self.source_stats = {"source_frames": 0}
        self.tag = ""  # alert-line prefix; the multi-camera runner sets the camera id
        (self.out / "run.json").write_text(json.dumps({
            "run_id": self.run_id, "camera_id": self.camera_id, "video": str(video),
            "zones": str(zones_path) if zones_path else None, "started": datetime.now(timezone.utc).isoformat(),
            "process_fps": self.fps, "idle_fps": self.idle_fps,
            "detector_model": str(detector_model_path(cfg["detector"]))}, indent=2))

    def frames(self):
        return sample_frames(self.cap, self.src_fps, self.fps, self.source_stats,
                             self._want if self.idle_fps else None)

    def _want(self, ts):
        """Process this sample? Always while someone was seen in the last idle_after_s, else at idle_fps."""
        return ts - self.last_person_ts < self.idle_after_s - 1e-6 or ts + 1e-6 >= self.next_idle_ts

    def _record(self, kind, **fields):
        with self._lock:
            if self.live.closed:
                return
            self.live.write(json.dumps({"record": kind, "run_id": self.run_id, "camera_id": self.camera_id,
                                        **fields}) + "\n")
            self.live.flush()

    def _clip_done(self, ev):
        self._record("clip", event_id=ev.event_id, clip=ev.clip, clip_status=ev.clip_status,
                     clip_start_ts=ev.clip_start_ts, clip_end_ts=ev.clip_end_ts,
                     clip_truncated=ev.clip_truncated, **({"clip_error": ev.clip_error} if ev.clip_error else {}))

    def _episode_end(self, ev):
        self._record("episode_end", event_id=ev.event_id, end_ts=round(ev.end_ts, 3),
                     suppressed_episodes=ev.to_json()["suppressed_episodes"])

    def step(self, frame_idx, ts, frame, decode_ms, det=None, detect_ms=None, quiet=False):
        """Process one sampled frame; `det` may come from a shared detector thread. Returns the overlay
        and this frame's time in ms from decode to media (perf.json's pipeline_ms_per_frame leaves
        media out). A skipped sample (frame None) only advances the tracker's clock, so lost tracks
        still expire after track_buffer_s of real time; it returns (None, decode_ms)."""
        if frame is None:
            self.tracker.tick()
            self.skipped += 1
            self.n_src = frame_idx + 1
            return None, decode_ms
        t1 = time.perf_counter()
        if det is None:
            det = self.detector(frame)
            detect_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        tracks = self.tracker(det.persons)
        if len(tracks) or (len(det.persons) and float(np.max(det.persons.conf)) >= self.person_conf):
            self.last_person_ts = ts
        if self.idle_fps:
            self.next_idle_ts = ts + 1.0 / self.idle_fps
        descs = [appearance(frame, t[:4], self.app_px) for t in tracks]
        t3 = time.perf_counter()
        boxes = tracks[:, :4]
        meas = measure(boxes, det, self.pcfg, frame, self.helmet_clf, self.zones.occluders)
        t3b = time.perf_counter()
        alerts = self._check_view(frame_idx, ts, frame)
        observations, people = observe(tracks, meas, decide(meas, boxes, self.pcfg, self.thr), self.zones,
                                       ts, self.confirm, descs)
        alerts += self.engine.step(frame_idx, ts, observations)
        t4 = time.perf_counter()
        for ev in self.engine.pop_closed():
            self._episode_end(ev)
        if self.readings_log is not None:
            self._log_readings(frame_idx, ts, people, meas)

        hidden = self.tracker.hidden(int(round(self.cfg["output"]["hidden_s"] * self.fps)))
        overlay = {"ts": ts, "people": [{k: p[k] for k in ("tid", "box", "flags", "foot", "head")} for p in people]
                   + [{"tid": int(h[4]), "box": list(h[:4]), "flags": {}, "hidden": True} for h in hidden]}
        if self.zones.suspended:
            overlay["banner"] = "CAMERA MOVED: zones paused"
        for ev in alerts:
            ev.run_id = self.run_id
            self.media.on_alert(ev, frame, overlay)
            self._record("alert", **ev.to_json())
            if not quiet:
                who = f"track #{ev.track_id}" if ev.track_id >= 0 else ev.note
                zone = f" zone={ev.zone_id}" if ev.zone_id else ""
                print(f"[ALERT] {self.tag}t={ev.alert_ts:6.2f}s {ev.type:<10} "
                      f"{who}{zone} (started {ev.start_ts:.2f}s, delay {ev.delay_s:.2f}s, "
                      f"conf {ev.confidence:.2f})")
        self.media.push(frame, overlay)
        t5 = time.perf_counter()
        self.n_src = frame_idx + 1
        self.ms["decode"].append(decode_ms)
        self.ms["detect"].append(detect_ms)
        for s, a, b in (("track", t2, t3), ("helmet", t3, t3b), ("rules", t3b, t4), ("media", t4, t5)):
            self.ms[s].append((b - a) * 1000)
        return overlay, decode_ms + detect_ms + (t5 - t2) * 1000

    def _log_readings(self, frame_idx, ts, people, meas):
        if not people:  # a frame with nobody tracked still happened
            self.readings_log.write(json.dumps({"frame": frame_idx, "ts": round(ts, 3), "track_id": None}) + "\n")
        for p, m in zip(people, meas):
            self.readings_log.write(json.dumps({
                "frame": frame_idx, "ts": round(ts, 3), "track_id": p["tid"], "person_id": self.engine.person_of(p["tid"]),
                "box": [round(v, 1) for v in p["box"]], "conf": round(p["conf"], 3), "confirmed": p["confirmed"],
                "head": [round(v, 1) for v in m["head"]], "head_src": m["head_src"],
                "p_bare": None if m["p_bare"] is None else round(m["p_bare"], 4), "ambiguous": m.get("ambiguous", []),
                "torso_cover": m.get("torso_cover"), "foot": None if p["foot"] is None else
                [None if p["foot"][0] is None else [round(float(v), 1) for v in p["foot"][0]], p["foot"][1]],
                "readings": {f"{k[0]}:{k[1]}" if k[1] else k[0]: [r[0], round(r[1], 3), r[2] if len(r) > 2 else None]
                             for k, r in p["readings"].items()}}) + "\n")

    def _check_view(self, frame_idx, ts, frame):
        """One camera_moved event per stretch of time the view doesn't match the zones' reference; zones
        read "can't tell" meanwhile."""
        moved = self.view.update(ts, frame) if self.view else None
        if self.moved_ev is not None:
            self.moved_ev.end_ts = ts
        if moved is None:
            return []
        self.zones.suspended = moved
        if not moved:
            if self.moved_ev is not None:
                self.moved_ev.episode_open = False
                self._episode_end(self.moved_ev)
            self.moved_ev = None
            return []
        shift = self.view.shift
        note = ("view no longer matches the zone reference" if math.isinf(shift) else
                f"zone corners moved {shift * self.size[0]:.0f} px") + "; zone alerts paused"
        self.moved_ev = self.engine.camera_event("camera_moved", self.view.since, round(self.view.since * self.src_fps),
                                                 ts, frame_idx, self.size, note)
        return [self.moved_ev]

    def close(self, wall0):
        """Finish clips, write the final events.jsonl and perf.json; returns (events, perf)."""
        try:
            self.media.close()
        finally:
            if self.view:
                self.view.pool.shutdown(wait=True)
            self.cap.release()
            if self.readings_log is not None:
                self.readings_log.close()
            self._record("run_end", events=len(self.engine.events))
            with self._lock:
                self.live.close()
        wall = time.perf_counter() - wall0
        with open(self.out / "events.jsonl.tmp", "w") as f:  # one finalised line per alert
            for ev in self.engine.events:
                f.write(json.dumps(ev.to_json()) + "\n")
        os.replace(self.out / "events.jsonl.tmp", self.out / "events.jsonl")
        perf = _perf(self.ms, wall, self.source_stats["source_frames"], self.src_fps, self.size, self.fps, self.cfg["detector"],
                     self.detector, self.helmet_clf)
        perf["run_id"] = self.run_id
        perf["idle_fps"], perf["frames_skipped"] = self.idle_fps, self.skipped
        with open(self.out / "perf.json", "w") as f:
            json.dump(perf, f, indent=2)
        return self.engine.events, perf


def run(video, zones_path, out_root, cfg, camera_id=None, save_video=False, models=None, show=False):
    cam = Camera(video, zones_path, out_root, cfg, models or build_models(cfg), camera_id, save_video)
    wall0 = time.perf_counter()
    try:
        for frame_idx, ts, frame, decode_ms in cam.frames():
            overlay, frame_ms = cam.step(frame_idx, ts, frame, decode_ms)
            if show and overlay is not None and not _show(frame, overlay, cam.zones.polygons, cam.media.log, cfg["output"]["video_width"],
                                  frame_ms, ts, wall0, f"{cam.fps:g} of {cam.src_fps:g} fps"):
                break
    finally:
        if show:
            cv2.destroyAllWindows()
        events, perf = cam.close(wall0)
    _print_summary(cam.camera_id, events, perf, cam.out)
    return events, perf


def _show(frame, overlay, zones, log, width, frame_ms, ts, wall0, rate):
    """Live view at the video's own pace (not included in the stage timings). False = quit."""
    scale = width / frame.shape[1]
    img = draw(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), overlay, zones,
               scale, log=log)
    hud = f"frame {frame_ms:.0f} ms incl. media = {1000 / frame_ms:.0f} fps capacity | processing {rate} | q quits"
    cv2.putText(img, hud, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(img, hud, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    ahead_s = ts - (time.perf_counter() - wall0)
    if ahead_s > 0:
        time.sleep(ahead_s)
    cv2.imshow("zapdos: live", img)
    return cv2.waitKey(1) & 0xFF != ord("q")


def _perf(ms, wall, n_src, src_fps, size, fps, dcfg, detector, helmet_clf):
    n = len(ms["detect"])
    stages = {s: {"mean_ms": round(float(np.mean(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
              for s, v in ms.items() if v}
    per_frame = sum(stages.get(s, {}).get("mean_ms", 0) for s in ("decode", "detect", "track", "helmet", "rules"))
    return {
        "input_res": f"{size[0]}x{size[1]}", "source_fps": src_fps, "process_fps": fps,
        "detector": f"{Path(getattr(detector, 'model_path', None) or dcfg['weights']).name} on {detector.backend}",
        "helmet_classifier": helmet_clf.backend if helmet_clf else None,
        "frames_decoded": n_src, "frames_processed": n, "wall_s": round(wall, 2),
        "video_s": round(n_src / src_fps, 2),
        "throughput_fps": round(n / wall, 1),            # processed frames per wall-second, all stages
        "pipeline_ms_per_frame": round(per_frame, 1),     # decode to rules; media (snapshots, clips) is in stages
        "streams_at_process_fps": round(1000 / per_frame / fps, 2) if per_frame else 0,
        "stream_capacity_basis": "stage-time estimate excluding media; not measured multi-camera capacity",
        "source_coverage": round((n_src / src_fps) / wall, 3) if wall else 0,
        "stages": stages,
    }


def _print_summary(camera_id, events, perf, out):
    s = perf["stages"]
    print(f"\n{camera_id}: {len(events)} alerts -> {out / 'events.jsonl'}")
    print(f"  {perf['frames_processed']} frames at {perf['process_fps']} fps from {perf['video_s']} s of video "
          f"in {perf['wall_s']} s  ({perf['throughput_fps']} fps end to end)")
    print("  ms/frame  " + "  ".join(f"{k} {v['mean_ms']}" for k, v in s.items()))
    print(f"  pipeline {perf['pipeline_ms_per_frame']} ms/frame -> "
          f"{perf['streams_at_process_fps']} streams at {perf['process_fps']} fps")
