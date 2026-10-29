"""Runtime regressions for real tracker recovery, sampler accounting and PPE abstention."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from ultralytics.engine.results import Boxes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.pipeline import sample_frames
from zapdos.ppe import measure, decide
from zapdos.track import PersonTracker

CFG = yaml.safe_load((ROOT / "configs/rules.yaml").read_text())


class Capture:
    def __init__(self, count, broken=False):
        self.index, self.count, self.broken = -1, count, broken

    def grab(self):
        self.index += 1
        return self.index < self.count

    def retrieve(self):
        return (False, None) if self.broken else (True, np.zeros((16, 16, 3), np.uint8))


def test_sampler_counts_unsampled_tail_frames():
    stats = {}
    frames = list(sample_frames(Capture(300), 25, 5, stats))
    assert len(frames) == 60
    assert frames[-1][0] == 295
    assert stats["source_frames"] == 300


def test_sampler_surfaces_retrieve_failure():
    with pytest.raises(OSError, match="source frame 0"):
        list(sample_frames(Capture(1, broken=True), 25, 5))


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_sampler_rejects_invalid_sampling_rates(fps):
    with pytest.raises(ValueError, match="FPS"):
        list(sample_frames(Capture(1), 25, fps))


def persons(rows):
    return Boxes(np.asarray(rows, np.float32).reshape(-1, 6), (1080, 1920))


@pytest.mark.parametrize("fps", [3.2, 5, 10])
def test_bytetrack_keeps_id_after_short_occlusion(fps):
    tracker = PersonTracker(CFG["tracker"], fps)
    row = [100, 100, 160, 280, .9, 0]
    first = tracker(persons([row]))[0, 4]
    for _ in range(max(1, int(.5 * fps))):
        assert not len(tracker(persons([])))
    recovered = tracker(persons([row]))
    assert len(recovered) == 1 and recovered[0, 4] == first


def test_bytetrack_uses_low_confidence_recovery_pool():
    tracker = PersonTracker(CFG["tracker"], 10)
    first = tracker(persons([[100, 100, 160, 280, .9, 0]]))[0, 4]
    recovered = tracker(persons([[101, 100, 161, 280, .15, 0]]))
    assert len(recovered) == 1 and recovered[0, 4] == first


def test_bytetrack_keeps_id_during_dev_selected_longer_buffer():
    tracker = PersonTracker(CFG["tracker"], 10)
    row = [100, 100, 160, 280, .9, 0]
    first = tracker(persons([row]))[0, 4]
    for _ in range(35):
        tracker(persons([]))
    recovered = tracker(persons([row]))
    assert len(recovered) == 1 and recovered[0, 4] == first


def test_bytetrack_keeps_two_neighbours_separate():
    tracker = PersonTracker(CFG["tracker"], 10)
    rows = [[100, 100, 160, 280, .9, 0], [180, 100, 240, 280, .9, 0]]
    first = tracker(persons(rows))[:, 4].tolist()
    for i in range(1, 10):
        moved = [[x1 + i, y1, x2 + i, y2, conf, cls] for x1, y1, x2, y2, conf, cls in rows]
        assert tracker(persons(moved))[:, 4].tolist() == first
    assert first[0] != first[1]


def test_ineligible_head_crops_are_not_classified_or_called_bare():
    class Classifier:
        def p_bare(self, frame, heads):
            assert heads == []  # missing head detections: skip expensive inference
            return []

    cfg = copy.deepcopy(CFG["ppe"])
    empty = np.empty((0, 5), np.float32)
    det = SimpleNamespace(heads=empty, helmets=empty, vests=empty)
    boxes = np.array([[100, 100, 160, 300]], np.float32)
    evidence = measure(boxes, det, cfg, np.zeros((400, 400, 3), np.uint8), Classifier())
    assert evidence[0]["p_bare"] is None
    assert decide(evidence, boxes, cfg, .8)[0]["no_helmet"][0] is None


def test_failed_inference_releases_camera_and_media(tmp_path, monkeypatch):
    import cv2
    import zapdos.pipeline as pipeline

    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 128))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((128, 160, 3), np.uint8))
    writer.release()
    cameras = []
    original = pipeline.Camera

    class RecordingCamera(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            cameras.append(self)

    class BrokenDetector:
        backend = "fake"
        def __call__(self, frame):
            raise RuntimeError("injected detector failure")

    monkeypatch.setattr(pipeline, "Camera", RecordingCamera)
    with pytest.raises(RuntimeError, match="injected detector failure"):
        pipeline.run(video, None, tmp_path / "out", copy.deepcopy(CFG), models=(BrokenDetector(), None))
    assert not cameras[0].cap.isOpened()
    assert cameras[0].live.closed
    assert cameras[0].media.pool._shutdown


class CountingCapture(Capture):
    retrieved = 0

    def retrieve(self):
        self.retrieved += 1
        return super().retrieve()


def test_sampler_skips_unwanted_samples_without_converting_them():
    cap = CountingCapture(100)
    out = list(sample_frames(cap, 25, 5, want=lambda ts: ts < 1.0))
    assert len(out) == 20  # the sample grid is unchanged
    assert [f is None for _, _, f, _ in out] == [False] * 5 + [True] * 15
    assert cap.retrieved == 5


def test_skipped_samples_still_age_lost_tracks_in_real_time():
    row = [100, 100, 160, 280, .9, 0]
    for ticks, same in ((35, True), (60, False)):  # 3.5 s and 6 s at 10 fps; buffer 5 s
        tracker = PersonTracker(CFG["tracker"], 10)
        first = tracker(persons([row]))[0, 4]
        for _ in range(ticks):
            tracker.tick()
        again = tracker(persons([row]))
        assert (len(again) == 1 and again[0, 4] == first) == same


def _video(path, n, fps=10, size=(160, 128)):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for i in range(n):
        w.write(np.zeros((size[1], size[0], 3), np.uint8))
    w.release()
    return path


class TimedDetector:
    """One person in frame only between `on` and `off` seconds; `now` is set by TimedCamera."""
    backend, model_path = "fake", None

    def __init__(self, on, off):
        self.on, self.off, self.now, self.calls = on, off, None, []

    def __call__(self, frame):
        from zapdos.detect import Detections
        ts = self.now
        self.calls.append(ts)
        rows = [[40, 20, 70, 110, .9, 0]] if self.on <= ts < self.off else []
        empty = np.zeros((0, 5), np.float32)
        return Detections(persons(rows), empty, empty, empty)


def _camera_cfg(idle_fps):
    cfg = copy.deepcopy(CFG)
    cfg["sampling"].update(process_fps=10, idle_fps=idle_fps, idle_after_s=1.0)
    cfg["ppe"]["helmet_method"] = "detector"
    return cfg


def _timed_run(monkeypatch, tmp_path, det, n, idle_fps):
    import zapdos.pipeline as pipeline

    class TimedCamera(pipeline.Camera):
        def step(self, frame_idx, ts, frame, *a, **kw):
            det.now = ts
            return super().step(frame_idx, ts, frame, *a, **kw)

    monkeypatch.setattr(pipeline, "Camera", TimedCamera)
    return pipeline.run(_video(tmp_path / "v.mp4", n), None, tmp_path / "out", _camera_cfg(idle_fps),
                        models=(det, None))[1]


def test_adaptive_sampling_processes_full_rate_only_around_people(monkeypatch, tmp_path):
    det = TimedDetector(3.0, 5.0)
    perf = _timed_run(monkeypatch, tmp_path, det, 100, 2)  # 10 s at 10 fps
    calls = np.array(det.calls)
    idle = calls[(calls < 3.0)]
    assert np.allclose(np.diff(idle), 0.5)              # nobody yet: 2 fps
    busy = calls[(calls >= 3.0) & (calls < 6.0)]         # person 3-5 s, then idle_after_s of full rate
    assert len(busy) >= 28 and np.allclose(np.diff(busy), 0.1)
    assert perf["frames_processed"] + perf["frames_skipped"] == 100
    assert perf["frames_skipped"] > 40


def test_idle_skip_off_processes_every_sample(monkeypatch, tmp_path):
    det = TimedDetector(3.0, 5.0)
    perf = _timed_run(monkeypatch, tmp_path, det, 50, None)
    assert len(det.calls) == 50 and perf["frames_skipped"] == 0


def test_clips_play_in_real_time_across_idle_gaps(tmp_path):
    import cv2
    from zapdos.events import Event
    from zapdos.media import MediaRecorder
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "clips").mkdir()
    out_cfg = dict(CFG["output"], clip_width=160)
    media = MediaRecorder(tmp_path, out_cfg, 10, {}, (160, 128), min_fps=2)
    frame = np.zeros((128, 160, 3), np.uint8)
    ts_list = [0.0, 0.5, 1.0, 1.5] + [round(2.0 + 0.1 * i, 1) for i in range(61)]  # idle, then full rate
    # pre-roll wanted from 0.3 s; the nearest frame is the 0.5 s idle sample: a gap, not a truncated clip
    ev = Event("c-001", "c", "no_helmet", 1, None, start_ts=3.3, first_frame=33, alert_ts=4.8, alert_frame=48,
               end_ts=4.8, confidence=1.0, vote=1.0, bbox=[0, 0, 10, 10])
    for ts in ts_list:
        overlay = {"ts": ts, "people": []}
        if ts == 4.8:
            media.on_alert(ev, frame, overlay)
        media.push(frame, overlay)
    media.close()
    assert ev.clip_status == "ready" and ev.clip_truncated == []
    assert ev.clip_start_ts == 0.5 and ev.clip_end_ts == 7.8
    cap = cv2.VideoCapture(str(tmp_path / ev.clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == round((ev.clip_end_ts - ev.clip_start_ts) * 10) + 1  # one output frame per 0.1 s


def test_int8_calibration_excludes_every_evaluation_camera():
    sys.path.insert(0, str(ROOT / "scripts"))
    from download_data import LABELLED
    from export_models import _camera
    held = {_camera(c) for c in LABELLED.values()}
    assert _camera("clip_CRM_WORK_SHOP_20260403_000215_0009") in held  # same camera, other day and case
    assert _camera("clip_RERS_2_RERS-2_20251212_000040_0001") not in held
    assert _camera("clip_Tippler_3&4_19.1-19.2TEND_20251205_000258_0011") in held


INT8 = ROOT / "weights/sh17/yolo8s_736x1280_int8_openvino_model"


@pytest.mark.skipif(not INT8.is_dir(), reason="INT8 OpenVINO detector not exported")
def test_openvino_int8_cpu_matches_pytorch_detections():
    pytest.importorskip("openvino")
    import cv2
    from zapdos.detect import PPEDetector
    cap = cv2.VideoCapture(str(ROOT / "clips/sb_crm_workshop_0109.mp4"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip("labelled clip not downloaded")
    dcfg = copy.deepcopy(CFG["detector"])
    ref = PPEDetector(dict(dcfg, backend="torch", device="cpu"))(frame)
    ov = PPEDetector(dict(dcfg, backend="openvino", openvino_device="CPU"))
    assert ov.backend == "openvino-cpu-int8"
    got = ov(frame)
    for name in ("heads", "helmets", "vests"):
        assert len(getattr(got, name)) == len(getattr(ref, name)), name
    strong = lambda b: int((np.asarray(b.conf) >= .5).sum())  # noqa: E731
    assert strong(got.persons) == strong(ref.persons)


def test_tracking_score_does_not_sacrifice_valid_matches(monkeypatch):
    sys.path.insert(0, str(ROOT / "eval"))
    import track_eval
    overlaps = {(0, 1000): .30, (0, 2000): .29, (100, 1000): .99, (100, 2000): .30}
    monkeypatch.setattr(track_eval, "iou", lambda a, b: overlaps[(int(a[0]), int(b[0]))])
    people = {"A": {"boxes": {"0": [0, 0, 10, 10]}}, "B": {"boxes": {"0": [100, 0, 110, 10]}}}
    tracks = np.array([[1000, 0, 1010, 10, 1, .9], [2000, 0, 2010, 10, 2, .9]])
    rows = track_eval.score({"frames": [(0, 0., tracks, None)]}, people)
    assert rows["A"]["matched_frames"] == rows["B"]["matched_frames"] == 1
