"""PPE ownership and visibility, evaluator matching and presence, and live evidence output.

python -m pytest tests -q
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from ultralytics.engine.results import Boxes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
from zapdos.ppe import decide, measure  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/rules.yaml").read_text())
EMPTY = np.empty((0, 5), np.float32)


def det(heads=(), helmets=(), vests=()):
    arr = lambda rows: np.asarray(rows, np.float32).reshape(-1, 5)  # noqa: E731
    return SimpleNamespace(heads=arr(heads), helmets=arr(helmets), vests=arr(vests))


def readings(persons, d, frame_hw=(1080, 1920), cfg=None, threshold=0.5, clf=None):
    cfg = cfg or copy.deepcopy(CFG["ppe"])
    boxes = np.asarray(persons, np.float32)
    frame = np.zeros((*frame_hw, 3), np.uint8)
    meas = measure(boxes, d, cfg, frame, clf)
    return decide(meas, boxes, cfg, threshold), meas


def vest_box(cx, y=200, w=60, h=80, conf=0.8):
    return [cx - w / 2, y, cx + w / 2, y + h, conf]


# PPE ownership ---------------------------------------------------------------------------------

def test_two_vests_two_overlapping_workers_each_get_one():
    # greedy nearest-centre gave both vests to P0 (175 is nearer P0's centre) and called P1 no_vest
    persons = [[100, 100, 200, 400], [160, 100, 260, 400]]
    r, _ = readings(persons, det(vests=[vest_box(140), vest_box(175)]))
    assert [x["no_vest"][0] for x in r] == [False, False]


def test_vest_between_two_overlapping_workers_is_ambiguous_for_both():
    persons = [[100, 100, 200, 400], [160, 100, 260, 400]]
    r, _ = readings(persons, det(vests=[vest_box(180)]))
    assert [x["no_vest"] for x in r] == [(None, 0.0, "ambiguous_owner")] * 2


def test_head_assignment_does_not_depend_on_person_order():
    a, b = [100, 100, 200, 400], [150, 120, 250, 420]
    head = [[175, 135, 205, 165, 0.9]]   # inside both boxes' head bands, clearly nearer b's centre
    _, m1 = readings([a, b], det(heads=head))
    _, m2 = readings([b, a], det(heads=head))
    assert m1[1]["head_src"] == m2[0]["head_src"] == "head"
    assert m1[0]["head_src"] == m2[1]["head_src"] == "est"


def test_duplicate_part_boxes_are_one_object():
    persons = [[100, 100, 200, 400], [205, 100, 305, 400]]
    vests = [vest_box(150), vest_box(152, conf=0.6)]  # NMS left a second box on the same vest
    r, _ = readings(persons, det(vests=vests))
    assert r[0]["no_vest"][0] is False and r[1]["no_vest"][0] is True


# torso visibility ------------------------------------------------------------------------------

def test_torso_hidden_by_person_in_front_is_unknown_not_no_vest():
    behind, front = [100, 100, 200, 400], [120, 150, 220, 450]  # front: lower box bottom = nearer the camera
    r, m = readings([behind, front], det())
    assert r[0]["no_vest"] == (None, 0.0, "torso_hidden") and m[0]["torso_cover"] > 0.3
    assert r[1]["no_vest"][0] is True


def test_torso_cut_by_bottom_of_frame_is_unknown():
    r, _ = readings([[100, 200, 200, 600]], det(), frame_hw=(480, 640))
    assert r[0]["no_vest"] == (None, 0.0, "torso_truncated")


def test_head_and_shoulders_on_the_bottom_edge_is_not_no_vest():
    # sb_rers1_0012 R5: a 110x128 box resting on the bottom edge. Its visible box "contains" a torso
    # band, but the real torso is below the frame.
    r, _ = readings([[472, 952, 582, 1080]], det())
    assert r[0]["no_vest"] == (None, 0.0, "torso_truncated")


def test_configured_occluder_hides_torso():
    cfg = copy.deepcopy(CFG["ppe"])
    boxes = np.array([[100, 100, 200, 400]], np.float32)
    meas = measure(boxes, det(), cfg, np.zeros((1080, 1920, 3), np.uint8), None,
                   occluders=[np.array([[90, 140], [210, 140], [210, 330], [90, 330]], np.float32)])
    assert decide(meas, boxes, cfg)[0]["no_vest"] == (None, 0.0, "torso_hidden")


# helmet readings -------------------------------------------------------------------------------

class FixedClassifier:
    def __init__(self, p):
        self.p = p

    def p_bare(self, frame, heads):
        return np.full(len(heads), self.p)


def test_unknown_band_between_confident_helmet_and_bare():
    cfg = copy.deepcopy(CFG["ppe"])
    cfg["classifier"]["unknown_band"] = [0.3, 0.8]
    head = [[130, 100, 160, 135, 0.9]]
    for p, want in ((0.9, True), (0.1, False), (0.5, None)):
        r, _ = readings([[100, 95, 200, 400]], det(heads=head), cfg=cfg, clf=FixedClassifier(p))
        assert r[0]["no_helmet"][0] is want, p
    assert r[0]["no_helmet"][2] == "classifier_uncertain"


def test_unknown_helmet_readings_carry_a_reason():
    small = [[140, 100, 150, 110, 0.9]]  # 10 px head < min_head_px
    r, _ = readings([[100, 95, 200, 400]], det(heads=small), clf=FixedClassifier(0.9))
    assert r[0]["no_helmet"] == (None, 0.0, "head_small")
    r, _ = readings([[100, 95, 200, 400]], det(), clf=FixedClassifier(0.9))
    assert r[0]["no_helmet"] == (None, 0.0, "no_head_box")


# evaluator -------------------------------------------------------------------------------------

def _labels(events, boxes=None):
    return {"people": {"A": {"boxes": boxes or {"0": [100, 100, 200, 400], "20": [100, 100, 200, 400]}}},
            "events": events, "ignore": []}


def _alert(i, t, etype="no_helmet"):
    return {"event_id": f"e{i}", "type": etype, "zone_id": None, "alert_ts": t, "track_id": 1,
            "bbox": [105, 105, 195, 395]}


def test_event_scoring_uses_maximum_one_to_one_matching():
    import events_eval
    # greedy took the first eligible label for the 3.5 s alert (the late episode), then called the
    # 10 s alert a duplicate and the early episode missed
    labels = _labels([{"type": "no_helmet", "person": "A", "start": 4.0, "end": 20.0},
                      {"type": "no_helmet", "person": "A", "start": 0.0, "end": 3.0}])
    s = events_eval.score_clip(labels, [_alert(1, 3.5), _alert(2, 10.0)])
    assert (s["true"], s["duplicates"], s["missed"]) == (2, 0, 0)


def test_extra_alert_on_a_matched_event_is_still_a_duplicate():
    import events_eval
    labels = _labels([{"type": "no_helmet", "person": "A", "start": 0.0, "end": 20.0}])
    s = events_eval.score_clip(labels, [_alert(1, 2.0), _alert(2, 9.0)])
    assert (s["true"], s["duplicates"], s["false"]) == (1, 1, 0)


def test_label_boxes_are_not_extrapolated_past_the_annotated_period():
    import events_eval
    person = {"boxes": {"0": [100, 100, 200, 400], "3": [100, 100, 200, 400]}}
    assert events_eval.box_at(person, 3.4) is not None
    assert events_eval.box_at(person, 10.0) is None
    labels = _labels([{"type": "no_helmet", "person": "A", "start": 0.0, "end": 14.0}], boxes=person["boxes"])
    s = events_eval.score_clip(labels, [_alert(1, 10.0)])
    assert s["false"] == 1 and s["missed"] == 1
    explicit = {**person, "present": [[0, 12]]}
    assert events_eval.box_at(explicit, 10.0) is not None


def test_missing_event_output_is_an_error_not_zero_alerts(tmp_path):
    import events_eval
    with pytest.raises(SystemExit, match="no output"):
        events_eval.load_events(tmp_path / "cam")
    (tmp_path / "cam").mkdir()
    (tmp_path / "cam" / "events.live.jsonl").write_text("")
    with pytest.raises(SystemExit, match="did not complete"):
        events_eval.load_events(tmp_path / "cam")
    missing = []
    assert events_eval.load_events(tmp_path / "cam", allow_missing=True, missing=missing) == []
    assert missing == ["cam"]


def test_tracking_diagnostics_report_one_id_on_two_people():
    import track_eval
    people = {"A": {"boxes": {"0": [0, 0, 100, 300], "1": [0, 0, 100, 300]}},
              "B": {"boxes": {"4": [500, 0, 600, 300], "5": [500, 0, 600, 300]}}}
    t = lambda x, tid: [x, 0, x + 100, 300, tid, .9]  # noqa: E731
    frames = [(0, 0.0, np.array([t(0, 7)]), None), (1, 1.0, np.array([t(0, 7)]), None),
              (4, 4.0, np.array([t(500, 7), t(1500, 9)]), None), (5, 5.0, np.array([t(500, 7)]), None)]
    rows, diag = track_eval.score_full({"frames": frames}, people)
    assert rows["A"]["ids"] == rows["B"]["ids"] == [7]   # "one ID per person" looks perfect...
    assert diag["id_reuse"] == {"7": ["A", "B"]}         # ...but it is one ID for two people
    assert diag["unmatched_tracks"] == {"9": 1}


def test_detector_fingerprint_follows_the_exported_model(tmp_path):
    from zapdos.detect import detector_model_path, model_fingerprint
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"graph-1")
    cfg = {"backend": "onnx", "onnx": str(onnx), "weights": "weights/sh17/yolo8s.pt"}
    assert detector_model_path(cfg) == onnx
    before = model_fingerprint(onnx)
    onnx.write_bytes(b"graph-2")  # re-exported at the same path
    assert model_fingerprint(onnx) != before


# live evidence output --------------------------------------------------------------------------

class VestlessPerson:
    """A detector that always sees one tall person without a vest (and no head box)."""
    backend = "fake"

    def __call__(self, frame):
        from zapdos.detect import Detections
        persons = Boxes(np.array([[100, 50, 200, 350, .9, 0]], np.float32), frame.shape[:2])
        return Detections(persons, EMPTY, EMPTY, EMPTY)


def _video(path, seconds, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 480))
    assert w.isOpened()
    for _ in range(int(seconds * fps)):
        w.write(np.full((480, 320, 3), 90, np.uint8))
    w.release()
    return path


def _run(video, out, monkeypatch=None, seen=None):
    import zapdos.pipeline as pipeline
    cfg = copy.deepcopy(CFG)
    cfg["sampling"]["process_fps"] = 10
    if seen is not None:
        original = pipeline.Camera._record

        def spy(self, kind, **fields):
            if kind == "alert":
                seen.append((fields["clip_status"], (self.out / fields["clip"]).exists()))
            original(self, kind, **fields)
        monkeypatch.setattr(pipeline.Camera, "_record", spy)
    return pipeline.run(video, None, out, cfg, "cam", models=(VestlessPerson(), None))


def _live(out):
    return [json.loads(s) for s in (out / "cam" / "events.live.jsonl").read_text().splitlines()]


def test_live_alert_says_clip_pending_until_it_exists(tmp_path, monkeypatch):
    from verify_run import verify
    seen = []
    events, _ = _run(_video(tmp_path / "v.mp4", 6), tmp_path / "out", monkeypatch, seen)
    assert len(events) == 1 and seen == [("pending", False)]
    kinds = [r["record"] for r in _live(tmp_path / "out")]
    assert kinds[0] == "alert" and "clip" in kinds and kinds[-1] == "run_end"
    final = json.loads((tmp_path / "out/cam/events.jsonl").read_text())
    assert final["clip_status"] == "ready" and (tmp_path / "out/cam" / final["clip"]).exists()
    assert final["clip_truncated"] == ["start"]  # 3 s of pre-roll requested before a 0.0 s onset
    assert final["run_id"] and verify(tmp_path / "out")["events"] == 1


def test_clip_cut_by_end_of_video_is_marked(tmp_path):
    _run(_video(tmp_path / "v.mp4", 2.5), tmp_path / "out")
    final = json.loads((tmp_path / "out/cam/events.jsonl").read_text())
    assert "end" in final["clip_truncated"] and final["episode_open"]


def test_encoder_failure_is_recorded_and_the_run_finishes(tmp_path, monkeypatch):
    import zapdos.media as media
    from verify_run import verify

    def broken(*a, **kw):
        raise RuntimeError("no usable mp4 codec")
    monkeypatch.setattr(media, "_writer", broken)
    events, _ = _run(_video(tmp_path / "v.mp4", 6), tmp_path / "out")
    final = json.loads((tmp_path / "out/cam/events.jsonl").read_text())
    assert final["clip_status"] == "failed" and "codec" in final["clip_error"]
    clip = [r for r in _live(tmp_path / "out") if r["record"] == "clip"]
    assert clip[0]["clip_status"] == "failed"
    assert verify(tmp_path / "out")["failed_clips"] == 1


def test_second_run_archives_the_first_instead_of_replacing_it(tmp_path):
    video = _video(tmp_path / "v.mp4", 6)
    _run(video, tmp_path / "out")
    first = json.loads((tmp_path / "out/cam/run.json").read_text())["run_id"]
    _run(video, tmp_path / "out")
    second = json.loads((tmp_path / "out/cam/run.json").read_text())["run_id"]
    assert first != second
    old = tmp_path / "out/cam/archive" / first
    assert (old / "events.jsonl").exists() and list((old / "clips").glob("*.mp4"))
