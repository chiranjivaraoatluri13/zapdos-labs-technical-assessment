"""Alert logic on synthetic tracks: one person = one alert, flicker doesn't alert, ID switches and
re-detections don't double-alert, and zone membership uses the feet with hysteresis.

python -m pytest tests -q
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.events import EventEngine, Observation  # noqa: E402
from zapdos.track import PersonConfirmation  # noqa: E402
from zapdos.zones import ZoneSet  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/rules.yaml").read_text())
FPS = 10


def feed(engine, frames, fps=FPS, key=("no_helmet", None)):
    """frames: list of [(track_id, box, value)] per processed frame; returns all alerts."""
    alerts = []
    for i, people in enumerate(frames):
        obs = [Observation(tid, box, {key: (value, 0.9)}) for tid, box, value in people]
        alerts += engine.step(i, i / fps, obs)
    return alerts


def box(x, y=400, w=60, h=180):
    return [x, y, x + w, y + h]


def test_sustained_violation_alerts_once_after_min_duration():
    alerts = feed(EventEngine(CFG["events"], "cam", FPS), [[(1, box(100), True)]] * 100)
    assert len(alerts) == 1
    assert abs(alerts[0].alert_ts - CFG["events"]["default"]["min_duration_s"]) < 0.25


def test_rules_keep_their_meaning_at_lower_frame_rates():
    # 3.2 fps is the per-camera frame budget from README, Part 0 (Q1): a 1 s zone window holds 3
    # frames, so a frame-count minimum of 4 could never be met
    for fps in (10, 5, 3.2):
        for key in (("no_helmet", None), ("zone_entry", "z")):
            p = {**CFG["events"]["default"], **CFG["events"].get(key[0], {})}
            frames = [[(1, box(100), True)]] * int(10 * fps)
            alerts = feed(EventEngine(CFG["events"], "cam", fps), frames, fps, key)
            assert len(alerts) == 1, (fps, key)
            assert alerts[0].alert_ts - p["min_duration_s"] < 1 / fps + 1e-6, (fps, key)


def test_flicker_does_not_alert():
    frames = [[(1, box(100), i % 3 == 0)] for i in range(100)]  # violating 1 frame in 3
    assert feed(EventEngine(CFG["events"], "cam", FPS), frames) == []


def test_unknown_frames_never_count_as_violations():
    assert feed(EventEngine(CFG["events"], "cam", FPS), [[(1, box(100), None)]] * 100) == []


def test_no_new_alert_on_unknown_or_compliant_current_reading():
    # Hold alerting while evidence accumulates; releasing the hold on an unreadable
    # or compliant frame must not emit the pending vote.
    key = ("no_helmet", None)
    for value in (None, False):
        engine = EventEngine(CFG["events"], "cam", FPS)
        for i in range(17):
            assert engine.step(i, i / FPS, [Observation(1, box(100), {key: (True, .9)}, frozenset([key]))]) == []
        assert engine.step(17, 1.7, [Observation(1, box(100), {key: (value, .9)})]) == []


def test_lost_gap_does_not_count_towards_pre_alert_duration():
    engine = EventEngine(CFG["events"], "cam", FPS)
    frames = [[(1, box(100), True)]] * 6 + [[]] * 94 + [[(1, box(100), True)]] * 22
    alerts = feed(engine, frames)
    assert len(alerts) == 1
    assert alerts[0].start_ts == 10.0
    assert alerts[0].alert_ts >= 11.5


def test_unknown_gap_does_not_count_towards_pre_alert_duration():
    frames = [[(1, box(100), True)]] * 6 + [[(1, box(100), None)]] * 94
    frames += [[(1, box(100), True)]] * 22
    alerts = feed(EventEngine(CFG["events"], "cam", FPS), frames)
    assert len(alerts) == 1
    assert alerts[0].start_ts == 10.0


def test_one_missed_frame_preserves_episode_without_duplicate():
    frames = [[(1, box(100), True)]] * 30 + [[]] + [[(1, box(100), True)]] * 30
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def test_id_switch_while_walking_inherits_the_event():
    frames = [[(1, box(100 + 3 * i), True)] for i in range(40)]
    frames += [[] for _ in range(10)]                                   # lost for 1 s
    frames += [[(7, box(250 + 3 * i), True)] for i in range(40)]        # new id, next to where #1 was
    engine = EventEngine(CFG["events"], "cam", FPS)
    assert len(feed(engine, frames)) == 1
    assert engine.events[0].merged_track_ids == [7]


def test_reacquired_after_long_occlusion_in_the_same_place():
    frames = [[(1, box(100), True)]] * 40 + [[]] * 80 + [[(9, box(102), True)]] * 40   # 8 s behind a pillar
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def test_second_box_on_part_of_the_same_body_does_not_alert():
    upper_body = [105, 400, 155, 480]
    frames = [[(1, box(100), True)]] * 30 + [[(1, box(100), True), (2, upper_body, True)]] * 40
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def test_neighbour_does_not_inherit_when_the_owner_drops_a_frame():
    # #1 is violating; #2 walks up beside them already on screen; #1's detection flickers out for two
    # frames just as #2's own violation confirms: #2 is a second person, not #1 re-identified
    frames = [[(1, box(100), True), (2, box(400 - 2 * i), i >= 20)] for i in range(40)]
    frames[35] = frames[36] = [(2, box(330), True)]
    frames += [[(1, box(100), True), (2, box(320), True)]] * 20
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 2


def test_two_people_side_by_side_alert_separately():
    frames = [[(1, box(100), True), (2, box(170), True)]] * 40
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 2


def test_cooldown_stops_realert_for_the_same_track():
    frames = [[(1, box(100), True)]] * 30 + [[(1, box(100), False)]] * 50 + [[(1, box(100), True)]] * 30
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def zone_set(tmp_path):
    f = tmp_path / "z.json"
    f.write_text('{"camera_id": "cam", "frame_size": [1920, 1080], '
                 '"zones": [{"zone_id": "z", "polygon": [[500, 500], [900, 500], [900, 800], [500, 800]]}]}')
    return ZoneSet(f, (1920, 1080), CFG["zones"], FPS)


def test_zone_uses_feet_not_box_overlap(tmp_path):
    zs = zone_set(tmp_path)
    # box overlaps the zone but the feet are 40 px below its bottom edge: outside
    assert zs.read(1, [600, 640, 660, 840], 0.9)[0]["z"][0] is False
    # feet well inside
    assert zs.read(2, [600, 500, 660, 700], 0.9)[0]["z"][0] is True


def test_zone_edge_hysteresis(tmp_path):
    zs = zone_set(tmp_path)
    feet = lambda tid, y: zs.read(tid, [600, y - 180, 660, y], 0.9)[0]["z"][0]  # noqa: E731
    # entering and leaving both need 0.1 x 180 = 18 px past the 800 px edge, so feet swaying between
    # 790 and 805 keep whatever state the person already had
    assert [feet(1, y) for y in (760, 790, 805, 790, 805)] == [True] * 5
    assert [feet(2, y) for y in (840, 805, 790, 805, 790)] == [False] * 5


def test_person_confirmation_needs_repeated_confident_person_like_detections():
    c = PersonConfirmation(CFG["tracker"]["confirm"])
    n = CFG["tracker"]["confirm"]["min_hits"]
    assert not any(c.update(1, box(100), 0.35, i / FPS) for i in range(30))  # only ever weak: a grate pattern
    assert not any(c.update(2, [100, 400, 400, 430], 0.9, i / FPS) for i in range(30))  # 300x30: not a person shape
    got = [c.update(3, box(100), 0.9, i / FPS) for i in range(n + 3)]
    assert got.index(True) >= n - 1 and all(got[got.index(True):])


def test_unconfirmed_person_alerts_only_once_confirmed():
    # PPE as well as zones: a persistent false person box must not alert. Its evidence still counts,
    # so the first confirmed frame alerts at once instead of starting the persistence from scratch.
    for key in (("no_helmet", None), ("no_vest", None), ("zone_entry", "z")):
        engine = EventEngine(CFG["events"], "cam", FPS)
        held = [engine.step(i, i / FPS, [Observation(1, box(100), {key: (True, 0.4)}, confirmed=False)])
                for i in range(20)]
        assert held == [[]] * 20, key
        assert len(engine.step(20, 2.0, [Observation(1, box(100), {key: (True, 0.8)})])) == 1, key


def test_hold_still_blocks_a_key():
    key = ("zone_entry", "z")
    engine = EventEngine(CFG["events"], "cam", FPS)
    assert not any(engine.step(i, i / FPS, [Observation(1, box(100), {key: (True, .9)}, frozenset([key]))])
                   for i in range(30))


def test_hidden_feet_use_remembered_height(tmp_path):
    zs = zone_set(tmp_path)
    zs.at(0.0)
    zs.read(1, [600, 400, 660, 700], 0.9)                      # full body seen once: 300 px tall
    zs.at(0.1)
    reading, (pt, source) = zs.read(1, [600, 420, 660, 560], 0.9)  # cart hides the legs
    assert source == "estimated" and abs(pt[1] - (420 + 300 * CFG["zones"]["height_decay_per_s"] ** 0.1)) < 2
    assert reading["z"][0] is True


def test_truncated_first_sighting_reads_unknown_and_creates_no_entry(tmp_path):
    zs = zone_set(tmp_path)
    # first ever sighting is cut by the bottom of the frame: no full-body history to estimate the feet
    reading, (pt, source) = zs.read(1, [600, 900, 660, 1080], 0.9)
    assert source == "unknown" and pt is None and reading["z"][0] is None
    engine = EventEngine(CFG["events"], "cam", FPS)
    key = ("zone_entry", "z")
    assert not any(engine.step(i, i / FPS, [Observation(1, box(100), {key: reading["z"]})]) for i in range(30))


def test_stale_full_body_history_is_not_used(tmp_path):
    zs = zone_set(tmp_path)
    zs.at(0.0)
    zs.read(1, [600, 400, 660, 700], 0.9)
    zs.at(CFG["zones"]["feet_history_s"] + 1)
    assert zs.read(1, [600, 420, 660, 1080], 0.9)[1][1] == "unknown"


def test_estimate_below_the_image_is_unknown(tmp_path):
    zs = zone_set(tmp_path)
    zs.at(0.0)
    zs.read(1, [600, 300, 660, 900], 0.9)                       # 600 px tall
    zs.at(0.1)
    assert zs.read(1, [600, 600, 660, 1080], 0.9)[1][1] == "unknown"  # 600 + ~600 is off the frame


def test_unknown_point_keeps_debounced_state(tmp_path):
    zs = zone_set(tmp_path)
    zs.at(0.0)
    assert zs.read(1, [600, 500, 660, 700], 0.9)[0]["z"][0] is True
    zs.at(10.0)
    assert zs.read(1, [600, 900, 660, 1080], 0.9)[0]["z"][0] is None   # truncated, history stale
    zs.at(10.1)
    assert zs.read(1, [600, 500, 660, 700], 0.9)[0]["z"][0] is True   # still inside, no flicker


def test_feet_ending_on_an_occluder_are_hidden(tmp_path):
    f = tmp_path / "z.json"
    f.write_text('{"camera_id": "cam", "frame_size": [1920, 1080], '
                 '"zones": [{"zone_id": "z", "polygon": [[500, 500], [900, 500], [900, 800], [500, 800]]}], '
                 '"occluders": [{"polygon": [[550, 650], [750, 650], [750, 760], [550, 760]]}]}')
    zs = ZoneSet(f, (1920, 1080), CFG["zones"], FPS)
    # a cart in front: the box ends on the cart's top edge on first sight, the feet are behind it
    assert zs.read(1, [600, 500, 660, 655], 0.9)[1][1] == "unknown"


# identity, episodes and cooldown ----------------------------------------------------------------

def test_different_worker_at_the_same_spot_later_alerts_again():
    # worker 1 violates and leaves; 15 s later (inside the 60 s cooldown) worker 2 violates in the same place
    frames = [[(1, box(100), True)]] * 40 + [[]] * 150 + [[(2, box(100), True)]] * 40
    engine = EventEngine(CFG["events"], "cam", FPS)
    alerts = feed(engine, frames)
    assert len(alerts) == 2
    assert alerts[0].person_id != alerts[1].person_id
    assert engine.events[0].merged_track_ids == []


def test_walker_leaving_does_not_lend_their_event_to_someone_at_their_last_spot():
    # #1 walks right at 1 person-height/s and is lost; #2 appears where #1 was last seen 2 s later.
    # #1's motion predicts them two heights further right: #2 is someone else.
    frames = [[(1, box(100 + 18 * i), True)] for i in range(40)]
    frames += [[]] * 20 + [[(2, box(100 + 18 * 39), True)]] * 40
    alerts = feed(EventEngine(CFG["events"], "cam", FPS), frames)
    assert len(alerts) == 2


def test_two_plausible_returning_identities_do_not_merge():
    # #1 and #2 stand close together and are both hidden by a passing truck; one new box appears between them
    frames = [[(1, box(100), True), (2, box(150), True)]] * 40 + [[]] * 10 + [[(3, box(125), True)]] * 40
    engine = EventEngine(CFG["events"], "cam", FPS)
    alerts = feed(engine, frames)
    assert len(alerts) == 3                                   # a possible duplicate, never a silenced worker
    assert alerts[-1].identity["how"] == "ambiguous" and sorted(alerts[-1].identity["candidates"]) == [1, 2]


def test_best_candidate_wins_not_the_most_recent():
    # #1 is lost at 4 s at x=100; #2 is lost later (5 s) at x=260, still a plausible (worse) candidate;
    # the new #3 at x=105 is #1
    frames = [[(1, box(100), True), (2, box(260), True)]] * 40 + [[(2, box(260), True)]] * 10
    frames += [[]] * 5 + [[(3, box(105), True)]] * 30
    engine = EventEngine(CFG["events"], "cam", FPS)
    alerts = feed(engine, frames)
    assert len(alerts) == 2
    first = next(e for e in engine.events if e.track_id == 1)
    assert first.merged_track_ids == [3]


def test_two_new_tracks_claiming_one_lost_person_both_stay_separate():
    frames = [[(1, box(100), True)]] * 40 + [[]] * 5 + [[(2, box(95), True), (3, box(140), True)]] * 40
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 3


def test_genuine_short_occlusion_keeps_one_event():
    frames = [[(1, box(100), True)]] * 40 + [[]] * 15 + [[(4, box(103), True)]] * 40
    engine = EventEngine(CFG["events"], "cam", FPS)
    assert len(feed(engine, frames)) == 1
    assert engine.events[0].merged_track_ids == [4]


def test_recovery_then_new_violation_inside_cooldown_is_suppressed_without_moving_end():
    frames = [[(1, box(100), True)]] * 30 + [[(1, box(100), False)]] * 100 + [[(1, box(100), True)]] * 30
    engine = EventEngine(CFG["events"], "cam", FPS)
    alerts = feed(engine, frames)
    assert len(alerts) == 1
    ev = engine.events[0]
    assert abs(ev.end_ts - 2.9) < 1e-6 and not ev.episode_open        # the first episode's own end
    assert len(ev.suppressed_episodes) == 1 and ev.suppressed_episodes[0]["start_ts"] == 13.0


def test_new_violation_after_the_cooldown_alerts_again():
    gap = int((CFG["events"]["default"]["cooldown_s"] + 5) * FPS)
    frames = [[(1, box(100), True)]] * 30 + [[(1, box(100), False)]] * gap + [[(1, box(100), True)]] * 30
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 2


def test_one_alert_per_episode_with_closed_record():
    engine = EventEngine(CFG["events"], "cam", FPS)
    frames = [[(1, box(100), True)]] * 50 + [[(1, box(100), False)]] * 40
    assert len(feed(engine, frames)) == 1
    closed = engine.pop_closed()
    assert [e.event_id for e in closed] == [engine.events[0].event_id] and engine.pop_closed() == []


def test_appearance_vetoes_relinking_a_different_looking_person():
    import numpy as np
    red = np.zeros(128, np.float32); red[0] = 1.0     # noqa: E702
    blue = np.zeros(128, np.float32); blue[100] = 1.0  # noqa: E702
    tall = lambda x: [x, 200, x + 80, 440]  # noqa: E731  240 px: large enough to compare colour
    engine = EventEngine(CFG["events"], "cam", FPS)
    key = ("no_helmet", None)
    alerts = []
    for i in range(40):
        alerts += engine.step(i, i / FPS, [Observation(1, tall(100), {key: (True, .9)}, desc=red)])
    for i in range(50, 90):
        alerts += engine.step(i, i / FPS, [Observation(2, tall(102), {key: (True, .9)}, desc=blue)])
    assert len(alerts) == 2


def test_one_missed_frame_at_low_fps_keeps_pending_evidence():
    # 3.2 fps: one dropped processed frame is a 0.625 s gap, longer than max_evidence_gap_s 0.6
    fps = 3.2
    frames = []
    for i in range(12):
        frames.append([] if i in (3, 6) else [(1, box(100), True)])
    assert len(feed(EventEngine(CFG["events"], "cam", fps), frames, fps)) == 1
