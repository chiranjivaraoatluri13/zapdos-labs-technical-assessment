"""Verify alert records, snapshots and evidence clips from one completed run.

python scripts/verify_run.py PATH_TO_OUTPUT_ROOT

Per camera folder: events.jsonl has the required fields, unique IDs, one run_id, and no clip still
"pending"; every "ready" snapshot and clip opens; a "failed" one is reported, not hidden. When
events.live.jsonl is there, it must hold one alert record and one clip record per event, the alert
record before the clip record, and end with run_end.
"""
import argparse
import json
from pathlib import Path

import cv2

REQUIRED = {"event_id", "camera_id", "type", "track_id", "start_ts", "end_ts", "first_frame", "confidence", "bbox",
            "snapshot", "clip", "alert_ts", "clip_status", "snapshot_status", "run_id"}


def _check_live(path, events):
    recs = [json.loads(s) for s in path.read_text().splitlines() if s]
    if not recs or recs[-1]["record"] != "run_end":
        raise ValueError(f"{path}: no run_end record (run incomplete)")
    order = {}
    for i, r in enumerate(recs):
        if r["record"] in ("alert", "clip"):
            order.setdefault(r["event_id"], []).append((r["record"], i))
    for e in events:
        if e["type"] == "camera_moved" and e["clip_status"] == "":
            continue
        kinds = [k for k, _ in order.get(e["event_id"], [])]
        if kinds != ["alert", "clip"]:
            raise ValueError(f"{path}: {e['event_id']} has live records {kinds}, expected alert then clip")
        alert = next(r for r in recs if r["record"] == "alert" and r["event_id"] == e["event_id"])
        if alert["clip_status"] != "pending":
            raise ValueError(f"{path}: alert record for {e['event_id']} claims clip {alert['clip_status']}")


def verify(root):
    counts = {"cameras": 0, "events": 0, "failed_snapshots": 0, "failed_clips": 0, "truncated_clips": 0}
    for path in Path(root).rglob("events.jsonl"):
        if "archive" in path.relative_to(root).parts:
            continue
        counts["cameras"] += 1
        events = [json.loads(s) for s in path.read_text().splitlines() if s]
        if len({e["event_id"] for e in events}) != len(events):
            raise ValueError(f"duplicate event IDs in {path}")
        if len({e.get("run_id") for e in events}) > 1:
            raise ValueError(f"events from more than one run in {path}")
        for e in events:
            if REQUIRED - e.keys() or not e["start_ts"] <= e["alert_ts"] <= e["end_ts"] + 1e-6:
                raise ValueError(f"invalid event record: {e}")
            if e["clip_status"] == "pending":
                raise ValueError(f"{e['event_id']}: clip still pending in a finalised events.jsonl")
            if e["snapshot_status"] == "failed":
                counts["failed_snapshots"] += 1
            elif cv2.imread(str(path.parent / e["snapshot"])) is None:
                raise ValueError(f"unreadable snapshot for {e['event_id']}")
            if e["clip_status"] == "failed":
                counts["failed_clips"] += 1
            else:
                cap = cv2.VideoCapture(str(path.parent / e["clip"]))
                try:
                    ok, _ = cap.read()
                    if not ok or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
                        raise ValueError(f"unreadable evidence clip for {e['event_id']}")
                finally:
                    cap.release()
            counts["truncated_clips"] += bool(e.get("clip_truncated"))
            counts["events"] += 1
        live = path.parent / "events.live.jsonl"
        if live.exists():
            _check_live(live, events)
    if not counts["cameras"]:
        raise ValueError(f"no completed event outputs under {root}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    print(verify(ap.parse_args().root))
