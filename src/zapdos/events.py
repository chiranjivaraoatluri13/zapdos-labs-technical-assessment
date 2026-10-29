"""Turn per-frame readings into deduplicated alert events.

Three things are kept apart:

person identity - which tracker IDs are one worker. A tracker ID is attached to an identity the
    first time it is seen, and only then:
      * nested: its box lies inside exactly one visible identity's box (a second detection of part
        of the same body). Inside two visible people's boxes is ambiguous: a new identity.
      * re-linked: otherwise every *lost* identity is scored against it: distance from where that
        identity's recent motion predicts it to be (in person heights), box-size change, and, when
        both crops are big enough, torso colour. Gates: within identity.window_s and identity.radius,
        or, for someone who was standing still, back in the same place within stationary_window_s.
        It joins the best candidate only when that candidate is unique: no other candidate within
        identity.margin of it, and no other new track in the same frame also plausibly that person.
      * otherwise a new identity. When there were candidates but none was clearly best, the identity
        records them as `ambiguous`, so a possible duplicate alert can be audited.
    Conservative on purpose: a missed re-link costs a duplicate alert, a wrong one silences a
    different worker.

violation episode - per (identity, type, zone). Each key keeps a sliding window of readings. An
    episode alerts once when, within the window, enough frames are readable, the violating share
    reaches the threshold, and the current violating run has lasted min_duration_s. It closes when
    the violation has been unseen for end_after_s; its end_ts is the last violating frame.

alert cooldown - a new episode of the same key that would alert within cooldown_s of the previous
    episode's end is suppressed: recorded on the earlier event as a suppressed episode, without
    moving that event's end_ts. Cooldown is per identity, never shared between identities.

Readings from an identity that is not yet a confirmed person (track.PersonConfirmation), or keys in
an observation's `hold`, gather evidence but cannot alert.
"""
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

from .track import appearance_distance

MOTION_WINDOW_S = 1.0    # velocity from the detections of the last second
MIN_MOTION_SPAN_S = 0.3  # ...when they span at least this long; less = assume standing still


@dataclass
class Event:
    event_id: str
    camera_id: str
    type: str
    track_id: int
    zone_id: str | None
    start_ts: float
    first_frame: int
    alert_ts: float
    alert_frame: int
    end_ts: float            # last violating frame of the episode (provisional while episode_open)
    confidence: float        # vote x mean confidence of the violating readings in the window
    vote: float              # share of readable readings in the window that were violating
    bbox: list
    person_id: int = -1      # identity (several tracker IDs may be one person)
    identity: dict = None    # how that identity was formed: new | linked | nested | ambiguous
    snapshot: str = ""
    clip: str = ""
    snapshot_status: str = ""  # ready | failed
    clip_status: str = ""      # pending (still recording or encoding) | ready | failed
    clip_start_ts: float = None
    clip_end_ts: float = None
    clip_truncated: list = field(default_factory=list)  # "start" and/or "end": shorter than requested
    clip_error: str = None
    merged_track_ids: list = field(default_factory=list)  # other tracker IDs that fed this episode
    suppressed_episodes: list = field(default_factory=list)  # later episodes inside the cooldown
    episode_open: bool = True
    run_id: str = ""
    note: str = None         # camera-level events: what was measured

    @property
    def delay_s(self):
        return round(self.alert_ts - self.start_ts, 3)

    def to_json(self):
        d = asdict(self)
        if d["note"] is None:
            d.pop("note")
        for k in ("identity", "clip_error"):
            if d[k] is None:
                d.pop(k)
        d["delay_s"] = self.delay_s
        for k in ("start_ts", "alert_ts", "end_ts", "clip_start_ts", "clip_end_ts"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        d["suppressed_episodes"] = [{k: round(v, 3) for k, v in s.items()} for s in self.suppressed_episodes]
        return d


@dataclass
class Observation:
    track_id: int
    box: list
    readings: dict  # (type, zone_id) -> (value, conf[, reason])
    hold: frozenset = frozenset()  # reading keys that gather evidence but may not alert yet
    confirmed: bool = True         # the track is a confirmed person (track.PersonConfirmation)
    desc: object = None            # appearance descriptor (track.appearance) or None


@dataclass
class _Frame:
    idx: int
    ts: float


class _Identity:
    def __init__(self, pid, ts, link):
        self.pid = pid
        self.first_ts = self.last_ts = ts
        self.last_box = None
        self.motion = deque()  # (ts, cx, cy) of recent detections
        self.desc = None
        self.confirmed = False
        self.link = link

    def observe(self, ts, box, desc, confirmed):
        self.last_ts, self.last_box = ts, list(box)
        self.motion.append((ts, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        while self.motion[0][0] < ts - MOTION_WINDOW_S:
            self.motion.popleft()
        if desc is not None:
            self.desc = desc
        self.confirmed = self.confirmed or confirmed

    def velocity(self):
        """Pixels per second over the recent detections; (0, 0) when they span too short a time."""
        (t0, x0, y0), (t1, x1, y1) = self.motion[0], self.motion[-1]
        if t1 - t0 < MIN_MOTION_SPAN_S:
            return 0.0, 0.0
        return (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)


class _KeyState:
    def __init__(self):
        self.window = deque()   # (ts, frame_idx, value, conf)
        self.run_start = None   # (ts, frame_idx) of the first violating frame of the current run
        self.last_true = None
        self.event = None       # open alerted episode
        self.suppressed = None  # open episode suppressed by cooldown: {"start_ts", "end_ts"}
        self.last_event = None  # latest alerted event of this key
        self.last_end = None    # end of the latest episode, alerted or suppressed: cooldown runs from here

    def add(self, now, value, conf, window_s):
        """Slide the window to `now` and extend or end the current violating run."""
        self.window.append((now.ts, now.idx, value, conf))
        while self.window[0][0] <= now.ts - window_s:
            self.window.popleft()
        if value:
            if self.run_start is None:
                self.run_start = (now.ts, now.idx)
            self.last_true = now.ts
        elif not any(w[2] for w in self.window):
            self.run_start = None


class EventEngine:
    def __init__(self, cfg, camera_id, fps):
        self.cfg = cfg
        self.idcfg = cfg["identity"]
        self.camera_id = camera_id
        self.fps = fps
        self.keys = {}
        self.events = []
        self.ids = {}         # person id -> _Identity
        self.track_pid = {}   # tracker id -> person id
        self.alias = set()    # tracker ids attached as a nested second box of someone else
        self.closed = []      # events whose episode closed since the last pop_closed()
        self._params = {}

    def params(self, etype):
        """The rule settings for one event type: its own section over `default`."""
        if etype not in self._params:
            p = {**self.cfg["default"], **self.cfg.get(etype, {})}
            p["min_observed"] = math.ceil(p["min_observed_s"] * self.fps - 1e-6)  # frames at this fps
            # a gap of max_missed_frames dropped frames never resets evidence, whatever the fps
            p["evidence_gap_s"] = max(p.get("max_evidence_gap_s", 0.6),
                                      (1 + p.get("max_missed_frames", 1)) / self.fps)
            self._params[etype] = p
        return self._params[etype]

    def step(self, frame_idx, ts, observations):
        """Feed one processed frame; returns the events that alerted on this frame."""
        now = _Frame(frame_idx, ts)
        # Expire evidence before ingesting a returning track. Its first fresh
        # reading must not inherit a pre-alert clock from before an occlusion.
        self._end_episodes(ts)
        pids = self._resolve(observations, ts)
        new, fed = [], set()
        for o in sorted(observations, key=lambda o: -_area(o.box)):
            pid = pids[o.track_id]
            if pid in fed:
                continue  # a second, nested box of someone already read this frame
            fed.add(pid)
            ident = self.ids[pid]
            ident.observe(ts, o.box, o.desc, o.confirmed)
            for key, r in o.readings.items():
                ev = self._update(o, pid, ident, key, r[0], r[1], now)
                if ev is not None:
                    new.append(ev)
        return new

    def person_of(self, track_id):
        return self.track_pid.get(track_id, -1)

    def pop_closed(self):
        out, self.closed = self.closed, []
        return out

    def camera_event(self, etype, start_ts, first_frame, ts, frame_idx, frame_size, note):
        """An event about the camera rather than a person (camera_moved): no track, whole-frame box."""
        ev = Event(event_id=self._next_id(), camera_id=self.camera_id,
                   type=etype, track_id=-1, zone_id=None, start_ts=start_ts, first_frame=first_frame,
                   alert_ts=ts, alert_frame=frame_idx, end_ts=ts, confidence=1.0, vote=1.0,
                   bbox=[0.0, 0.0, float(frame_size[0]), float(frame_size[1])], note=note)
        self.events.append(ev)
        return ev

    def _next_id(self):
        return f"{self.camera_id}-{len(self.events) + 1:03d}"

    # identity ------------------------------------------------------------------------------------

    def _resolve(self, observations, ts):
        """tracker id -> person id for this frame's observations, attaching first-seen tracker ids."""
        known = [o for o in observations if o.track_id in self.track_pid]
        self._split_aliases(known, ts)
        visible = {self.track_pid[o.track_id]: o for o in known}
        new = [o for o in observations if o.track_id not in self.track_pid]
        plausible = {}
        for o in new:
            hosts = {pid for pid, v in visible.items()
                     if _area(o.box) <= _area(v.box) and _nested(o.box, v.box) >= self.idcfg["nested_overlap"]}
            if len(hosts) == 1:
                pid = hosts.pop()
                self.track_pid[o.track_id] = pid
                self.alias.add(o.track_id)
                continue
            if hosts:
                self._new_identity(o.track_id, ts, {"how": "ambiguous", "nested_in": sorted(hosts)})
                continue
            plausible[o.track_id] = sorted((c, pid) for pid, ident in self.ids.items()
                                           if pid not in visible and (c := self._link_cost(o, ident, ts)) is not None)
        claims = Counter(pid for cands in plausible.values() for _, pid in cands)
        for tid, cands in plausible.items():
            if not cands:
                self._new_identity(tid, ts, {"how": "new"})
                continue
            (best, pid), rest = cands[0], cands[1:]
            clear = not rest or rest[0][0] - best >= self.idcfg["margin"]
            if clear and claims[pid] == 1:
                self.track_pid[tid] = pid
                self.ids[pid].link = {**self.ids[pid].link, "relinked": self.ids[pid].link.get("relinked", 0) + 1}
            else:
                self._new_identity(tid, ts, {"how": "ambiguous", "candidates": [p for _, p in cands]})
        return {o.track_id: self.track_pid[o.track_id] for o in observations}

    def _split_aliases(self, known, ts):
        """A nested second box that now stands apart from its host was a different person after all."""
        by_pid = {}
        for o in known:
            by_pid.setdefault(self.track_pid[o.track_id], []).append(o)
        for group in by_pid.values():
            if len(group) < 2:
                continue
            host = max(group, key=lambda o: _area(o.box))
            for o in group:
                if o is not host and o.track_id in self.alias and \
                        _nested(o.box, host.box) < self.idcfg["nested_overlap"]:
                    self.alias.discard(o.track_id)
                    self._new_identity(o.track_id, ts, {"how": "split", "from": self.track_pid[o.track_id]})

    def _new_identity(self, tid, ts, link):
        pid = len(self.ids) + 1
        self.ids[pid] = _Identity(pid, ts, link)
        self.track_pid[tid] = pid
        return pid

    def _link_cost(self, o, ident, ts):
        """How well lost identity `ident` explains new box `o.box` (lower is better), or None if not plausible."""
        c = self.idcfg
        gap = ts - ident.last_ts
        if ident.last_box is None or gap <= 0 or gap > max(c["window_s"], c["stationary_window_s"]):
            return None
        last = ident.last_box
        h_last, h_new = last[3] - last[1], o.box[3] - o.box[1]
        if h_last <= 0 or h_new <= 0:
            return None
        size = max(h_new, h_last) / min(h_new, h_last)
        if size > c["max_size_ratio"]:
            return None
        vx, vy = ident.velocity()
        horizon = min(gap, c["window_s"])
        px = (last[0] + last[2]) / 2 + vx * horizon
        py = (last[1] + last[3]) / 2 + vy * horizon
        d = math.hypot((o.box[0] + o.box[2]) / 2 - px, (o.box[1] + o.box[3]) / 2 - py) / h_last
        still = math.hypot(vx, vy) / h_last <= c["stationary_speed"]
        walked = gap <= c["window_s"] and d <= c["radius"]
        stayed = still and gap <= c["stationary_window_s"] and _nested(o.box, last) >= c["stationary_overlap"]
        if not (walked or stayed):
            return None
        cost = d + math.log(size)
        if o.desc is not None and ident.desc is not None and min(h_new, h_last) >= c["appearance_min_px"]:
            dist = appearance_distance(o.desc, ident.desc)
            if dist > c["appearance_max_dist"]:
                return None
            cost += dist
        return cost

    # episodes ------------------------------------------------------------------------------------

    def _update(self, o, pid, ident, key, value, conf, now):
        """Add one reading to its (person, type, zone) state; returns the event if it alerts now."""
        st = self.keys.setdefault((pid, *key), _KeyState())
        st.add(now, value, conf, self.params(key[0])["window_s"])
        if value and st.event:
            st.event.end_ts = now.ts
            if o.track_id != st.event.track_id and o.track_id not in st.event.merged_track_ids:
                st.event.merged_track_ids.append(o.track_id)
        elif value and st.suppressed:
            st.suppressed["end_ts"] = now.ts
        can_alert = ident.confirmed and key not in o.hold
        if value and st.event is None and st.suppressed is None and st.run_start is not None and can_alert:
            return self._maybe_alert(st, o, pid, ident, key, now)
        return None

    def _end_episodes(self, ts):
        for (_, etype, _), st in self.keys.items():
            p = self.params(etype)
            if st.last_true is None:
                continue
            if ts - st.last_true > p["evidence_gap_s"] + 1e-6:
                st.window.clear()
                st.run_start = None
            if (st.event or st.suppressed) and ts - st.last_true >= p["end_after_s"]:
                if st.event:
                    st.event.episode_open = False
                    self.closed.append(st.event)
                st.last_end = st.last_true
                st.event = st.suppressed = None
                st.run_start = None

    def _maybe_alert(self, st, o, pid, ident, key, now):
        p = self.params(key[0])
        readable = [(w[2], w[3]) for w in st.window if w[2] is not None]
        if len(readable) < p["min_observed"] or now.ts - st.run_start[0] < p["min_duration_s"]:
            return None
        hits = [c for v, c in readable if v]
        vote = len(hits) / len(readable)
        if vote < p["threshold"]:
            return None

        if st.last_end is not None and now.ts - st.last_end <= p["cooldown_s"]:
            st.suppressed = {"start_ts": st.run_start[0], "end_ts": now.ts}
            st.last_event.suppressed_episodes.append(st.suppressed)
            return None

        ev = Event(
            event_id=self._next_id(),
            camera_id=self.camera_id, type=key[0], track_id=o.track_id, zone_id=key[1],
            start_ts=st.run_start[0], first_frame=st.run_start[1],
            alert_ts=now.ts, alert_frame=now.idx, end_ts=now.ts,
            confidence=round(vote * sum(hits) / len(hits), 3), vote=round(vote, 3),
            bbox=[round(float(v), 1) for v in o.box], person_id=pid, identity=dict(ident.link),
        )
        self.events.append(ev)
        st.event = st.last_event = ev
        return ev


def _area(a):
    return max(a[2] - a[0], 0) * max(a[3] - a[1], 0)


def _nested(a, b):
    """Intersection over the smaller box's area: 1.0 when one box lies inside the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min(_area(a), _area(b))
    return w * h / small if small > 0 else 0.0
