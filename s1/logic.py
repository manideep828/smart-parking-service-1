import math
from dataclasses import dataclass

import cv2
import numpy as np

import logging
log = logging.getLogger(__name__)
from .geometry import signed_distance

def still_run_start(hist, still_frac):
    """Return the start time of the current stationary run."""
    if not hist:
        return 0.0

    latest_ts, latest_x, latest_y, latest_w = hist[-1]
    tolerance = float(still_frac) * max(float(latest_w), 1.0)
    start_ts = latest_ts

    for sample in reversed(hist):
        sample_ts, sample_x, sample_y, _ = sample

        if math.hypot(
            float(sample_x) - float(latest_x),
            float(sample_y) - float(latest_y),
        ) > tolerance:
            break

        start_ts = sample_ts

    return float(start_ts)


class Tripwire:
    """Classify crossings using apparent vehicle-size change when supplied.

    Growing apparent size indicates entry; shrinking size indicates
    exit. If size evidence is inconclusive, no direction is emitted.
    Calls without size retain the positional legacy behavior for compatibility.
    """

    def __init__(
        self,
        p1,
        p2,
        inside_point,
        margin_px=12.0,
        size_samples=3,
        min_size_change=0.15,
    ):
        self.p1, self.p2, self.margin = p1, p2, float(margin_px)
        self.inside_side = (
            1 if signed_distance(p1, p2, *inside_point) > 0 else -1
        )
        self.size_samples = max(2, int(size_samples))
        self.min_size_change = max(0.0, float(min_size_change))
        self.state = {}

    def _new_state(self):
        return {
            "side": 0,
            "last_d": None,
            "last_ts": None,
            "sizes": [],
            "candidate": None,
        }

    def update(self, tid, x, y, ts, size=None):
        d = signed_distance(self.p1, self.p2, x, y)
        st = self.state.setdefault(tid, self._new_state())

        if size is None:
            old_d, old_ts = st["last_d"], st["last_ts"]
            st["last_d"], st["last_ts"] = d, ts

            side = (
                1
                if d > self.margin
                else (-1 if d < -self.margin else 0)
            )

            if not side:
                return None

            previous = st["side"]
            st["side"] = side

            if previous and previous != side:
                cross_ts = ts

                if (
                    old_d is not None
                    and old_ts is not None
                    and old_d != d
                ):
                    cross_ts = (
                        old_ts
                        + old_d / (old_d - d) * (ts - old_ts)
                    )

                return (
                    "entry" if side == self.inside_side else "exit",
                    cross_ts,
                )

            return None

        size = float(size)

        if size > 0:
            st["sizes"].append((float(ts), size))
            st["sizes"] = st["sizes"][-120:]

        candidate = st["candidate"]

        if candidate is not None and ts > candidate["cross_ts"] and size > 0:
            candidate["after"].append(size)

            if len(candidate["after"]) >= self.size_samples:
                st["candidate"] = None

                before = candidate["before"]
                after = candidate["after"][: self.size_samples]

                if len(before) < self.size_samples:
                    st["last_d"], st["last_ts"] = d, ts
                    return None

                before_size = float(np.median(before))
                after_size = float(np.median(after))

                change = (
                    (after_size - before_size) / before_size
                    if before_size > 0
                    else 0.0
                )

                st["last_d"], st["last_ts"] = d, ts

                if abs(change) >= self.min_size_change:
                    return candidate["direction"], candidate["cross_ts"]


                return None

        side = (
            1
            if d > self.margin
            else (-1 if d < -self.margin else 0)
        )

        if side:
            previous = st["side"]
            st["side"] = side

            if previous and previous != side:
                old_d, old_ts = st["last_d"], st["last_ts"]
                cross_ts = ts

                if (
                    old_d is not None
                    and old_ts is not None
                    and old_d != d
                ):
                    cross_ts = (
                        old_ts
                        + old_d / (old_d - d) * (ts - old_ts)
                    )

                before = [
                    sz
                    for sample_ts, sz in st["sizes"]
                    if sample_ts < cross_ts
                ]

                st["candidate"] = {
                    "cross_ts": cross_ts,
                    "before": before[-self.size_samples :],
                    "after": [],
                    "direction": "entry" if side == self.inside_side else "exit",
                }

        st["last_d"], st["last_ts"] = d, ts
        if st["candidate"] is not None:
            log.debug(
                "TRIPWIRE_CANDIDATE tid=%s cross_ts=%.3f before=%s after=%s",
                tid,
                st["candidate"]["cross_ts"],
                st["candidate"]["before"],
                st["candidate"]["after"],
            )
        return None

    def forget(self, tid):
        self.state.pop(tid, None)


@dataclass
class Occupant:
    vehicle: object
    track_id: object
    parked_at: float
    last_seen: float
    lost_since: object = None
    baseline: bool = False


class SlotManager:
    """Predefined bays -> PARK_START / PARK_END per vehicle.

    PARKED  = footprint inside a bay polygon AND stationary for park_dwell_s.
              Event time is BACKDATED to when the vehicle actually stopped.
    LEFT    = out of the bay / moved away for leave_dwell_s.
              Event time is BACKDATED to the last moment it was at rest in the bay.
    If a parked vehicle's track is lost (occlusion), the bay is held for
    lost_grace_s and a new stationary track in the same bay inherits the identity.
    """

    def __init__(self, slots, frame_size, cfg):
        w, h = int(frame_size[0]), int(frame_size[1])
        self.w, self.h, self.cfg = w, h, cfg
        self.slots = {}

        for s in slots:
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(
                mask,
                [np.array(s["polygon"], np.int32)],
                1,
            )

            self.slots[s["slot_id"]] = {
                "slot_id": s["slot_id"],
                "level": s.get("level"),
                "mask": mask,
            }

        self.tstate = {}
        self.occupants = {}
        self.stream_start_ts = None

    def slot_info(self, sid):
        s = self.slots[sid]
        return {
            "slot_id": sid,
            "level": s["level"],
        }

    def overlaps(self, bbox):
        x1, y1, x2, y2 = bbox

        fy1 = y2 - self.cfg["footprint_frac"] * (y2 - y1)

        rx1 = int(max(0, x1))
        rx2 = int(min(self.w, math.ceil(x2)))
        ry1 = int(max(0, fy1))
        ry2 = int(min(self.h, math.ceil(y2)))

        area = (rx2 - rx1) * (ry2 - ry1)

        if area <= 0:
            return {}

        return {
            sid: cv2.countNonZero(
                s["mask"][ry1:ry2, rx1:rx2]
            ) / area
            for sid, s in self.slots.items()
        }

    def recover_lost_track(self, track, ts, registry):
        """Recover a parked vehicle's identity before tripwire processing.

        When a parked vehicle temporarily loses its tracker ID, the old
        Vehicle object remains attached to its occupied slot. If a new
        tracker ID appears again inside that same occupied slot while the
        old occupant is still within the lost grace period, restore the
        old Vehicle identity before any ENTRY/EXIT event is generated.

        Returns True when a recovery was performed.
        """

        ov = self.overlaps(track.bbox)

        if not ov:
            return False

        sid, frac = max(
            ov.items(),
            key=lambda kv: kv[1],
        )

        if frac < self.cfg["min_overlap"]:
            return False

        occ = self.occupants.get(sid)

        if occ is None:
            return False

        if occ.track_id is not None:
            return False

        if occ.lost_since is None:
            return False

        if (
            ts - occ.lost_since
            >= self.cfg["lost_grace_s"]
        ):
            return False

        vehicle = occ.vehicle

        # Rebind the new tracker ID to the original Vehicle object.
        registry.rebind(track.id, vehicle)

        # Treat the recovered track as already occupying this slot.
        hist = track.history

        if hist:
            cx = hist[-1][1]
            cy = hist[-1][2]
        else:
            x1, y1, x2, y2 = track.bbox
            cx = (x1 + x2) / 2.0
            cy = y2

        self.tstate[track.id] = {
            "slot": sid,
            "anchor": (cx, cy),
            "depart_since": None,
        }

        occ.track_id = track.id
        occ.lost_since = None
        occ.last_seen = ts

        return True

    def update(self, tracks, ts, registry):
        """Returns list of (event_type, vehicle, event_time, slot_id, inferred)."""
        c, events = self.cfg, []

        if self.stream_start_ts is None:
            self.stream_start_ts = float(ts)

        for t in tracks:
            st = self.tstate.setdefault(
                t.id,
                {
                    "slot": None,
                    "anchor": None,
                    "depart_since": None,
                    "baseline_slot": None,
                    "baseline_frac": 0.0,
                },
            )

            hist = t.history
            cx, cy, w = (
                hist[-1][1],
                hist[-1][2],
                hist[-1][3],
            )

            ov = self.overlaps(t.bbox)

            if st["slot"] is None:
                if not ov:
                    continue

                sid, frac = max(
                    ov.items(),
                    key=lambda kv: kv[1],
                )

                start = still_run_start(
                    hist,
                    c["still_frac"],
                )

                # Baseline EXIT clip: the stream begins with a vehicle
                # already parked. Do not require the normal parking dwell.
                baseline_window = 2.0
                baseline_candidate = (
                    self.stream_start_ts is not None
                    and ts - self.stream_start_ts <= baseline_window
                )

                if baseline_candidate:
                    if frac < 0.20:
                        continue

                    # Remember the strongest slot observed during the
                    # opening baseline window.
                    if frac > st.get("baseline_frac", 0.0):
                        st["baseline_frac"] = frac
                        st["baseline_slot"] = sid

                    continue

                # After the opening baseline window, register the remembered
                # occupant immediately. Normal arriving vehicles still use
                # the existing 4-second dwell path.
                if st.get("baseline_slot") is not None:
                    sid = st["baseline_slot"]
                    frac = st["baseline_frac"]
                    st["baseline_slot"] = None
                    st["baseline_frac"] = 0.0
                    baseline_candidate = True
                else:
                    baseline_candidate = False

                if not baseline_candidate:
                    if frac < float(c["min_overlap"]):
                        continue

                    if ts - start < c["park_dwell_s"]:
                        continue

                occ = self.occupants.get(sid)

                if occ is not None and occ.track_id is None:
                    # Lost vehicle reacquired: preserve its identity.
                    registry.rebind(t.id, occ.vehicle)

                    occ.track_id = t.id
                    occ.lost_since = None
                    occ.last_seen = ts

                    st.update(
                        slot=sid,
                        anchor=(cx, cy),
                        depart_since=None,
                    )

                elif occ is not None:
                    continue

                else:
                    v = registry.by_track[t.id]

                    baseline_window = max(
                        0.0,
                        float(c.get("baseline_window_s", 2.0)),
                    )
                    is_baseline = (
                        self.stream_start_ts is not None
                        and start >= self.stream_start_ts
                        and start - self.stream_start_ts <= baseline_window
                    )
                    self.occupants[sid] = Occupant(
                        v,
                        t.id,
                        start,
                        ts,
                        baseline=is_baseline,
                    )

                    v.slot_id = sid

                    st.update(
                        slot=sid,
                        anchor=(cx, cy),
                        depart_since=None,
                    )

                    if is_baseline:
                        log.info(
                            "BASELINE_OCCUPANT slot=%s vehicle=%s "
                            "parked_at=%.3f; ENTRY suppressed",
                            sid,
                            v.uid,
                            start,
                        )
                    else:
                        events.append(
                            (
                                "PARK_START",
                                v,
                                start,
                                sid,
                                False,
                            )
                        )

            else:
                sid = st["slot"]
                occ = self.occupants[sid]

                occ.last_seen = ts

                ax, ay = st["anchor"]

                moved = (
                    math.hypot(cx - ax, cy - ay)
                    > c["moved_frac"] * w
                )

                leaving = (
                    ov.get(sid, 0.0) < c["leave_overlap"]
                )

                if not leaving:
                    st["depart_since"] = None
                    continue

                if st["depart_since"] is None:
                    st["depart_since"] = ts


                if (
                    ts - st["depart_since"]
                    >= c["leave_dwell_s"]
                ):
                    last_rest = st["depart_since"]

                    for hts, hx, hy, _ in reversed(hist):
                        if (
                            math.hypot(hx - ax, hy - ay)
                            <= c["still_frac"] * w
                        ):
                            last_rest = hts
                            break

                    events.append(
                        (
                            "PARK_END",
                            occ.vehicle,
                            last_rest,
                            sid,
                            False,
                        )
                    )

                    occ.vehicle.slot_id = None
                    del self.occupants[sid]

                    st.update(
                        slot=None,
                        anchor=None,
                        depart_since=None,
                    )

        events += self.expire(ts)
        return events

    def on_track_removed(self, track):
        st = self.tstate.pop(track.id, None)

        if st and st["slot"]:
            occ = self.occupants.get(st["slot"])

            if occ and occ.track_id == track.id:
                occ.track_id = None
                occ.lost_since = occ.last_seen

    def expire(self, ts):
        events = []

        for sid, occ in list(self.occupants.items()):
            if (
                occ.track_id is None
                and ts - occ.lost_since
                >= self.cfg["lost_grace_s"]
            ):
                events.append(
                    (
                        "PARK_END",
                        occ.vehicle,
                        occ.lost_since,
                        sid,
                        True,
                    )
                )

                occ.vehicle.slot_id = None
                del self.occupants[sid]

        return events




