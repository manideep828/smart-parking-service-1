import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .attributes import TYPE_BY_COCO
from .plates import PlateVoter


@dataclass
class PendingCross:
    direction: str
    event_time: float
    deadline: float


@dataclass
class Vehicle:
    uid: str
    track_id: int
    first_ts: float
    cls_votes: Counter = field(default_factory=Counter)
    color_votes: Counter = field(default_factory=Counter)
    voter: PlateVoter = field(default_factory=PlateVoter)
    plate_final: bool = False
    plate_attempts: int = 0
    plate_inflight: bool = False
    last_plate_submit: float = 0.0
    last_color_ts: float = 0.0
    last_crop: object = None
    pending: list = field(default_factory=list)
    any_emitted: bool = False
    emitted_plate: str = None
    slot_id: str = None

    # Last tracker observation, used only for safe identity recovery.
    last_bbox: object = None
    last_seen_ts: float = 0.0
    pending_entry_received_at: float = None
    pending_exit_crossing_ts: float = None

    def vehicle_type(self):
        if not self.cls_votes:
            return None
        return TYPE_BY_COCO.get(
            self.cls_votes.most_common(1)[0][0],
            "vehicle",
        )

    def color(self):
        return (
            self.color_votes.most_common(1)[0][0]
            if self.color_votes
            else None
        )


class VehicleRegistry:
    IDENTITY_RECOVERY_MAX_S = 5.0
    IDENTITY_RECOVERY_MIN_IOU = 0.10
    IDENTITY_RECOVERY_MAX_CENTER_DISTANCE = 0.45

    def __init__(self):
        self.by_track = {}
        self.by_uid = {}

    @staticmethod
    def _iou(a, b):
        if a is None or b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = aa + ab - inter
        if union <= 0.0:
            return 0.0

        return inter / union

    @staticmethod
    def _center_distance_ratio(a, b):
        if a is None or b is None:
            return float("inf")

        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)

        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        distance = float(np.hypot(acx - bcx, acy - bcy))

        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        scale = max(aw, ah, bw, bh)

        return distance / scale

    def _remember_observation(self, vehicle, track, ts):
        vehicle.track_id = track.id
        vehicle.last_bbox = np.asarray(track.bbox, dtype=float).copy()
        vehicle.last_seen_ts = float(ts)

    def _find_recoverable_vehicle(self, track, ts):
        candidates = []

        for vehicle in self.by_uid.values():
            if not (vehicle.plate_final or vehicle.plate_inflight):
                continue

            if vehicle.slot_id is not None:
                continue

            if vehicle.track_id in self.by_track:
                continue

            if vehicle.last_bbox is None:
                continue

            age = float(ts) - float(vehicle.last_seen_ts)

            if age < 0.0 or age > self.IDENTITY_RECOVERY_MAX_S:
                continue

            iou = self._iou(vehicle.last_bbox, track.bbox)
            distance = self._center_distance_ratio(
                vehicle.last_bbox,
                track.bbox,
            )

            if (
                iou >= self.IDENTITY_RECOVERY_MIN_IOU
                or distance <= self.IDENTITY_RECOVERY_MAX_CENTER_DISTANCE
            ):
                candidates.append((iou, distance, age, vehicle))

        if not candidates:
            return None

        # Prefer strongest spatial match, then newest observation.
        candidates.sort(
            key=lambda x: (-x[0], x[1], x[2])
        )

        best = candidates[0]

        # Never recover if two retained identities are equally plausible.
        if len(candidates) > 1:
            second = candidates[1]

            if (
                abs(best[0] - second[0]) < 0.05
                and abs(best[1] - second[1]) < 0.05
            ):
                return None

        return best[3]

    def get_or_create(self, track, ts):
        v = self.by_track.get(track.id)

        if v is not None:
            self._remember_observation(v, track, ts)
            return v

        # A new tracker ID may actually be the same physical vehicle
        # whose old tracker disappeared briefly.
        recovered = self._find_recoverable_vehicle(track, ts)

        print(
            "IDENTITY_CHECK",
            "new_track=", track.id,
            "ts=", round(float(ts), 3),
            "candidates=", [
                (
                    v.uid,
                    v.track_id,
                    v.plate_final,
                    v.plate_inflight,
                    round(float(ts) - float(v.last_seen_ts), 3),
                )
                for v in self.by_uid.values()
                if v.plate_final or v.plate_inflight
            ],
            flush=True,
        )

        if recovered is not None:
            old_track_id = recovered.track_id

            self.by_track.pop(old_track_id, None)

            recovered.track_id = track.id
            self.by_track[track.id] = recovered
            self.by_uid[recovered.uid] = recovered
            self._remember_observation(recovered, track, ts)

            return recovered

        v = Vehicle(
            uid=uuid.uuid4().hex,
            track_id=track.id,
            first_ts=ts,
        )

        self.by_track[track.id] = v
        self.by_uid[v.uid] = v
        self._remember_observation(v, track, ts)

        return v

    def rebind(self, track_id, vehicle):
        dup = self.by_track.get(track_id)

        if dup is not None and dup is not vehicle:
            self.by_uid.pop(dup.uid, None)

        vehicle.track_id = track_id
        self.by_track[track_id] = vehicle
        self.by_uid[vehicle.uid] = vehicle

    def drop_track(self, track_id):
        v = self.by_track.pop(track_id, None)

        if v is not None:
            # Keep identities while OCR is still in flight so the
            # asynchronous OCR result can update the original Vehicle.
            # Also keep finalized identities for short tracker-ID recovery.
            if v.plate_inflight or v.plate_final:
                return v

            if v.slot_id is None:
                self.by_uid.pop(v.uid, None)

        return v

    def release(self, vehicle):
        self.by_uid.pop(vehicle.uid, None)
