"""Synthetic garage: fake detector + fake plate reader drive the REAL tracker,
tripwire, slot state machine, voting, deferred emission and event builder."""
import datetime as dt
import random
import unittest

import cv2
import numpy as np

from s1.config import load_config
from s1.common import Frame
from s1.geometry import Geometry
from s1.pipeline import Pipeline
from s1.plates import PlateRead

T0 = 1_800_000_000.0
W, H = 640, 480
GEO = {
    "frame_size": [W, H],
    "tripwire": [[0, 100], [640, 100]],
    "inside_point": [320, 300],
    "plate_zone": None,
    "slots": [
        {
            "slot_id": "B-01", "level": "L1",
            "polygon": [[300, 240], [460, 240], [460, 340], [300, 340]]
        },
        {
            "slot_id": "B-02", "level": "L1",
            "polygon": [[500, 240], [620, 240], [620, 340], [500, 340]]
        },
    ],
}


def epoch(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def by_at(car, t):
    s = t - car["t0"]
    if s < 0:
        return None

    g = car.get("gap")
    if g and g[0] <= s < g[1]:
        return None

    arrive = (car["stop_by"] - 40) / 60.0
    if s <= arrive:
        return 40 + 60 * s
    if s < car["leave"]:
        return car["stop_by"]

    by = car["stop_by"] - 60 * (s - car["leave"])
    return by if by > 40 else None



class FakeWorker:
    TEXTS = ["KA01AB1234"]

    def __init__(self):
        self.n, self.pending = 0, []

    def submit(self, key, crop):
        self.pending.append((key, crop))
        return True

    def poll(self):
        out = []
        for key, crop in self.pending:
            self.n += 1
            out.append((
                key,
                [PlateRead(self.TEXTS[0], 0.9, 0.9, (0, 0, 1, 1))],
                crop
            ))
        self.pending = []
        return out


class Pub:
    def __init__(self):
        self.events = []

    def publish(self, e):
        self.events.append(e)


def run(cars, total=40.0, dt_=0.2, seed=1):
    rng = random.Random(seed)
    cfg = load_config(None, {
    "parking_area_id": None,
    "plate": {"min_vehicle_height_px": 60},
    "publisher": {"snapshot_dir": None},
})
    clock = {"t": T0}

    class Det:
        dets = None

        def detect(self, img):
            return self.dets

    det, pub = Det(), Pub()
    pipe = Pipeline(
        cfg, Geometry(GEO), (W, H), det, pub, FakeWorker(),
        now_fn=lambda: clock["t"] + 0.05
    )
    n = int(total / dt_)

    for i in range(n):
        t = i * dt_
        clock["t"] = T0 + t
        img = np.zeros((H, W, 3), np.uint8)
        dets = []

        for car in cars:
            by = by_at(car, t)
            if by is None:
                continue

            j = lambda: rng.uniform(-1.5, 1.5)

            # Simulate perspective: approaching vehicles grow in the image,
            # while departing vehicles shrink. Keep the bottom point as the
            # tripwire trajectory so expected crossing timestamps stay stable.
            scale = 0.4 + 0.6 * min(max(by, 0.0), 320.0) / 320.0
            half_w, box_h = 60.0 * scale, 80.0 * scale
            x1, x2 = car["cx"] - half_w + j(), car["cx"] + half_w + j()
            y1, y2 = max(0, by - box_h) + j(), by + j()

            cv2.rectangle(
                img, (int(x1), int(max(0, y1))),
                (int(x2), int(y2)), (0, 0, 200), -1
            )
            dets.append([x1, max(0, y1), x2, y2, 0.8, 2])

        det.dets = np.array(dets, float) if dets else np.zeros((0, 6))
        pipe.process(Frame(T0 + t, img, i))

    pipe.flush(T0 + total)
    return pub.events


def types(evs):
    return [e["event_type"] for e in evs]


class TestE2E(unittest.TestCase):
    def test_single_car_full_lifecycle(self):
        evs = run([{"t0": 0, "cx": 380, "stop_by": 320, "leave": 20}])
        self.assertEqual(types(evs), ["ENTRY", "PARK_START", "PARK_END", "EXIT"])

        entry, ps, pe, ex = evs
        self.assertEqual(entry["direction"], "entry")
        self.assertEqual(ex["direction"], "exit")
        self.assertAlmostEqual(
            epoch(entry["event_time"]) - T0, 1.0, delta=0.3
        )
        self.assertAlmostEqual(
            epoch(ps["event_time"]) - T0, 4.6, delta=0.5
        )
        self.assertAlmostEqual(
            epoch(pe["event_time"]) - T0, 20.2, delta=0.5
        )
        self.assertAlmostEqual(
            epoch(ex["event_time"]) - T0, 23.67, delta=0.4
        )
        self.assertGreater(
            epoch(entry["detected_at"]), epoch(entry["event_time"])
        )
        self.assertEqual(ps["bay"], {"id": "B-01", "level": "L1"})

        for e in evs:
            self.assertEqual(e["plate"]["text"], "KA01AB1234")
            self.assertEqual({k: e["vehicle"][k] for k in ("type", "color")}, {"type": "car", "color": "red"})

        self.assertEqual(len({e["vehicle_uid"] for e in evs}), 1)
        self.assertEqual(len({e["event_id"] for e in evs}), 4)

    def test_two_vehicles_overlapping_in_time(self):
        cars = [
            {"t0": 0, "cx": 380, "stop_by": 320, "leave": 22},
            {"t0": 3, "cx": 560, "stop_by": 300, "leave": 18},
        ]
        evs = run(cars, total=42)
        by_uid = {}

        for e in evs:
            by_uid.setdefault(e["vehicle_uid"], []).append(e)

        self.assertEqual(len(by_uid), 2)
        bays = set()

        for lst in by_uid.values():
            self.assertEqual(
                types(lst), ["ENTRY", "PARK_START", "PARK_END", "EXIT"]
            )
            bays.add(lst[1]["bay"]["id"])

        self.assertEqual(bays, {"B-01", "B-02"})

    def test_track_lost_while_parked_keeps_identity(self):
        # Car hidden for 5 s while parked; tracker drops it, then it reappears.
        evs = run(
            [{
                "t0": 0, "cx": 380, "stop_by": 320,
                "leave": 32, "gap": (10, 15)
            }],
            total=50
        )

        self.assertEqual(types(evs).count("PARK_START"), 1)
        self.assertEqual(types(evs).count("PARK_END"), 1)
        self.assertEqual(
            len({
                e["vehicle_uid"] for e in evs
                if e["event_type"] != "VEHICLE_UPDATE"
            }),
            1
        )

        pe = [e for e in evs if e["event_type"] == "PARK_END"][0]
        self.assertFalse(pe["inferred"])

    def test_car_only_passing_through_does_not_park(self):
        evs = run([
            {"t0": 0, "cx": 380, "stop_by": 320, "leave": 5.0}
        ])

        self.assertNotIn("PARK_START", types(evs))
        self.assertIn("ENTRY", types(evs))
        self.assertIn("EXIT", types(evs))


if __name__ == "__main__":
    unittest.main()