import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from s1.attributes import dominant_color
from s1.detector import YoloxDetector, decode_outputs, make_grids
from s1.logic import Tripwire
from s1.outbox import Outbox
from s1.plates import PlateVoter, fix_indian
from s1.tracker import ByteTracker
from s2_stub.server import make_server


class TestPlates(unittest.TestCase):
    def test_fix_and_validate(self):
        self.assertEqual(fix_indian("KAO1ABI234"), ("KA01AB1234", True))
        self.assertEqual(fix_indian("ka 01 ab 1234"), ("KA01AB1234", True))
        self.assertEqual(fix_indian("MH12DE1433"), ("MH12DE1433", True))
        self.assertEqual(fix_indian("DL8CAF5030"), ("DL8CAF5030", True))
        self.assertEqual(fix_indian("22BH1234AA"), ("22BH1234AA", True))
        self.assertEqual(fix_indian("TS09EA1234"), ("TS09EA1234", True))
        self.assertFalse(fix_indian("XX01AB1234")[1])
        self.assertFalse(fix_indian("HELLO")[1])

    def test_voting_recovers_from_bad_reads(self):
        v = PlateVoter()
        for txt, c in [
            ("KA01AB1234", .9), ("KA01A81234", .6),
            ("KA01AB1234", .85), ("KAO1AB1Z34", .7)
        ]:
            v.add(txt, c)
        c = v.consensus()
        self.assertEqual(c.text, "KA01AB1234")
        self.assertTrue(c.valid)
        self.assertTrue(v.is_final(3, 0.6))

    def test_not_final_without_enough_reads(self):
        v = PlateVoter()
        v.add("KA01AB1234", .9)
        self.assertFalse(v.is_final(3, 0.5))


class TestColor(unittest.TestCase):
    def test_colors(self):
        for bgr, name in [
            ((0, 0, 220), "red"), ((220, 40, 20), "blue"),
            ((20, 20, 20), "black"), ((250, 250, 250), "white"),
            ((30, 180, 30), "green")
        ]:
            img = np.zeros((100, 100, 3), np.uint8)
            img[:] = bgr
            self.assertEqual(dominant_color(img), name)


class TestTracker(unittest.TestCase):
    def test_stable_ids_for_two_vehicles(self):
        tr = ByteTracker()
        ids = set()
        for i in range(30):
            ts = i * 0.2
            dets = [
                [100, 100 + 10 * i, 220, 180 + 10 * i, .9, 2],
                [400, 400 - 8 * i, 520, 480 - 8 * i, .8, 7]
            ]
            active, _ = tr.update(np.array(dets, float), ts)
            if i >= 3:
                self.assertEqual(len(active), 2)
                ids |= {t.id for t in active}
        self.assertEqual(len(ids), 2)

    def test_removed_after_max_age(self):
        tr = ByteTracker(max_age_s=1.0)
        for i in range(5):
            tr.update(np.array([[10, 10, 100, 100, .9, 2]], float), i * 0.2)
        removed_all = []
        for i in range(5, 15):
            _, rem = tr.update(np.zeros((0, 6)), i * 0.2)
            removed_all += rem
        self.assertEqual(len(removed_all), 1)


class TestTripwire(unittest.TestCase):
    def test_direction_and_hysteresis(self):
        tw = Tripwire((0, 100), (640, 100), (320, 300), margin_px=12)
        res = []
        for i, y in enumerate([40, 70, 98, 104, 99, 103, 130, 160]):
            r = tw.update(1, 300, y, i * 0.2)
            if r:
                res.append(r)
        self.assertEqual([r[0] for r in res], ["entry"])

        tw2 = Tripwire((0, 100), (640, 100), (320, 300), margin_px=12)
        res2 = [
            tw2.update(1, 300, y, i * 0.2)
            for i, y in enumerate([200, 150, 110, 95, 80, 60])
        ]
        self.assertEqual([r[0] for r in res2 if r], ["exit"])

    def test_size_change_classifies_entry_exit_and_rejects_ambiguous(self):
        def collect(samples):
            tw = Tripwire(
                (0, 100), (640, 100), (320, 300),
                margin_px=12, size_samples=3, min_size_change=0.15
            )
            found = []
            for i, (y, size) in enumerate(samples):
                result = tw.update(1, 300, y, i * 0.2, size=size)
                if result:
                    found.append(result[0])
            return found

        entering = [
            (y, size) for y, size in [
                (40, 100), (60, 120), (80, 140), (90, 150), (110, 170),
                (130, 210), (150, 240), (170, 270), (190, 300)
            ]
        ]
        leaving = [
            (y, size) for y, size in [
                (190, 300), (170, 270), (150, 240), (130, 210), (110, 170),
                (90, 150), (70, 130), (50, 110), (30, 90), (10, 70)
            ]
        ]
        ambiguous = [
            (y, 100.0) for y in [40, 60, 80, 90, 110, 130, 150, 170, 190]
        ]

        self.assertEqual(collect(entering), ["entry"])
        self.assertEqual(collect(leaving), ["exit"])
        self.assertEqual(collect(ambiguous), [])


class TestYolox(unittest.TestCase):
    def test_decode_and_detect(self):
        S = 416
        grids, strides = make_grids(S)
        raw = np.zeros((grids.shape[0], 85), np.float32)
        i = 20 * (S // 8) + 10
        raw[i, :2] = [0.5, 0.5]
        raw[i, 2:4] = [np.log(4), np.log(2)]
        raw[i, 4] = 0.9
        raw[i, 5 + 2] = 0.95
        dec = decode_outputs(raw, grids, strides)
        self.assertAlmostEqual(float(dec[i, 0]), 84.0)
        self.assertAlmostEqual(float(dec[i, 1]), 164.0)
        self.assertAlmostEqual(float(dec[i, 2]), 32.0)

        class Inp:
            name = "images"

        class FakeSess:
            def get_inputs(self):
                return [Inp()]

            def run(self, *_):
                return [raw[None]]

        det = YoloxDetector(session=FakeSess(), input_size=S)
        out = det.detect(np.zeros((416, 416, 3), np.uint8))
        self.assertEqual(out.shape, (1, 6))
        self.assertEqual(int(out[0, 5]), 2)
        self.assertAlmostEqual(
            float((out[0, 0] + out[0, 2]) / 2), 84.0, places=0
        )


class TestOutbox(unittest.TestCase):
    def test_retry_and_idempotency(self):
        with tempfile.TemporaryDirectory() as d:
            srv = make_server(18765, os.path.join(d, "s2.db"))
            ob = Outbox(
                os.path.join(d, "ob.db"),
                "http://127.0.0.1:18765/events"
            )
            for i in range(3):
                ob.publish({"event_id": f"e{i}", "event_type": "ENTRY"})
            time.sleep(0.8)
            self.assertEqual(ob.pending(), 3)

            threading.Thread(target=srv.serve_forever, daemon=True).start()
            for _ in range(80):
                if ob.pending() == 0:
                    break
                time.sleep(0.25)
            self.assertEqual(ob.pending(), 0)

            ob.publish({"event_id": "e1", "event_type": "ENTRY"})
            time.sleep(0.5)
            n = srv.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            self.assertEqual(n, 3)
            ob.stop()
            srv.shutdown()
            srv.server_close()
            srv.db.close()
            ob.db.close()


if __name__ == "__main__":
    unittest.main()