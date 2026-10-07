import os
import tempfile
import unittest

import numpy as np

from s1.config import load_config
from s1.geometry import Geometry
from s1.outbox import Outbox
from s1.pipeline import Pipeline
from s1.vehicles import Vehicle


W, H = 640, 480
GEO = {
    "frame_size": [W, H],
    "tripwire": [[0, 100], [640, 100]],
    "inside_point": [320, 300],
    "plate_zone": None,
    "slots": [],
}


class EmptyDetector:
    def detect(self, image):
        return np.zeros((0, 6), dtype=float)


class TestPendingRecovery(unittest.TestCase):
    def make_config(self):
        return load_config(None, {
            "parking_area_id": None,
            "publisher": {"snapshot_dir": None},
        })

    def make_pipeline(self, cfg, outbox, now):
        return Pipeline(
            cfg,
            Geometry(GEO),
            (W, H),
            EmptyDetector(),
            outbox,
            plate_worker=None,
            now_fn=lambda: now,
        )

    def make_pending(self, event_id, vehicle_uid, finalized=True):
        vehicle = Vehicle(
            uid=vehicle_uid,
            track_id=7,
            first_ts=1800000000.0,
        )
        vehicle.cls_votes[2] = 5
        vehicle.color_votes["red"] = 3

        if finalized:
            vehicle.voter.add("KA01AB1234", 0.99)
            vehicle.plate_final = True

        return {
            "event_id": event_id,
            "event_type": "ENTRY",
            "vehicle": vehicle,
            "event_time": 1800000001.0,
            "detected_at": 1800000002.0,
            "direction": "entry",
            "slot": None,
            "inferred": False,
            "created_at": 1800000002.0,
            "deadline": 1800000030.0,
            "timed_out": False,
            "recovery_crop_retry": False,
            "recovery_retry_due": False,
        }

    def test_pending_event_restored_and_queued_after_restart(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "outbox.db")
            cfg = self.make_config()

            first_outbox = Outbox(
                db_path, "http://127.0.0.1:1/events"
            )
            try:
                first_pipeline = self.make_pipeline(
                    cfg, first_outbox, 1800000010.0
                )
                pending = self.make_pending(
                    "restart-event-1", "restart-vehicle-1"
                )

                first_pipeline.pending_parking_events.append(pending)
                self.assertTrue(first_pipeline._save_pending_event(pending))
                self.assertEqual(first_outbox.pending_event_count(), 1)
            finally:
                first_outbox.stop()
                first_outbox.db.close()

            second_outbox = Outbox(
                db_path, "http://127.0.0.1:1/events"
            )
            try:
                second_pipeline = self.make_pipeline(
                    cfg, second_outbox, 1800000011.0
                )

                self.assertEqual(
                    len(second_pipeline.pending_parking_events), 1
                )
                restored = second_pipeline.pending_parking_events[0]
                self.assertEqual(restored["event_id"], "restart-event-1")
                self.assertEqual(
                    restored["vehicle"].uid, "restart-vehicle-1"
                )

                outgoing = []
                second_pipeline._release_ready_parking_events(outgoing)

                self.assertEqual(len(outgoing), 1)
                self.assertEqual(outgoing[0]["event_id"], "restart-event-1")

                second_pipeline._publish_event(outgoing[0])

                self.assertEqual(second_outbox.pending_event_count(), 0)
                with second_outbox.lock:
                    row = second_outbox.db.execute(
                        "SELECT event_id, sent FROM outbox WHERE event_id=?",
                        ("restart-event-1",),
                    ).fetchone()

                self.assertIsNotNone(row)
                self.assertEqual(row[0], "restart-event-1")
            finally:
                second_outbox.stop()
                second_outbox.db.close()

    def test_flush_persists_unresolved_event_for_restart(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "outbox.db")
            cfg = self.make_config()

            first_outbox = Outbox(
                db_path, "http://127.0.0.1:1/events"
            )
            try:
                first_pipeline = self.make_pipeline(
                    cfg, first_outbox, 1800000010.0
                )
                pending = self.make_pending(
                    "flush-event-1",
                    "flush-vehicle-1",
                    finalized=False,
                )
                first_pipeline.pending_parking_events.append(pending)

                # Simulate orderly shutdown while the plate is unresolved.
                first_pipeline.flush(1800000010.0)

                self.assertEqual(first_outbox.pending_event_count(), 1)
            finally:
                first_outbox.stop()
                first_outbox.db.close()

            second_outbox = Outbox(
                db_path, "http://127.0.0.1:1/events"
            )
            try:
                second_pipeline = self.make_pipeline(
                    cfg, second_outbox, 1800000011.0
                )

                self.assertEqual(
                    len(second_pipeline.pending_parking_events), 1
                )
                restored = second_pipeline.pending_parking_events[0]
                self.assertEqual(restored["event_id"], "flush-event-1")
                self.assertEqual(
                    restored["vehicle"].uid, "flush-vehicle-1"
                )
                self.assertFalse(restored["vehicle"].plate_final)
                self.assertEqual(second_outbox.pending_event_count(), 1)
            finally:
                second_outbox.stop()
                second_outbox.db.close()


if __name__ == "__main__":
    unittest.main()
