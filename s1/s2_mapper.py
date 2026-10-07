import logging
import os
import sqlite3

log = logging.getLogger("s2_mapper")

_MAP = {
    "PARK_START": "ENTRY",
    "PARK_END": "EXIT",
}


class S2Mapper:
    """Maps Service 1 parking lifecycle events to the Service 2 contract.

    Internal lifecycle:
        ENTRY      = first vehicle sighting
        PARK_START = vehicle confirmed parked
        PARK_END   = vehicle confirmed gone from slot

    Service 2 receives:
        ENTRY:
            timestamp   = original first-seen time
            captured_at = parking confirmation time
            received_at = null

        EXIT:
            timestamp   = same first-seen time as ENTRY
            captured_at = same parking confirmation time as ENTRY
            received_at = actual leave time

    Lifecycle state is persisted in SQLite so an EXIT can occur in a
    different Service 1 process/run from the ENTRY.
    """

    def __init__(self, inner, db_path="data/lifecycle.db"):
        self._inner = inner
        self._db_path = db_path

        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._db = sqlite3.connect(
            self._db_path,
            timeout=10,
            check_same_thread=False,
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_sessions (
                plate TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                parking_area_id TEXT,
                slot_id TEXT,
                first_seen TEXT NOT NULL,
                parked_at TEXT,
                status TEXT NOT NULL DEFAULT 'OPEN',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (
                    plate,
                    camera_id,
                    parking_area_id
                )
            )
            """
        )
        self._db.commit()

    @staticmethod
    def _plate(event):
        vehicle = event.get("vehicle") or {}
        plate = vehicle.get("plate")

        if not plate:
            plate_obj = event.get("plate") or {}
            plate = plate_obj.get("text")

        return str(plate).strip().upper() if plate else None

    @staticmethod
    def _slot(event):
        return (event.get("parking") or {}).get("slot_id")

    @staticmethod
    def _camera(event):
        return event.get("camera_id") or ""

    @staticmethod
    def _area(event):
        return event.get("parking_area_id")

    def _remember_entry(self, event):
        plate = self._plate(event)

        if not plate:
            log.warning(
                "Cannot remember ENTRY without plate event_id=%s",
                event.get("event_id"),
            )
            return

        first_seen = event.get("timestamp") or event.get("event_time")

        if not first_seen:
            log.warning(
                "Cannot remember ENTRY without timestamp event_id=%s",
                event.get("event_id"),
            )
            return

        camera = self._camera(event)
        area = self._area(event)

        self._db.execute(
            """
            INSERT INTO parking_sessions (
                plate,
                camera_id,
                parking_area_id,
                slot_id,
                first_seen,
                parked_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, NULL, 'OPEN')
            ON CONFLICT (
                plate,
                camera_id,
                parking_area_id
            )
            DO UPDATE SET
                first_seen = excluded.first_seen,
                slot_id = excluded.slot_id,
                parked_at = NULL,
                status = 'OPEN'
            """,
            (
                plate,
                camera,
                area,
                self._slot(event),
                first_seen,
            ),
        )
        self._db.commit()

        log.info(
            "Lifecycle ENTRY remembered plate=%s first_seen=%s slot=%s",
            plate,
            first_seen,
            self._slot(event),
        )

    def _remember_park_start(self, event):
        plate = self._plate(event)

        if not plate:
            log.warning(
                "Cannot remember PARK_START without plate event_id=%s",
                event.get("event_id"),
            )
            return None

        parked_at = event.get("timestamp") or event.get("event_time")
        camera = self._camera(event)
        area = self._area(event)
        slot = self._slot(event)

        row = self._db.execute(
            """
            SELECT first_seen
            FROM parking_sessions
            WHERE plate = ?
              AND camera_id = ?
              AND parking_area_id IS ?
              AND status = 'OPEN'
            """,
            (plate, camera, area),
        ).fetchone()

        if row:
            first_seen = row[0]
        else:
            # Recovery case: if Service 1 starts from a parked vehicle and
            # no earlier ENTRY exists, use the PARK_START time as the best
            # available first-seen time.
            first_seen = event.get("timestamp") or event.get("event_time")

        self._db.execute(
            """
            INSERT INTO parking_sessions (
                plate,
                camera_id,
                parking_area_id,
                slot_id,
                first_seen,
                parked_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'PARKED')
            ON CONFLICT (
                plate,
                camera_id,
                parking_area_id
            )
            DO UPDATE SET
                slot_id = excluded.slot_id,
                parked_at = excluded.parked_at,
                status = 'PARKED'
            """,
            (
                plate,
                camera,
                area,
                slot,
                first_seen,
                parked_at,
            ),
        )
        self._db.commit()

        log.info(
            "Lifecycle PARK_START remembered plate=%s "
            "first_seen=%s captured_at=%s slot=%s",
            plate,
            first_seen,
            parked_at,
            slot,
        )

        return first_seen, parked_at

    def _build_entry(self, event, first_seen, parked_at):
        out = dict(event)

        out["event_type"] = "ENTRY"
        out["timestamp"] = first_seen
        out["captured_at"] = parked_at
        out["received_at"] = None

        return out

    def _build_exit(self, event, first_seen, parked_at):
        out = dict(event)

        received_at = event.get("timestamp") or event.get("event_time")

        out["event_type"] = "EXIT"
        out["timestamp"] = first_seen
        out["captured_at"] = parked_at
        out["received_at"] = received_at

        return out

    def _publish_entry(self, event):
        result = self._remember_park_start(event)

        if not result:
            return

        first_seen, parked_at = result

        out = self._build_entry(
            event,
            first_seen,
            parked_at,
        )

        self._inner.publish(out)

        log.info(
            "S2 ENTRY published plate=%s timestamp=%s captured_at=%s",
            self._plate(out),
            out.get("timestamp"),
            out.get("captured_at"),
        )

    def _publish_exit(self, event):
        plate = self._plate(event)

        if not plate:
            log.warning(
                "Not sent to Service 2: EXIT without plate event_id=%s",
                event.get("event_id"),
            )
            return

        camera = self._camera(event)
        area = self._area(event)
        slot = self._slot(event)

        row = self._db.execute(
            """
            SELECT first_seen, parked_at, slot_id
            FROM parking_sessions
            WHERE plate = ?
              AND camera_id = ?
              AND parking_area_id IS ?
              AND status = 'PARKED'
            """,
            (plate, camera, area),
        ).fetchone()

        if not row:
            log.warning(
                "Not sent to Service 2: EXIT has no stored PARKED session "
                "plate=%s slot=%s event_id=%s",
                plate,
                slot,
                event.get("event_id"),
            )
            return

        first_seen, parked_at, stored_slot = row

        received_at = event.get("timestamp") or event.get("event_time")

        if parked_at and received_at and received_at < parked_at:
            log.warning(
                "EXIT received_at is earlier than captured_at "
                "plate=%s captured_at=%s received_at=%s",
                plate,
                parked_at,
                received_at,
            )

        out = self._build_exit(
            event,
            first_seen,
            parked_at,
        )

        self._inner.publish(out)

        self._db.execute(
            """
            UPDATE parking_sessions
            SET status = 'CLOSED'
            WHERE plate = ?
              AND camera_id = ?
              AND parking_area_id IS ?
              AND status = 'PARKED'
            """,
            (plate, camera, area),
        )
        self._db.commit()

        log.info(
            "S2 EXIT published plate=%s timestamp=%s "
            "captured_at=%s received_at=%s slot=%s",
            plate,
            out.get("timestamp"),
            out.get("captured_at"),
            out.get("received_at"),
            stored_slot,
        )

    def publish(self, event):
        etype = event.get("event_type")

        # ENTRY is internal to S1. Remember it, but don't send it yet.
        if etype == "ENTRY":
            self._remember_entry(event)
            return

        # PARK_START becomes the Service 2 ENTRY.
        if etype == "PARK_START":
            slot = self._slot(event)

            if not slot:
                log.warning(
                    "Not sent to Service 2: PARK_START without "
                    "parking.slot_id event_id=%s",
                    event.get("event_id"),
                )
                return

            self._publish_entry(event)
            return

        # PARK_END becomes the Service 2 EXIT.
        if etype == "PARK_END":
            slot = self._slot(event)

            if not slot:
                log.warning(
                    "Not sent to Service 2: PARK_END without "
                    "parking.slot_id event_id=%s",
                    event.get("event_id"),
                )
                return

            self._publish_exit(event)
            return

        log.info(
            "Not sent to Service 2 (internal event): %s event_id=%s",
            etype,
            event.get("event_id"),
        )

    def stop(self, wait=True, timeout=5.0):
        self._db.close()
        return self._inner.stop(wait=wait, timeout=timeout)

    def __getattr__(self, name):
        return getattr(self._inner, name)

