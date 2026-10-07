import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger("outbox")


class Outbox:
    """Durable event queue and durable storage for pending parking events."""

    def __init__(self, db_path, url, batch=20, timeout=3.0):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.url, self.batch, self.timeout = url, batch, timeout
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")

        # Existing outbox table: keep its schema and records unchanged.
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                payload TEXT,
                sent INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                next_try REAL DEFAULT 0
            )
        """)

        # Separate durable storage for parking events awaiting a valid plate.
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS pending_parking_events(
                event_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        self.db.commit()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="outbox"
        )
        self._thread.start()

    def publish(self, event):
        """Persist an event in the outgoing queue."""
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO outbox(event_id, payload) VALUES (?,?)",
                (event["event_id"], json.dumps(event)),
            )
            self.db.commit()

    def pending(self):
        """Count unsent outgoing events."""
        with self.lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM outbox WHERE sent=0"
            ).fetchone()[0]

    def save_pending_event(self, event_key, payload):
        """Insert or update an event waiting for plate finalization."""
        serialized = json.dumps(payload)
        with self.lock:
            self.db.execute("""
                INSERT INTO pending_parking_events(event_key, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(event_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
            """, (event_key, serialized, time.time()))
            self.db.commit()

    def load_pending_events(self):
        """Load saved pending events as (event_key, payload) pairs."""
        with self.lock:
            rows = self.db.execute("""
                SELECT event_key, payload
                FROM pending_parking_events
                ORDER BY updated_at, event_key
            """).fetchall()

        events = []
        for event_key, serialized in rows:
            try:
                payload = json.loads(serialized)
                if isinstance(payload, dict):
                    events.append((event_key, payload))
                else:
                    log.error(
                        "Ignoring pending event %s: payload is not an object",
                        event_key,
                    )
            except (TypeError, json.JSONDecodeError):
                log.exception(
                    "Ignoring pending event %s: invalid JSON", event_key
                )
        return events

    def delete_pending_event(self, event_key):
        """Delete a pending event after its outgoing event is persisted."""
        with self.lock:
            self.db.execute(
                "DELETE FROM pending_parking_events WHERE event_key=?",
                (event_key,),
            )
            self.db.commit()

    def pending_event_count(self):
        """Count events waiting for plate finalization."""
        with self.lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM pending_parking_events"
            ).fetchone()[0]

    def _post(self, payload):
        req = urllib.request.Request(
            self.url,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read().decode("utf-8", errors="replace")
                log.info(
                    "Service 2 HTTP %s url=%s response=%s",
                    r.status,
                    self.url,
                    body[:1000],
                )
                return 200 <= r.status < 300

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            log.warning(
                "Service 2 HTTP %s url=%s response=%s",
                e.code,
                self.url,
                body[:1000],
            )
            return e.code == 409

        except Exception:
            log.exception("Service 2 request failed url=%s", self.url)
            return None

    def _run(self):
        while not self._stop.is_set():
            with self.lock:
                rows = self.db.execute(
                    "SELECT id, payload, attempts FROM outbox "
                    "WHERE sent=0 AND next_try<=? ORDER BY id LIMIT ?",
                    (time.time(), self.batch),
                ).fetchall()

            if not rows:
                self._stop.wait(0.2)
                continue

            for rid, payload, attempts in rows:
                if self._stop.is_set():
                    break

                result = self._post(payload)

                with self.lock:
                    if result:
                        self.db.execute(
                            "UPDATE outbox SET sent=1 WHERE id=?", (rid,)
                        )
                    else:
                        delay = min(60.0, 2.0 ** min(attempts + 1, 6))
                        self.db.execute(
                            "UPDATE outbox SET attempts=attempts+1, next_try=? "
                            "WHERE id=?",
                            (time.time() + delay, rid),
                        )
                    self.db.commit()

                if result is None:
                    self._stop.wait(1.0)
                    break

    
    def stop(self, wait=True, timeout=5.0):
        """Request worker shutdown and optionally wait for it to finish."""
        self._stop.set()

        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)

        stopped = not self._thread.is_alive()
        if not stopped:
            log.warning(
                "Outbox worker did not stop within %.1f seconds",
                timeout,
            )
        return stopped