"""Minimal stand-in for Service 2 (dependency-free) so you can test the S1->S2 contract.
Your real S2 should do the same: validate, dedupe on event_id, set ingested_at, store.

    python s2_stub/server.py --port 8000 --db data/s2.db
    curl http://127.0.0.1:8000/events?limit=20
"""
import argparse
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def make_server(port, db_path):
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.execute("""CREATE TABLE IF NOT EXISTS events(
        event_id TEXT PRIMARY KEY, event_type TEXT, vehicle_uid TEXT, camera_id TEXT,
        event_time TEXT, detected_at TEXT, ingested_at REAL, payload TEXT)""")
    db.commit()
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if urlparse(self.path).path != "/events":
                return self._send(404, {"error": "not found"})
            try:
                ev = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                eid = ev["event_id"]
            except Exception:
                return self._send(400, {"error": "bad event"})
            with lock:
                cur = db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
                                 (eid, ev.get("event_type"), ev.get("vehicle_uid"), ev.get("camera_id"),
                                  ev.get("event_time"), ev.get("detected_at"), time.time(), json.dumps(ev)))
                db.commit()
            self._send(201 if cur.rowcount else 409, {"stored": bool(cur.rowcount)})

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if u.path == "/events":
                limit = int(parse_qs(u.query).get("limit", ["50"])[0])
                with lock:
                    rows = db.execute("SELECT payload FROM events ORDER BY ingested_at DESC LIMIT ?",
                                      (limit,)).fetchall()
                return self._send(200, [json.loads(r[0]) for r in rows])
            self._send(404, {"error": "not found"})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.db = db
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="data/s2.db")
    a = ap.parse_args()
    print(f"S2 stub listening on http://127.0.0.1:{a.port}")
    make_server(a.port, a.db).serve_forever()
