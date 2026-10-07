import logging
import os
import threading
import time

import cv2

from .common import Frame

log = logging.getLogger("grabber")


def _open(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return cv2.VideoCapture(source)


class LatestFrameGrabber:
    """Reads in a background thread and keeps ONLY the newest frame.

    Latency can never build up: if processing is slower than the camera,
    old frames are dropped. Timestamps are taken when the frame leaves the
    decoder, not when it is processed or stored.
    """

    def __init__(self, source, paced=False, start_time=None):
        self.source = source
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.paced = paced and self.is_file
        self.start_time = start_time
        self._cond = threading.Condition()
        self._frame = None
        self._count = 0
        self._stop = threading.Event()
        self.eof = False
        self.last_frame_wall = 0.0
        self.reconnects = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="grabber")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cap = _open(self.source)
            if not cap.isOpened():
                log.warning("cannot open %s, retry in %.0fs", self.source, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)
                self.reconnects += 1
                continue
            backoff = 1.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            wall0 = time.time()
            base = self.start_time if self.start_time is not None else wall0
            n = 0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                if self.is_file:
                    ts = time.time()
                    if self.paced:
                        delay = wall0 + n / fps - time.time()
                        if delay > 0:
                            time.sleep(delay)
                else:
                    ts = time.time()
                n += 1
                self.last_frame_wall = time.time()
                with self._cond:
                    self._frame = Frame(ts, img, self._count)
                    self._count += 1
                    self._cond.notify_all()
            cap.release()
            if self.is_file:
                with self._cond:
                    self.eof = True
                    self._cond.notify_all()
                return
            self.reconnects += 1
            log.warning("stream dropped, reconnecting")
            time.sleep(1.0)

    def read(self, last_idx=-1, timeout=1.0):
        """Block until a frame newer than last_idx exists. None on timeout/EOF."""
        with self._cond:
            self._cond.wait_for(
                lambda: (self._frame is not None and self._frame.idx > last_idx)
                or self.eof or self._stop.is_set(),
                timeout,
            )
            if self._frame is not None and self._frame.idx > last_idx:
                return self._frame
            return None


class SequentialFileSource:
    """Every frame, in order, no dropping. For deterministic tuning runs."""

    def __init__(self, path, start_time=None):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.base = start_time if start_time is not None else time.time()
        self.n = 0
        self.eof = False

    def start(self):
        return self

    def stop(self):
        self.cap.release()

    def read(self, last_idx=-1, timeout=1.0):
        ok, img = self.cap.read()
        if not ok:
            self.eof = True
            return None
        f = Frame(self.base + self.n / self.fps, img, self.n)
        self.n += 1
        return f


def open_source(cfg):
    src = cfg["source"]
    start = cfg.get("replay_start_time")
    is_file = isinstance(src, str) and os.path.isfile(src)
    if is_file and cfg.get("replay") == "sync":
        return SequentialFileSource(src, start).start()
    return LatestFrameGrabber(src, paced=True, start_time=start).start()
