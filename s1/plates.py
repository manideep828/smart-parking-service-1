import logging
import queue
import re
import threading
import time
from dataclasses import dataclass

log = logging.getLogger("plates")

STATE_CODES = set(
    "AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TG TN TR TS UA UK UP WB".split()
)
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
             "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}
_STD = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{0,3}\d{4}$")
_STD1 = re.compile(r"^[A-Z]{2}\d[A-Z]{1,3}\d{4}$")
_BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _apply(t, types):
    out = []
    for ch, ty in zip(t, types):
        if ty == "D":
            out.append(ch if ch.isdigit() else _TO_DIGIT.get(ch, ch))
        else:
            out.append(ch if ch.isalpha() else _TO_LETTER.get(ch, ch))
    return "".join(out)


def fix_indian(text):
    """Return (corrected_text, format_valid). Corrects look-alike characters
    by position (letters/digits expected at each slot of an Indian plate)."""
    t = normalize(text)
    n = len(t)
    if n in (9, 10) and t[2:4] == "BH":
        f = _apply(t, "DDLLDDDD" + "L" * (n - 8))
        if _BH.match(f):
            return f, True

    best = None
    for r in (2, 1):
        s = n - 6 - r
        if not 1 <= s <= 3:
            continue
        f = _apply(t, "LL" + "D" * r + "L" * s + "DDDD")
        if _STD1.match(f) if r == 1 else _STD.match(f):
            if f[:2] in STATE_CODES and (r == 2 or f[:2] == "DL"):
                subs = sum(a != b for a, b in zip(t, f))
                if best is None or (subs, -r) < best[0]:
                    best = ((subs, -r), f)
    if best:
        return best[1], True
    return t, False


@dataclass
class Consensus:
    text: str
    conf: float
    n_reads: int
    valid: bool


class PlateVoter:
    """Collects reads across frames and votes character by character,
    weighted by OCR confidence. Prefers reads that match Indian plate format."""

    def __init__(self, max_reads=60):
        self.reads = []
        self.max_reads = max_reads

    def add(self, raw_text, conf):
        fixed, valid = fix_indian(raw_text)

        if len(fixed) < 6:
            return

        # Reject structurally invalid standard Indian plates.
        # Standard format:
        #   STATE + 1-2 district digits + 1-3 series letters + 4 digits.
        # Bharat (BH) plates keep their existing validation path.
        if valid and fixed[2:4] != "BH":
            if len(fixed) not in (9, 10):
                return

            state = fixed[:2]
            body = fixed[2:-4]
            suffix = fixed[-4:]

            if state not in STATE_CODES or not suffix.isdigit():
                return

            district_len = 0
            while (
                district_len < min(2, len(body))
                and body[district_len].isdigit()
            ):
                district_len += 1

            series = body[district_len:]

            if (
                district_len not in (1, 2)
                or not (1 <= len(series) <= 3)
                or not series.isalpha()
            ):
                return

        self.reads.append((fixed, float(conf), valid))

        if len(self.reads) > self.max_reads:
            self.reads.pop(0)

    def consensus(self):
        if not self.reads:
            return None
        valid = [r for r in self.reads if r[2]]
        pool = valid if valid else self.reads
        by_len = {}
        for t, c, _ in pool:
            by_len.setdefault(len(t), []).append((t, c))
        group = max(by_len.values(), key=lambda g: sum(c for _, c in g))
        chars, agree = [], []
        for pos in range(len(group[0][0])):
            w = {}
            for t, c in group:
                w[t[pos]] = w.get(t[pos], 0.0) + c
            best = max(w, key=w.get)
            chars.append(best)
            agree.append(w[best] / sum(w.values()))
        mean_conf = sum(c for _, c in group) / len(group)
        text = "".join(chars)
        return Consensus(
            text,
            mean_conf * (sum(agree) / len(agree)),
            len(group),
            fix_indian(text)[1],
        )

    def is_final(self, confirm_reads, confirm_conf):
        c = self.consensus()
        return bool(c and c.valid and c.n_reads >= confirm_reads and c.conf >= confirm_conf)


@dataclass
class PlateRead:
    text: str
    conf: float
    det_conf: float
    bbox: tuple


class PlateReader:
    """Thin wrapper over fast-alpr (plate detector + OCR, ONNX, CPU)."""

    def __init__(self, detector_model, ocr_model):
        from fast_alpr import ALPR
        self.alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)

    def read(self, img_bgr):
        out = []
        for r in self.alpr.predict(img_bgr):
            if r.ocr is None or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            if isinstance(conf, (list, tuple)):
                conf = sum(conf) / max(len(conf), 1)
            bb = r.detection.bounding_box
            out.append(
                PlateRead(
                    r.ocr.text,
                    float(conf),
                    float(r.detection.confidence),
                    (bb.x1, bb.y1, bb.x2, bb.y2),
                )
            )
        return sorted(out, key=lambda p: p.conf * p.det_conf, reverse=True)


class PlateWorker:
    """Runs plate reading off the main loop so tracking never stalls."""

    def __init__(self, reader):
        self.reader = reader
        self.jobs = queue.Queue(maxsize=2)
        self.results = queue.Queue()
        self._stop = threading.Event()
        self._accepting = True
        self._state_lock = threading.Lock()
        self.last_ms = 0.0
        self.thread = threading.Thread(target=self._run, daemon=False, name="plates")
        self.thread.start()

    def submit(self, key, crop):
        with self._state_lock:
            if not self._accepting:
                return False
            try:
                self.jobs.put_nowait((key, crop))
                return True
            except queue.Full:
                return False

    def _run(self):
        while True:
            try:
                item = self.jobs.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue

            try:
                key, crop = item
                t0 = time.perf_counter()
                try:
                    reads = self.reader.read(crop)
                    log.warning(
                        "OCR_RESULT key=%s crop=%sx%s reads=%d %s",
                        key,
                        crop.shape[1],
                        crop.shape[0],
                        len(reads),
                        [
                            (r.text, round(r.conf, 3), round(r.det_conf, 3))
                            for r in reads
                        ],
                    )
                except Exception:
                    log.exception("plate read failed")
                    reads = []
                self.last_ms = (time.perf_counter() - t0) * 1000
                self.results.put((key, reads, crop))
            finally:
                self.jobs.task_done()

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def stop(self, wait=True, timeout=None):
        """Stop accepting new jobs and optionally wait for queued jobs to finish."""
        with self._state_lock:
            self._accepting = False
            self._stop.set()

        if wait:
            self.thread.join(timeout=timeout)
            return not self.thread.is_alive()
        return True