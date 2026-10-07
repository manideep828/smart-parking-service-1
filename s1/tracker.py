from collections import Counter, deque

import numpy as np
from scipy.optimize import linear_sum_assignment

HISTORY_S = 120.0


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


class Track:
    def __init__(self, tid, bbox, score, cls, ts):
        self.id = tid
        self.bbox = np.array(bbox, float)
        self.score = float(score)
        self.cls = int(cls)
        self.cls_votes = Counter({int(cls): 1})
        self.first_ts = self.last_ts = ts
        self.hits = 1
        self.vel = np.zeros(2)                    # centre velocity, px/s
        self.history = deque()                    # (ts, cx, bottom_y, width)
        self.confirmed = False
        self._push(ts)

    def _push(self, ts):
        x1, y1, x2, y2 = self.bbox
        self.history.append((ts, (x1 + x2) / 2, y2, x2 - x1))
        while self.history and ts - self.history[0][0] > HISTORY_S:
            self.history.popleft()

    def predict(self, ts):
        dt = min(max(ts - self.last_ts, 0.0), 1.0)
        sx, sy = self.vel * dt
        return self.bbox + np.array([sx, sy, sx, sy])

    def update(self, bbox, score, cls, ts):
        dt = max(ts - self.last_ts, 1e-3)
        old_c = np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])
        new_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.vel = 0.6 * self.vel + 0.4 * (new_c - old_c) / dt
        self.bbox = np.array(bbox, float)
        self.score = float(score)
        self.cls = int(cls)
        self.cls_votes[int(cls)] += 1
        self.last_ts = ts
        self.hits += 1
        self._push(ts)

    @property
    def major_cls(self):
        return self.cls_votes.most_common(1)[0][0]


class ByteTracker:
    """ByteTrack-style two-stage association (high then low confidence
    detections) with constant-velocity prediction, IoU + Hungarian matching.
    Time-based (seconds), so it behaves the same at 3 fps or 10 fps."""

    def __init__(self, high_thr=0.45, low_thr=0.15, match_iou=0.25,
                 low_match_iou=0.15, max_age_s=2.0, min_hits=3):
        self.high_thr, self.low_thr = high_thr, low_thr
        self.match_iou, self.low_match_iou = match_iou, low_match_iou
        self.max_age_s, self.min_hits = max_age_s, min_hits
        self.tracks = []
        self._next = 1

    def _associate(self, preds, tidx, dets, thr):
        if not tidx or len(dets) == 0:
            return [], list(tidx), list(range(len(dets)))
        iou = iou_matrix(preds[tidx], dets[:, :4])
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches, mt, md = [], set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= thr:
                matches.append((tidx[r], c))
                mt.add(tidx[r])
                md.add(c)
        return (matches,
                [t for t in tidx if t not in mt],
                [d for d in range(len(dets)) if d not in md])

    def update(self, dets, ts):
        dets = np.zeros((0, 6)) if dets is None or len(dets) == 0 else np.asarray(dets, float)
        high = dets[dets[:, 4] >= self.high_thr]
        low = dets[(dets[:, 4] >= self.low_thr) & (dets[:, 4] < self.high_thr)]
        preds = np.array([t.predict(ts) for t in self.tracks]) if self.tracks else np.zeros((0, 4))

        m1, un_t, un_d = self._associate(preds, list(range(len(self.tracks))), high, self.match_iou)
        for ti, di in m1:
            d = high[di]
            self.tracks[ti].update(d[:4], d[4], d[5], ts)

        recent = [t for t in un_t if ts - self.tracks[t].last_ts <= 1.0]
        m2, _, _ = self._associate(preds, recent, low, self.low_match_iou)
        for ti, di in m2:
            d = low[di]
            self.tracks[ti].update(d[:4], d[4], d[5], ts)

        for di in un_d:
            d = high[di]
            self.tracks.append(Track(self._next, d[:4], d[4], d[5], ts))
            self._next += 1

        for t in self.tracks:
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True

        removed, alive = [], []
        for t in self.tracks:
            if ts - t.last_ts > self.max_age_s:
                if t.confirmed:
                    removed.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        active = [t for t in self.tracks if t.last_ts == ts and t.confirmed]
        return active, removed
