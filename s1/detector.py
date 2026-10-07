import cv2
import numpy as np


def make_grids(size, strides=(8, 16, 32)):
    grids, strs = [], []
    for s in strides:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        strs.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strs, 0).astype(np.float32)


def decode_outputs(raw, grids, strides):
    """YOLOX raw head output (N, 85) -> pixel-space cx,cy,w,h + obj + cls scores."""
    out = raw.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides
    return out


class YoloxDetector:
    """YOLOX ONNX (Apache-2.0 code) on CPU via ONNX Runtime.

    Returns an (M, 6) array: x1, y1, x2, y2, score, coco_class_id
    in ORIGINAL frame pixel coordinates.
    """

    def __init__(self, model_path=None, input_size=416, score_thr=0.15,
                 nms_thr=0.45, classes=None, threads=2, session=None):
        self.size = input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.allowed = sorted((classes or {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}).keys())
        if session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.sess = session
        self.inp = session.get_inputs()[0].name
        self.grids, self.strides = make_grids(self.size)

    def _preprocess(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(h * r), int(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0-255, no mean/std
        return blob, r

    def detect(self, img):
        blob, r = self._preprocess(img)
        raw = self.sess.run(None, {self.inp: blob})[0][0]
        if raw.shape[0] != self.grids.shape[0]:
            raise RuntimeError(
                f"Model output has {raw.shape[0]} anchors, expected {self.grids.shape[0]}. "
                "Use the official YOLOX ONNX export and a matching input_size."
            )
        out = decode_outputs(raw, self.grids, self.strides)
        scores_all = out[:, 4:5] * out[:, 5:][:, self.allowed]
        ci = scores_all.argmax(1)
        score = scores_all[np.arange(len(ci)), ci]
        keep = score >= self.score_thr
        if not keep.any():
            return np.zeros((0, 6), np.float32)
        out, score, ci = out[keep], score[keep], ci[keep]
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        H, W = img.shape[:2]
        x1 = np.clip((cx - w / 2) / r, 0, W - 1)
        y1 = np.clip((cy - h / 2) / r, 0, H - 1)
        x2 = np.clip((cx + w / 2) / r, 0, W - 1)
        y2 = np.clip((cy + h / 2) / r, 0, H - 1)
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
        idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), score.tolist(), self.score_thr, self.nms_thr)
        idx = np.array(idx).reshape(-1)
        if idx.size == 0:
            return np.zeros((0, 6), np.float32)
        cls = np.array(self.allowed)[ci]
        res = np.stack([x1, y1, x2, y2, score, cls], 1)[idx]
        return res.astype(np.float32)
