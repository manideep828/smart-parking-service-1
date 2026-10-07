from collections import Counter

import cv2
import numpy as np

TYPE_BY_COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def classify_pixels(hsv):
    H, S, V = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    names = np.empty(len(hsv), dtype=object)
    names[:] = "other"
    names[(H >= 130) & (H < 170)] = "purple"
    names[(H >= 85) & (H < 130)] = "blue"
    names[(H >= 35) & (H < 85)] = "green"
    names[(H >= 20) & (H < 35)] = "yellow"
    orange = (H >= 8) & (H < 20)
    names[orange] = "orange"
    names[orange & (V < 140)] = "brown"
    names[(H < 8) | (H >= 170)] = "red"
    names[S < 50] = "gray"
    names[(S < 50) & (V > 140)] = "silver"
    names[(S < 40) & (V > 190)] = "white"
    names[V < 60] = "black"
    return names


def dominant_color(crop_bgr):
    """v1 heuristic: pixel-vote in the vehicle body region. Replace with a small
    CNN trained on your garage footage once you have labelled data."""
    h, w = crop_bgr.shape[:2]
    roi = crop_bgr[int(h * 0.35):int(h * 0.85), int(w * 0.2):int(w * 0.8)]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    votes = Counter(classify_pixels(hsv))
    votes.pop("other", None)
    return votes.most_common(1)[0][0] if votes else None
