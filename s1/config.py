import copy

import yaml

DEFAULTS = {
    "camera_id": "gate-cam-01",
    "source": "sample.mp4",
    "replay": "realtime",          # realtime = drop frames like a live camera | sync = every frame
    "replay_start_time": None,     # epoch seconds for file replay; None = now
    "geometry_file": "config/geometry.json",
    "detector": {
        "model_path": "models/yolox_nano.onnx",
        "input_size": 416,
        "score_thr": 0.15,
        "nms_thr": 0.45,
        "classes": {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
        "max_fps": 5,
        "threads": 2,
    },
    "tracker": {
        "high_thr": 0.45, "low_thr": 0.15,
        "match_iou": 0.25, "low_match_iou": 0.15,
        "max_age_s": 2.0, "min_hits": 3,
    },
    "gate": {"margin_px": 12, "size_samples": 3, "min_size_change": 0.15},
    "color": {"interval_s": 1.0, "max_samples": 8, "min_box_h": 60},
    "plate": {
        "enabled": True,
        "min_vehicle_height_px": 100,
        "interval_s": 0.3,
        "detector_model": "yolo-v9-t-384-license-plate-end2end",
        "ocr_model": "cct-xs-v2-global-model",
        "min_ocr_conf": 0.5,
        "confirm_reads": 3,
        "confirm_conf": 0.75,
        "max_attempts": 40,
        "emit_deadline_s": 3.0,
    },
    "parking": {
        "footprint_frac": 0.25,
        "min_overlap": 0.3,
        "leave_overlap": 0.3,
        "park_dwell_s": 4.0,
        "leave_dwell_s": 3.0,
        "still_frac": 0.12,
        "moved_frac": 0.35,
        "gone_frac": 1.0,
        "lost_grace_s": 30.0,
        "baseline_window_s": 2.0,
        "publish_exit": False,
    },
    "publisher": {
        "s2_url": "http://127.0.0.1:8000/events",
        "outbox_db": "data/outbox.db",
        "snapshot_dir": "data/snapshots",
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    data = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return _merge(_merge(DEFAULTS, data), overrides or {})


