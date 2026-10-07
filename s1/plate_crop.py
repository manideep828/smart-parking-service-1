def cut_plate(crop, bbox, pad=0.2):
    """Cut the plate region out of a vehicle crop. bbox = (x1, y1, x2, y2) in crop pixels."""
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        h, w = crop.shape[:2]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        a, b = int(max(0, x1 - px)), int(max(0, y1 - py))
        c, d = int(min(w, x2 + px)), int(min(h, y2 + py))
        if c - a < 8 or d - b < 4:
            return None
        return crop[b:d, a:c].copy()
    except Exception:
        return None
