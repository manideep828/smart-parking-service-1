"""Click-to-define the entry tripwire, an 'inside' point, and parking bays.

    python scripts/calibrate.py --source sample.mp4 --level L1 --out config/geometry.json

Steps (instructions are shown on the window):
  1. click 2 points  -> tripwire line across the gate
  2. click 1 point   -> any point INSIDE the garage (decides which side means 'entry')
  3. click 4 points per bay; bays get ids B-01, B-02... (edit the JSON to rename)
Keys: u = undo last point, n = finish this bay early / next, Enter or s = save, q = quit.
"""
import argparse
import json

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="config/geometry.json")
    ap.add_argument("--level", default="L1")
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read a frame from source")
    H, W = img.shape[:2]

    wire, inside, bays, cur = [], [], [], []

    def phase():
        if len(wire) < 2:
            return "1/3 click 2 points: tripwire across the gate"
        if not inside:
            return "2/3 click 1 point INSIDE the garage"
        return f"3/3 click 4 corners of bay B-{len(bays) + 1:02d}  (Enter = save)"

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(wire) < 2:
            wire.append((x, y))
        elif not inside:
            inside.append((x, y))
        else:
            cur.append((x, y))
            if len(cur) == 4:
                bays.append(list(cur))
                cur.clear()

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        v = img.copy()
        if len(wire) == 2:
            cv2.line(v, wire[0], wire[1], (0, 255, 255), 2)
        for p in wire + inside:
            cv2.circle(v, p, 5, (0, 255, 255), -1)
        for i, b in enumerate(bays):
            cv2.polylines(v, [__import__("numpy").array(b)], True, (0, 200, 0), 2)
            cv2.putText(v, f"B-{i + 1:02d}", b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        for p in cur:
            cv2.circle(v, p, 4, (0, 0, 255), -1)
        cv2.putText(v, phase(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.imshow("calibrate", v)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("u"):
            if cur:
                cur.pop()
            elif bays:
                bays.pop()
            elif inside:
                inside.pop()
            elif wire:
                wire.pop()
        if k in (13, ord("s")) and len(wire) == 2 and inside:
            data = {
                "frame_size": [W, H], "tripwire": wire, "inside_point": inside[0], "plate_zone": None,
                "slots": [{"slot_id": f"B-{i + 1:02d}", "level": a.level, "polygon": b}
                          for i, b in enumerate(bays)],
            }
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"saved {a.out}: tripwire, {len(bays)} bays")
            return


if __name__ == "__main__":
    main()
