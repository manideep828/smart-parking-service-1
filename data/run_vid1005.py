import sys, json, traceback, logging
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(name)s: %(message)s")
from s1.config import load_config
from s1.main import run, MemoryPublisher

GEO = {"frame_size":[720,1280],"tripwire":[[0,900],[720,900]],"inside_point":[360,500],"plate_zone":None,"slots":[{"slot_id":"A-06","level":"L1","polygon":[[152,542],[189,539],[193,679],[155,668]]},{"slot_id":"A-07","level":"L1","polygon":[[189,539],[294,516],[284,696],[193,679]]},{"slot_id":"A-08","level":"L1","polygon":[[294,516],[428,516],[421,742],[284,696]]},{"slot_id":"A-09","level":"L1","polygon":[[428,516],[531,495],[539,744],[421,742]]},{"slot_id":"A-10","level":"L1","polygon":[[531,495],[724,435],[737,770],[539,744]]}]}
json.dump(GEO, open("data/geometry_vid1005.json", "w"), indent=1)
VIDEO = r"C:\Users\Admin\Downloads\VID20261005101149.mp4"

class Capture(MemoryPublisher):
    events = []
    def publish(self, e):
        Capture.events.append(e)
        v = e.get("vehicle") or {}
        import json
        print("\n" + "=" * 72)
        print("SERVICE 1 EVENT")
        print("=" * 72)
        print(json.dumps(e, indent=2, ensure_ascii=False, default=str))
        print("=" * 72)
        super().publish(e)

cfg = load_config("config/config.yaml", {"source": VIDEO, "replay": "sync", "geometry_file": "data/geometry_vid1005.json",
                                         "parking": {"park_dwell_s": 4.0}})
print("EFFECTIVE_PARKING_CFG:", cfg["parking"])
try:
    run(cfg, publisher=Capture())
except Exception:
    traceback.print_exc(file=sys.stdout)
print("\n" + "=" * 72)
print("FINAL SERVICE 1 TEST SUMMARY")
print("=" * 72)

for i, e in enumerate(Capture.events, 1):
    v = e.get("vehicle") or {}
    parking = e.get("parking") or {}
    print(f"""
EVENT #{i}
  event_type   : {e.get("event_type")}
  event_id     : {e.get("event_id")}
  plate        : {v.get("plate")}
  vehicle_type : {v.get("type")}
  color        : {v.get("color")}
  slot_id      : {parking.get("slot_id")}
  event_time   : {e.get("event_time") or e.get("timestamp")}
""".rstrip())

print("-" * 72)
print("TOTAL EVENTS :", len(Capture.events))
print("=" * 72)
