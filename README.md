# Garage Vision Service (S1)

Reads a camera stream, finds vehicles, tracks them, reads plates, works out
entry/exit and parking-bay events, and sends JSON events to S2.
Runs on CPU (tuned for a 2-core i5-5200U class machine).

```
camera/video -> grabber (latest frame wins) -> YOLOX-nano ONNX -> ByteTrack-style tracker
   -> per-vehicle: plate (fast-alpr, background thread, multi-read voting, Indian format rules)
                   colour, type, tripwire direction, bay state machine
   -> event builder -> SQLite outbox (survives crashes) -> HTTP POST -> S2
```

## Setup (Windows / Linux / Mac)

```
python --version                      # needs 3.10+
python -m venv venv
venv\Scripts\activate                 # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py     # vehicle detector (fast-alpr downloads its own models on first run)
```

## First run

1. Put a garage clip next to the project, e.g. `sample.mp4`.
2. Draw the gate line and bays (click on the first frame):
   `python scripts/calibrate.py --source sample.mp4 --level L1 --out config/geometry.json`
3. Start the stand-in S2 in a second terminal: `python s2_stub/server.py`
4. Speed test first, without plates: `python -m s1.main --source sample.mp4 --no-plates --show`
5. Then everything: `python -m s1.main --source sample.mp4 --show`
6. See what S2 stored: open http://127.0.0.1:8000/events

The log prints every 5 s: `fps`, `detect` ms, `total` ms, and `capture->event-ready` latency.
Those are YOUR real numbers. If `total` is above ~200 ms, lower `detector.max_fps`
or use a smaller `input_size` in config.

`--replay sync` processes every frame in order (deterministic tuning).
`--replay realtime` (default for files) drops frames like a live camera would.
For a live camera: `--source rtsp://user:pass@ip:554/stream` (use the camera's low-res substream).

## Tests

`python -m unittest discover -s tests -v`
The end-to-end tests drive the real tracker, tripwire, bay logic, plate voting and event builder with a
synthetic garage (two cars at once, a car hidden while parked, a car just passing through).

## Event contract (S1 -> S2)

`event_type`: ENTRY | EXIT | PARK_START | PARK_END | VEHICLE_UPDATE
- `event_time`   when it happened on camera (crossing time, or backdated to when the car stopped / started moving)
- `detected_at`  when S1 produced the event.  S2 should add its own `ingested_at`.
- `event_id`     idempotency key. S2 MUST ignore duplicates (return 200/201/409).
- `inferred`     true if the outcome was inferred (e.g. track lost for 30 s while parked)
- ENTRY/EXIT wait up to `plate.emit_deadline_s` for a confirmed plate. If the plate is confirmed later with a
  different value, S1 sends a VEHICLE_UPDATE.
- Events can arrive out of order (PARK_END is backdated). **S2 should sort by `event_time`, not arrival.**

## Known limits of v1 (be aware)

- Plate OCR is the pretrained global model: expect misreads on Indian plates until it is fine-tuned.
- Colour is a pixel-vote heuristic; type is the detector class (car/bus/truck/motorcycle), not sedan/SUV.
- Timestamps use the moment the frame leaves the decoder (not the camera's RTP clock). Sync camera NTP anyway.
- One camera per process. Bays must be visible in the same camera, or plates will not follow the car to a bay camera.
- If a parked car's track is lost and the car leaves before being re-acquired for park_dwell_s, the bay is released
  after lost_grace_s with an `inferred` PARK_END.
- Check the licence of every model before commercial use.
