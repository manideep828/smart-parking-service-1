import logging

log = logging.getLogger("s2_mapper")

_MAP = {"PARK_START": "ENTRY", "PARK_END": "EXIT"}


class S2Mapper:
    """Maps Service 1 lifecycle events to the Service 2 parking-session contract.
    PARK_START -> ENTRY, PARK_END -> EXIT (parking.slot_id required).
    Gate crossings and other internal events are NOT forwarded."""

    def __init__(self, inner):
        self._inner = inner

    def publish(self, event):
        etype = event.get("event_type")
        slot = (event.get("parking") or {}).get("slot_id")
        if etype not in _MAP:
            log.info("Not sent to Service 2 (internal event): %s event_id=%s", etype, event.get("event_id"))
            return
        if not slot:
            log.warning("Not sent to Service 2: %s without parking.slot_id event_id=%s", etype, event.get("event_id"))
            return
        out = dict(event)
        out["event_type"] = _MAP[etype]
        if not out.get("captured_at"):
            out["captured_at"] = out.get("event_time") or out.get("timestamp")
        self._inner.publish(out)

    def __getattr__(self, name):
        return getattr(self._inner, name)
