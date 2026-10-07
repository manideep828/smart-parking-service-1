import hashlib
import json
import logging
import os

from .outbox import Outbox

log = logging.getLogger("uploader")


class SnapshotUploader:
    """Uploads local snapshot files to S3-compatible storage and returns stable HTTPS URLs.
    Configured ONLY from environment variables (never from files):
      SNAPSHOT_STORAGE = off (default) | s3
      S3_BUCKET, S3_PUBLIC_BASE_URL (https://...), optional S3_ENDPOINT_URL, S3_REGION, S3_PREFIX
      credentials: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"""

    def __init__(self, enabled=False, bucket=None, public_base=None, prefix="snapshots", client=None):
        self.enabled = enabled
        self.bucket = bucket
        self.public_base = (public_base or "").rstrip("/")
        self.prefix = prefix.strip("/")
        self.client = client

    @classmethod
    def from_env(cls):
        if os.getenv("SNAPSHOT_STORAGE", "off").lower() != "s3":
            log.info("Snapshot upload OFF: local file paths are sent (development only)")
            return cls(enabled=False)
        bucket = os.getenv("S3_BUCKET")
        base = os.getenv("S3_PUBLIC_BASE_URL") or ""
        if not bucket or not base.startswith("https://"):
            raise SystemExit("SNAPSHOT_STORAGE=s3 needs S3_BUCKET and an https:// S3_PUBLIC_BASE_URL")
        import boto3
        client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
                              region_name=os.getenv("S3_REGION") or None)
        log.info("Snapshot upload ON: bucket=%s", bucket)
        return cls(True, bucket, base, os.getenv("S3_PREFIX", "snapshots"), client)

    def upload_file(self, path, event_id, kind):
        digest = hashlib.sha256((event_id + ":" + kind).encode()).hexdigest()[:32]
        key = "%s/%s.jpg" % (self.prefix, digest)
        self.client.upload_file(path, self.bucket, key, ExtraArgs={"ContentType": "image/jpeg"})
        return "%s/%s" % (self.public_base, key)

    def rewrite_event(self, ev):
        """Replace local snapshot paths with HTTPS URLs. Raises if an upload fails (caller retries)."""
        snap = ev.get("snapshot")
        if not isinstance(snap, dict):
            return ev
        for kind in ("vehicle", "plate"):
            val = snap.get(kind)
            if not val or str(val).startswith("https://"):
                continue
            if not os.path.isfile(val):
                log.warning("Snapshot file missing, sending null for %s: %s", kind, val)
                snap[kind] = None
                continue
            snap[kind] = self.upload_file(val, ev["event_id"], kind)
        return ev


class UploadingOutbox(Outbox):
    """Outbox that uploads snapshots in ITS OWN worker thread right before sending, so slow uploads
    never block the video loop. If an upload fails the event is retried; with upload ON it is never
    sent with a local path."""

    def __init__(self, *args, uploader=None, **kwargs):
        self.uploader = uploader or SnapshotUploader.from_env()
        super().__init__(*args, **kwargs)

    def _post(self, payload):
        if self.uploader.enabled:
            try:
                ev = self.uploader.rewrite_event(json.loads(payload))
                payload = json.dumps(ev)
            except Exception:
                log.exception("Snapshot upload failed; will retry")
                return None
        return super()._post(payload)
