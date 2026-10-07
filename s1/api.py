
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import load_config
from .main import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("s1.api")

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Garage Vision Service 1",
    description="Video upload and live camera processing for Smart Parking.",
    version="1.0.0",
)

_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_status = {
    "state": "idle",
    "source": None,
    "filename": None,
    "job_id": None,
    "error": None,
}


class LiveStartRequest(BaseModel):
    source: str = Field(
        ...,
        description="Camera index or RTSP URL, e.g. rtsp://camera-address/stream",
    )
    replay: str = Field(default="realtime", pattern="^(realtime|sync)$")
    no_plates: bool = False


def _set_status(**changes):
    with _lock:
        _status.update(changes)


def _worker_run(source: str, replay: str, no_plates: bool, job_id: str, filename=None):
    global _worker

    _set_status(
        state="starting",
        source=source,
        filename=filename,
        job_id=job_id,
        error=None,
    )
    log.info(
        "[JOB %s] Starting processing | filename=%s | source=%s | replay=%s | plates_enabled=%s",
        job_id,
        filename or "(live source)",
        source,
        replay,
        not no_plates,
    )

    try:
        overrides = {"source": source, "replay": replay}
        if no_plates:
            overrides["plate"] = {"enabled": False}

        cfg = load_config(str(ROOT / "config" / "config.yaml"), overrides)
        _set_status(state="running")
        log.info("[JOB %s] Pipeline started", job_id)

        run(cfg, show=False)

        _set_status(state="completed")
        log.info("[JOB %s] Pipeline completed successfully", job_id)

    except Exception as exc:
        log.exception("[JOB %s] Processing failed: %s", job_id, exc)
        _set_status(state="error", error=str(exc))

    finally:
        with _lock:
            if _worker is threading.current_thread():
                _worker = None
        log.info("[JOB %s] Worker stopped; final state=%s", job_id, _status["state"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Garage Vision Service 1",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/api/status", tags=["Processing"])
def get_status():
    with _lock:
        result = dict(_status)
        worker = _worker

    result["worker_alive"] = worker is not None and worker.is_alive()
    return result


def _start_worker(source: str, replay: str, no_plates: bool, filename=None):
    global _worker

    with _lock:
        if _worker is not None and _worker.is_alive():
            raise HTTPException(
                status_code=409,
                detail="A processing job is already running. Stop it before starting another.",
            )

        job_id = uuid.uuid4().hex
        _status.update(
            state="starting",
            source=source,
            filename=filename,
            job_id=job_id,
            error=None,
        )

        _worker = threading.Thread(
            target=_worker_run,
            args=(source, replay, no_plates, job_id, filename),
            daemon=True,
            name=f"s1-job-{job_id[:8]}",
        )
        _worker.start()

    log.info("[JOB %s] Worker thread created", job_id)
    return {"job_id": job_id, "state": "starting", "source": source, "filename": filename}


@app.post("/api/upload", status_code=202, tags=["Video Upload"])
async def upload_video(
    file: UploadFile = File(..., description="Video file to process"),
    replay: str = "realtime",
    no_plates: bool = False,
):
    if replay not in ("realtime", "sync"):
        raise HTTPException(status_code=400, detail="replay must be 'realtime' or 'sync'")

    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension. Allowed: {', '.join(sorted(allowed))}",
        )

    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    log.info("[UPLOAD] Received filename=%s", filename)

    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        log.exception("[UPLOAD] Failed to save filename=%s", filename)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded video: {exc}") from exc
    finally:
        await file.close()

    log.info(
        "[UPLOAD] Saved filename=%s | path=%s | bytes=%s",
        filename,
        destination,
        destination.stat().st_size,
    )

    try:
        result = _start_worker(str(destination), replay, no_plates, filename)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    return result


@app.post("/api/live/start", status_code=202, tags=["Live Camera"])
def start_live(request: LiveStartRequest):
    source = request.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Camera source is required")
    if source.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Use an RTSP URL or camera index for live video, not an HTTP page URL.",
        )
    return _start_worker(source, request.replay, request.no_plates)


@app.post("/api/live/stop", tags=["Live Camera"])
def stop_live():
    with _lock:
        worker = _worker
        state = _status["state"]

    if worker is None or not worker.is_alive():
        return {"state": state, "message": "No active processing job"}

    return {
        "state": "stop_not_supported_yet",
        "message": (
            "The current pipeline does not expose a safe stop signal. "
            "The active job is still running; restart the API process to terminate it."
        ),
    }


@app.get("/api/video/uploaded", tags=["Video Upload"])
def uploaded_videos():
    return {
        "files": [
            {"filename": path.name, "size_bytes": path.stat().st_size}
            for path in sorted(UPLOAD_DIR.iterdir())
            if path.is_file()
        ]
    }


@app.get("/api/frames/latest", tags=["Processed Output"])
def latest_frame():
    raise HTTPException(
        status_code=501,
        detail="Processed-frame output is not implemented yet.",
    )


@app.get("/api/output/stream", tags=["Processed Output"])
def output_stream():
    raise HTTPException(
        status_code=501,
        detail="Live processed-video streaming is not implemented yet.",
    )