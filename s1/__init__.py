import os

# Low-latency RTSP over TCP. Must be set before OpenCV opens a stream.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)
__version__ = "0.1.0"
