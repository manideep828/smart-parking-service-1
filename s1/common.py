import datetime as dt
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    ts: float          # CAPTURE time (epoch seconds), set when the frame left the decoder
    image: np.ndarray  # BGR full-resolution image
    idx: int           # running frame counter from the source


def iso(ts: float) -> str:
    """Epoch seconds -> ISO-8601 IST with milliseconds."""
    return (
        dt.datetime.fromtimestamp(ts, dt.timezone(dt.timedelta(hours=5, minutes=30)))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

